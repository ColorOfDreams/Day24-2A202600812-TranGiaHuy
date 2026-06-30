from __future__ import annotations

"""Module 2: Hybrid Search — BM25 (Vietnamese) + Dense + RRF."""

import os, sys, re, math
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (QDRANT_HOST, QDRANT_PORT, COLLECTION_NAME, EMBEDDING_MODEL,
                    EMBEDDING_DIM, BM25_TOP_K, DENSE_TOP_K, HYBRID_TOP_K)


def _model_is_cached(model_id: str) -> bool:
    cache = os.getenv("HF_HUB_CACHE", os.path.expanduser("~/.cache/huggingface/hub"))
    return os.path.isdir(os.path.join(cache, "models--" + model_id.replace("/", "--")))


@dataclass
class SearchResult:
    text: str
    score: float
    metadata: dict
    method: str  # "bm25", "dense", "hybrid"


class _LocalBM25:
    """Small BM25Okapi-compatible fallback for environments without rank_bm25."""

    def __init__(self, corpus: list[list[str]], k1: float = 1.5, b: float = 0.75):
        self.corpus = corpus
        self.k1, self.b = k1, b
        self.avgdl = sum(map(len, corpus)) / len(corpus) if corpus else 0.0
        self.document_frequency: dict[str, int] = {}
        for document in corpus:
            for token in set(document):
                self.document_frequency[token] = self.document_frequency.get(token, 0) + 1

    def get_scores(self, query: list[str]) -> list[float]:
        scores = []
        total_docs = len(self.corpus)
        for document in self.corpus:
            frequencies: dict[str, int] = {}
            for token in document:
                frequencies[token] = frequencies.get(token, 0) + 1
            score = 0.0
            for token in query:
                df = self.document_frequency.get(token, 0)
                if not df:
                    continue
                idf = math.log(1 + (total_docs - df + 0.5) / (df + 0.5))
                tf = frequencies.get(token, 0)
                denominator = tf + self.k1 * (1 - self.b + self.b * len(document) / (self.avgdl or 1))
                score += idf * (tf * (self.k1 + 1) / denominator) if denominator else 0.0
            scores.append(score)
        return scores


def segment_vietnamese(text: str) -> str:
    """Segment Vietnamese text into words."""
    try:
        from underthesea import word_tokenize
        return word_tokenize(text, format="text").replace("_", " ")
    except Exception:
        # Whitespace/punctuation normalisation still gives BM25 a useful
        # dependency-free fallback in minimal environments.
        return " ".join(re.findall(r"\w+", text.lower(), flags=re.UNICODE))


class BM25Search:
    def __init__(self):
        self.corpus_tokens = []
        self.documents = []
        self.bm25 = None

    def index(self, chunks: list[dict]) -> None:
        """Build BM25 index from chunks."""
        self.documents = list(chunks)
        self.corpus_tokens = [segment_vietnamese(c.get("text", "")).split() for c in self.documents]
        if not self.corpus_tokens:
            self.bm25 = None
            return
        try:
            from rank_bm25 import BM25Okapi
            self.bm25 = BM25Okapi(self.corpus_tokens)
        except ImportError:
            self.bm25 = _LocalBM25(self.corpus_tokens)

    def search(self, query: str, top_k: int = BM25_TOP_K) -> list[SearchResult]:
        """Search using BM25."""
        if self.bm25 is None or top_k <= 0:
            return []
        scores = self.bm25.get_scores(segment_vietnamese(query).split())
        ranked_indices = sorted(range(len(scores)), key=lambda i: float(scores[i]), reverse=True)
        return [
            SearchResult(
                text=self.documents[i].get("text", ""),
                score=float(scores[i]),
                metadata=dict(self.documents[i].get("metadata", {})),
                method="bm25",
            )
            for i in ranked_indices[:top_k]
            if scores[i] > 0
        ]


class DenseSearch:
    def __init__(self):
        try:
            from qdrant_client import QdrantClient
            self.client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
        except ImportError:
            self.client = None
        self._encoder = None
        self._fallback_documents: list[dict] = []

    def _get_encoder(self):
        if self._encoder is None:
            if not _model_is_cached(EMBEDDING_MODEL):
                raise FileNotFoundError(f"Embedding model {EMBEDDING_MODEL} is not cached locally")
            from sentence_transformers import SentenceTransformer
            self._encoder = SentenceTransformer(EMBEDDING_MODEL, local_files_only=True)
        return self._encoder

    def index(self, chunks: list[dict], collection: str = COLLECTION_NAME) -> None:
        """Index chunks into Qdrant."""
        self._fallback_documents = list(chunks)
        if not chunks:
            return
        try:
            if self.client is None:
                raise RuntimeError("qdrant-client is not installed")
            from qdrant_client.models import Distance, PointStruct, VectorParams

            vectors = self._get_encoder().encode(
                [c.get("text", "") for c in chunks], show_progress_bar=False
            )
            # recreate_collection is intentionally used here: each pipeline run
            # must index exactly the current corpus, not stale vectors.
            self.client.recreate_collection(
                collection, vectors_config=VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE)
            )
            points = [
                PointStruct(
                    id=i,
                    vector=vector.tolist(),
                    payload={**chunk.get("metadata", {}), "text": chunk.get("text", "")},
                )
                for i, (chunk, vector) in enumerate(zip(chunks, vectors))
            ]
            self.client.upsert(collection_name=collection, points=points, wait=True)
        except Exception as exc:
            print(f"  Dense search fallback enabled: {exc}")

    def search(self, query: str, top_k: int = DENSE_TOP_K, collection: str = COLLECTION_NAME) -> list[SearchResult]:
        """Search using dense vectors."""
        if top_k <= 0:
            return []
        try:
            if self.client is None:
                raise RuntimeError("qdrant-client is not installed")
            query_vector = self._get_encoder().encode(query).tolist()
            response = self.client.query_points(collection_name=collection, query=query_vector, limit=top_k)
            return [
                SearchResult(
                    text=str(point.payload.get("text", "")),
                    score=float(point.score),
                    metadata={k: v for k, v in point.payload.items() if k != "text"},
                    method="dense",
                )
                for point in response.points
            ]
        except Exception:
            # Token-overlap fallback mirrors the ranking contract, allowing the
            # pipeline to remain functional if Qdrant/model setup is absent.
            query_tokens = set(segment_vietnamese(query).split())
            scored = []
            for chunk in self._fallback_documents:
                tokens = set(segment_vietnamese(chunk.get("text", "")).split())
                score = len(query_tokens & tokens) / (len(query_tokens | tokens) or 1)
                if score > 0:
                    scored.append((score, chunk))
            scored.sort(key=lambda item: item[0], reverse=True)
            return [
                SearchResult(c.get("text", ""), score, dict(c.get("metadata", {})), "dense")
                for score, c in scored[:top_k]
            ]


def reciprocal_rank_fusion(results_list: list[list[SearchResult]], k: int = 60,
                           top_k: int = HYBRID_TOP_K) -> list[SearchResult]:
    """Merge ranked lists using RRF: score(d) = Σ 1/(k + rank)."""
    if top_k <= 0:
        return []
    scores: dict[str, dict] = {}
    for result_list in results_list:
        for rank, result in enumerate(result_list):
            item = scores.setdefault(result.text, {"score": 0.0, "result": result})
            item["score"] += 1.0 / (k + rank + 1)
    ranked = sorted(scores.values(), key=lambda item: item["score"], reverse=True)[:top_k]
    return [
        SearchResult(
            text=item["result"].text,
            score=float(item["score"]),
            metadata=dict(item["result"].metadata),
            method="hybrid",
        )
        for item in ranked
    ]


class HybridSearch:
    """Combines BM25 + Dense + RRF. (Đã implement sẵn — dùng classes ở trên)"""
    def __init__(self):
        self.bm25 = BM25Search()
        self.dense = DenseSearch()

    def index(self, chunks: list[dict]) -> None:
        self.bm25.index(chunks)
        self.dense.index(chunks)

    def search(self, query: str, top_k: int = HYBRID_TOP_K) -> list[SearchResult]:
        bm25_results = self.bm25.search(query, top_k=BM25_TOP_K)
        dense_results = self.dense.search(query, top_k=DENSE_TOP_K)
        return reciprocal_rank_fusion([bm25_results, dense_results], top_k=top_k)


if __name__ == "__main__":
    print(f"Original:  Nhân viên được nghỉ phép năm")
    print(f"Segmented: {segment_vietnamese('Nhân viên được nghỉ phép năm')}")
