from __future__ import annotations

"""Module 3: Reranking — Cross-encoder top-20 → top-3 + latency benchmark."""

import os, sys, time, re
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import RERANK_TOP_K


def _model_is_cached(model_id: str) -> bool:
    cache = os.getenv("HF_HUB_CACHE", os.path.expanduser("~/.cache/huggingface/hub"))
    return os.path.isdir(os.path.join(cache, "models--" + model_id.replace("/", "--")))


@dataclass
class RerankResult:
    text: str
    original_score: float
    rerank_score: float
    metadata: dict
    rank: int


class CrossEncoderReranker:
    def __init__(self, model_name: str = "BAAI/bge-reranker-v2-m3"):
        self.model_name = model_name
        self._model = None

    def _load_model(self):
        if self._model is None:
            try:
                if not _model_is_cached(self.model_name):
                    raise FileNotFoundError(f"Cross-encoder {self.model_name} is not cached locally")
                from sentence_transformers import CrossEncoder
                # Do not make unit tests or offline development wait for a
                # network download.  The pre-downloaded production model is
                # still used automatically when present in the local cache.
                self._model = CrossEncoder(self.model_name, local_files_only=True)
            except Exception as exc:
                print(f"  Cross-encoder fallback enabled: {exc}")
                self._model = _LexicalCrossEncoder()
        return self._model

    def rerank(self, query: str, documents: list[dict], top_k: int = RERANK_TOP_K) -> list[RerankResult]:
        """Rerank documents: top-20 → top-k."""
        if not documents or top_k <= 0:
            return []
        model = self._load_model()
        scores = model.predict([(query, doc.get("text", "")) for doc in documents])
        if isinstance(scores, (int, float)):
            scores = [scores]
        scored = sorted(zip(scores, documents), key=lambda item: float(item[0]), reverse=True)
        return [
            RerankResult(
                text=doc.get("text", ""),
                original_score=float(doc.get("score", 0.0)),
                rerank_score=float(score),
                metadata=dict(doc.get("metadata", {})),
                rank=rank,
            )
            for rank, (score, doc) in enumerate(scored[:top_k])
        ]


class _LexicalCrossEncoder:
    """Small deterministic fallback with the same ``predict`` interface."""

    def predict(self, pairs: list[tuple[str, str]]) -> list[float]:
        values = []
        for query, document in pairs:
            query_terms = set(re.findall(r"\w+", query.lower(), flags=re.UNICODE))
            document_terms = set(re.findall(r"\w+", document.lower(), flags=re.UNICODE))
            # Recall-oriented scoring works well as a safe fallback: a short
            # relevant answer should not be penalised just for being short.
            overlap = len(query_terms & document_terms)
            values.append(overlap / (len(query_terms) or 1))
        return values


class FlashrankReranker:
    """Lightweight alternative (<5ms). Optional."""
    def __init__(self):
        self._model = None

    def rerank(self, query: str, documents: list[dict], top_k: int = RERANK_TOP_K) -> list[RerankResult]:
        if not documents or top_k <= 0:
            return []
        try:
            from flashrank import Ranker, RerankRequest
            if self._model is None:
                self._model = Ranker()
            ranked = self._model.rerank(RerankRequest(query=query, passages=[{"text": d.get("text", "")} for d in documents]))
            return [
                RerankResult(
                    text=item["text"], original_score=float(documents[item["id"]].get("score", 0.0)),
                    rerank_score=float(item["score"]), metadata=dict(documents[item["id"]].get("metadata", {})), rank=i,
                )
                for i, item in enumerate(ranked[:top_k])
            ]
        except Exception:
            return CrossEncoderReranker().rerank(query, documents, top_k)


def benchmark_reranker(reranker, query: str, documents: list[dict], n_runs: int = 5) -> dict:
    """Benchmark latency over n_runs. (Đã implement sẵn)"""
    times = []
    for _ in range(n_runs):
        start = time.perf_counter()
        reranker.rerank(query, documents)
        elapsed = (time.perf_counter() - start) * 1000
        times.append(elapsed)
    return {"avg_ms": sum(times) / len(times), "min_ms": min(times), "max_ms": max(times)}


if __name__ == "__main__":
    query = "Nhân viên được nghỉ phép bao nhiêu ngày?"
    docs = [
        {"text": "Nhân viên được nghỉ 12 ngày/năm.", "score": 0.8, "metadata": {}},
        {"text": "Mật khẩu thay đổi mỗi 90 ngày.", "score": 0.7, "metadata": {}},
        {"text": "Thời gian thử việc là 60 ngày.", "score": 0.75, "metadata": {}},
    ]
    reranker = CrossEncoderReranker()
    for r in reranker.rerank(query, docs):
        print(f"[{r.rank}] {r.rerank_score:.4f} | {r.text}")
