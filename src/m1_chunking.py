from __future__ import annotations

"""
Module 1: Advanced Chunking Strategies
=======================================
Implement semantic, hierarchical, và structure-aware chunking.
So sánh với basic chunking (baseline) để thấy improvement.

Test: pytest tests/test_m1.py
"""

import os, sys, glob, re
from dataclasses import dataclass, field

# Windows consoles configured with legacy code pages cannot print all Vietnamese
# status text/emoji.  Keep document loading from failing just because a PDF is
# skipped.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(errors="backslashreplace")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (DATA_DIR, HIERARCHICAL_PARENT_SIZE, HIERARCHICAL_CHILD_SIZE,
                    SEMANTIC_THRESHOLD)


def _model_is_cached(model_id: str) -> bool:
    """Avoid importing/loading a transformer when it is not available locally."""
    cache = os.getenv("HF_HUB_CACHE", os.path.expanduser("~/.cache/huggingface/hub"))
    return os.path.isdir(os.path.join(cache, "models--" + model_id.replace("/", "--")))


@dataclass
class Chunk:
    text: str
    metadata: dict = field(default_factory=dict)
    parent_id: str | None = None


def _extract_pdf_text(path: str) -> str:
    """Extract text layer từ PDF. Trả về "" nếu PDF là scan ảnh (không có text)."""
    from pypdf import PdfReader

    reader = PdfReader(path)
    pages = [page.extract_text() or "" for page in reader.pages]
    return "\n\n".join(pages).strip()


def load_documents(data_dir: str = DATA_DIR) -> list[dict]:
    """Load tất cả markdown và PDF (có text layer) từ data/. (Đã implement sẵn)

    - .md: đọc trực tiếp.
    - .pdf: trích text layer bằng pypdf. PDF scan ảnh (không có text) bị bỏ qua
      kèm cảnh báo — RAG text-based không xử lý được scan nếu chưa OCR.
    """
    docs = []
    for fp in sorted(glob.glob(os.path.join(data_dir, "*.md"))):
        with open(fp, encoding="utf-8") as f:
            docs.append({"text": f.read(), "metadata": {"source": os.path.basename(fp)}})

    for fp in sorted(glob.glob(os.path.join(data_dir, "*.pdf"))):
        text = _extract_pdf_text(fp)
        if text:
            docs.append({"text": text, "metadata": {"source": os.path.basename(fp)}})
        else:
            print(f"  ⚠️  Bỏ qua {os.path.basename(fp)}: PDF scan ảnh, không có text layer (cần OCR).")

    return docs


# ─── Baseline: Basic Chunking (để so sánh) ──────────────


def chunk_basic(text: str, chunk_size: int = 500, metadata: dict | None = None) -> list[Chunk]:
    """
    Basic chunking: split theo paragraph (\\n\\n).
    Đây là baseline — KHÔNG phải mục tiêu của module này.
    (Đã implement sẵn)
    """
    metadata = metadata or {}
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks = []
    current = ""
    for i, para in enumerate(paragraphs):
        if len(current) + len(para) > chunk_size and current:
            chunks.append(Chunk(text=current.strip(), metadata={**metadata, "chunk_index": len(chunks)}))
            current = ""
        current += para + "\n\n"
    if current.strip():
        chunks.append(Chunk(text=current.strip(), metadata={**metadata, "chunk_index": len(chunks)}))
    return chunks


# ─── Strategy 1: Semantic Chunking ───────────────────────


def chunk_semantic(text: str, threshold: float = SEMANTIC_THRESHOLD,
                   metadata: dict | None = None) -> list[Chunk]:
    """
    Split text by sentence similarity — nhóm câu cùng chủ đề.
    Tốt hơn basic vì không cắt giữa ý.
    """
    metadata = metadata or {}
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+|\n\s*\n", text) if s.strip()]
    if not sentences:
        return []

    # Semantic models are optional at runtime.  Loading only from the local
    # cache keeps the lab usable on a machine without network access; when the
    # model has been pre-downloaded it is used as intended.
    similarities = None
    try:
        if not _model_is_cached("sentence-transformers/all-MiniLM-L6-v2"):
            raise FileNotFoundError("all-MiniLM-L6-v2 is not cached locally")
        import numpy as np
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer("all-MiniLM-L6-v2", local_files_only=True)
        vectors = model.encode(sentences, show_progress_bar=False)
        similarities = [
            float(np.dot(vectors[i - 1], vectors[i]) /
                  ((np.linalg.norm(vectors[i - 1]) * np.linalg.norm(vectors[i])) + 1e-9))
            for i in range(1, len(sentences))
        ]
    except Exception:
        # Paragraphs are the safest local approximation of semantic units.
        # Unlike sentence-level lexical overlap, this does not fragment a
        # policy section simply because neighbouring sentences use pronouns.
        return [
            Chunk(text=chunk.text, metadata={**chunk.metadata, "strategy": "semantic"})
            for chunk in chunk_basic(text, chunk_size=500, metadata=metadata)
        ]

    groups: list[list[str]] = [[sentences[0]]]
    for sentence, similarity in zip(sentences[1:], similarities):
        if similarity < threshold:
            groups.append([sentence])
        else:
            groups[-1].append(sentence)

    return [
        Chunk(text=" ".join(group), metadata={**metadata, "strategy": "semantic", "chunk_index": i})
        for i, group in enumerate(groups)
    ]


# ─── Strategy 2: Hierarchical Chunking ──────────────────


def chunk_hierarchical(text: str, parent_size: int = HIERARCHICAL_PARENT_SIZE,
                       child_size: int = HIERARCHICAL_CHILD_SIZE,
                       metadata: dict | None = None) -> tuple[list[Chunk], list[Chunk]]:
    """
    Parent-child hierarchy: retrieve child (precision) → return parent (context).
    Đây là default recommendation cho production RAG.

    Returns:
        (parents, children) — mỗi child có parent_id link đến parent.
    """
    metadata = metadata or {}
    if not text.strip():
        return [], []
    if parent_size <= 0 or child_size <= 0:
        raise ValueError("parent_size and child_size must be positive")

    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]

    def pack(parts: list[str], limit: int) -> list[str]:
        packed, current = [], ""
        for part in parts:
            # Split an unusually long paragraph on word boundaries so no
            # chunk becomes arbitrarily large.
            pieces = [part]
            if len(part) > limit:
                words = part.split()
                pieces, piece = [], ""
                for word in words:
                    if piece and len(piece) + len(word) + 1 > limit:
                        pieces.append(piece)
                        piece = word
                    else:
                        piece = f"{piece} {word}".strip()
                if piece:
                    pieces.append(piece)
            for piece in pieces:
                if current and len(current) + len(piece) + 2 > limit:
                    packed.append(current)
                    current = piece
                else:
                    current = f"{current}\n\n{piece}".strip()
        if current:
            packed.append(current)
        return packed

    parent_texts = pack(paragraphs, parent_size)
    parents: list[Chunk] = []
    children: list[Chunk] = []
    for parent_index, parent_text in enumerate(parent_texts):
        parent_id = f"parent_{parent_index}"
        parents.append(Chunk(
            text=parent_text,
            metadata={**metadata, "chunk_type": "parent", "parent_id": parent_id, "chunk_index": parent_index},
        ))
        # Preserve paragraph boundaries where possible, then split oversized
        # passages into retrieval-sized children.
        child_parts = [p.strip() for p in re.split(r"\n\s*\n", parent_text) if p.strip()]
        for child_index, child_text in enumerate(pack(child_parts, child_size)):
            children.append(Chunk(
                text=child_text,
                metadata={**metadata, "chunk_type": "child", "child_index": child_index},
                parent_id=parent_id,
            ))
    return parents, children


# ─── Strategy 3: Structure-Aware Chunking ────────────────


def chunk_structure_aware(text: str, metadata: dict | None = None) -> list[Chunk]:
    """
    Parse markdown headers → chunk theo logical structure.
    Giữ nguyên tables, code blocks, lists — không cắt giữa chừng.
    """
    metadata = metadata or {}
    if not text.strip():
        return []

    header_re = re.compile(r"^#{1,3}\s+.+$", re.MULTILINE)
    matches = list(header_re.finditer(text))
    if not matches:
        return [Chunk(text=text.strip(), metadata={**metadata, "section": "", "strategy": "structure", "chunk_index": 0})]

    chunks: list[Chunk] = []
    # Keep any preamble rather than silently dropping it.
    preamble = text[:matches[0].start()].strip()
    if preamble:
        chunks.append(Chunk(text=preamble, metadata={**metadata, "section": "", "strategy": "structure", "chunk_index": 0}))

    for match_index, match in enumerate(matches):
        end = matches[match_index + 1].start() if match_index + 1 < len(matches) else len(text)
        header = match.group(0).strip()
        body = text[match.end():end].strip()
        section_text = f"{header}\n\n{body}".strip()
        chunks.append(Chunk(
            text=section_text,
            metadata={**metadata, "section": header, "strategy": "structure", "chunk_index": len(chunks)},
        ))
    return chunks


# ─── A/B Test: Compare All Strategies ────────────────────


def compare_strategies(documents: list[dict]) -> dict:
    """
    Run all strategies on documents and compare.
    (Đã implement sẵn — sẽ hoạt động khi bạn implement 3 strategies ở trên)
    """
    def _stats(chunk_list):
        lengths = [len(c.text) for c in chunk_list]
        if not lengths:
            return {"count": 0, "avg_len": 0, "min_len": 0, "max_len": 0}
        return {
            "count": len(lengths),
            "avg_len": round(sum(lengths) / len(lengths)),
            "min_len": min(lengths),
            "max_len": max(lengths),
        }

    all_text = "\n\n".join(d["text"] for d in documents)
    meta = {"source": "all"}

    basic = chunk_basic(all_text, metadata=meta)
    semantic = chunk_semantic(all_text, metadata=meta)
    parents, children = chunk_hierarchical(all_text, metadata=meta)
    structure = chunk_structure_aware(all_text, metadata=meta)

    results = {
        "basic": _stats(basic),
        "semantic": _stats(semantic),
        "hierarchical": {**_stats(children), "parents": len(parents)},
        "structure": _stats(structure),
    }

    print(f"{'Strategy':<15} {'Chunks':>7} {'Avg':>5} {'Min':>5} {'Max':>5}")
    for name, s in results.items():
        print(f"{name:<15} {s['count']:>7} {s['avg_len']:>5} {s['min_len']:>5} {s['max_len']:>5}")

    return results


if __name__ == "__main__":
    docs = load_documents()
    print(f"Loaded {len(docs)} documents")
    results = compare_strategies(docs)
    for name, stats in results.items():
        print(f"  {name}: {stats}")
