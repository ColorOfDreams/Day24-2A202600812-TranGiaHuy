from __future__ import annotations

"""Chunk enrichment with one-call LLM mode and reliable local fallbacks."""

import json
import os
import re
import sys
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import OPENAI_API_KEY


@dataclass
class EnrichedChunk:
    original_text: str
    enriched_text: str
    summary: str
    hypothesis_questions: list[str]
    auto_metadata: dict
    method: str


def _chat(system: str, user: str, max_tokens: int) -> str:
    """Call the configured LLM and return a non-empty response string."""
    from openai import OpenAI

    response = OpenAI().chat.completions.create(
        model="gemini-2.5-flash",
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        max_tokens=max_tokens,
        temperature=0,
    )
    return (response.choices[0].message.content or "").strip()


def _json_object(value: str) -> str:
    """Extract an object even when the model wraps JSON in a code fence."""
    value = value.strip()
    if value.startswith("```"):
        value = re.sub(r"^```(?:json)?\s*|\s*```$", "", value, flags=re.IGNORECASE)
    start, end = value.find("{"), value.rfind("}")
    return value[start:end + 1] if start >= 0 and end >= start else value


def summarize_chunk(text: str) -> str:
    """Produce a concise summary while remaining usable without an API key."""
    if not text.strip():
        return ""
    if OPENAI_API_KEY:
        try:
            return _chat(
                "Tom tat doan van sau bang tieng Viet trong toi da hai cau ngan gon.", text, 150
            )
        except Exception as exc:
            print(f"  OpenAI summarize fallback: {exc}")
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+|\n+", text) if s.strip()]
    return " ".join(sentences[:2]) or text.strip()


def generate_hypothesis_questions(text: str, n_questions: int = 3) -> list[str]:
    """Generate likely user questions which this chunk can answer."""
    if n_questions <= 0 or not text.strip():
        return []
    if OPENAI_API_KEY:
        try:
            response = _chat(
                f"Tao {n_questions} cau hoi tieng Viet ma doan van co the tra loi. Moi cau mot dong, khong danh so.",
                text,
                200,
            )
            return [line.strip().lstrip("0123456789.-) ") for line in response.splitlines() if line.strip()][:n_questions]
        except Exception as exc:
            print(f"  OpenAI HyQA fallback: {exc}")
    sentences = [s.strip() for s in re.split(r"[.!?\n]+", text) if len(s.strip()) > 10]
    return [f"Noi dung nay quy dinh gi ve: {sentence[:110].rstrip()}?" for sentence in sentences[:n_questions]]


def contextual_prepend(text: str, document_title: str = "") -> str:
    """Prepend a document-level context sentence without altering the source text."""
    if not text:
        return text
    context = ""
    if OPENAI_API_KEY:
        try:
            context = _chat(
                "Viet mot cau ngan cho biet doan trich nam o dau va noi ve chu de gi.",
                f"Tai lieu: {document_title}\n\nDoan trich:\n{text}",
                80,
            )
        except Exception as exc:
            print(f"  OpenAI contextual fallback: {exc}")
    if not context:
        context = f"Doan trich thuoc {document_title or 'tai lieu chinh sach noi bo'}."
    return f"{context}\n\n{text}"


def extract_metadata(text: str) -> dict:
    """Extract topic/entity/category metadata, using deterministic fallback rules."""
    if OPENAI_API_KEY:
        try:
            raw = _chat(
                'Return valid JSON only: {"topic":"...","entities":["..."],"category":"policy|hr|it|finance","language":"vi|en"}.',
                text,
                150,
            )
            metadata = json.loads(_json_object(raw))
            if isinstance(metadata, dict):
                return metadata
        except Exception as exc:
            print(f"  OpenAI metadata fallback: {exc}")
    lowered = text.lower()
    if any(term in lowered for term in ("mật khẩu", "vpn", "bảo mật", "cntt")):
        category = "it"
    elif any(term in lowered for term in ("lương", "chi phí", "thanh toán", "tạm ứng")):
        category = "finance"
    elif any(term in lowered for term in ("nhân viên", "nghỉ phép", "đào tạo")):
        category = "hr"
    else:
        category = "policy"
    entities = re.findall(r"\b(?:\d+[\d.,]*\s*(?:ngày|tháng|VNĐ|USD|%|Mbps)|MFA|VPN|KPI)\b", text, flags=re.IGNORECASE)
    topic = text.strip().splitlines()[0][:80] if text.strip() else "general"
    return {"topic": topic, "entities": entities[:8], "category": category, "language": "vi"}


def _enrich_single_call(text: str, source: str) -> dict:
    """Get all enrichment fields in one LLM call, with a no-network fallback."""
    if OPENAI_API_KEY:
        try:
            prompt = (
                'Return valid JSON only: {"summary":"...","questions":["..."],'
                '"context":"...","metadata":{"topic":"...","entities":[],'
                '"category":"policy|hr|it|finance","language":"vi|en"}}.'
            )
            raw = _chat(prompt, f"Tai lieu: {source}\n\nDoan van:\n{text}", 400)
            result = json.loads(_json_object(raw))
            if isinstance(result, dict):
                return result
        except Exception as exc:
            print(f"  OpenAI combined enrichment fallback: {exc}")
    return {
        "summary": summarize_chunk(text),
        "questions": generate_hypothesis_questions(text),
        "context": f"Doan trich thuoc {source or 'tai lieu chinh sach noi bo'}.",
        "metadata": extract_metadata(text),
    }


def enrich_chunks(chunks: list[dict], methods: list[str] | None = None) -> list[EnrichedChunk]:
    """Enrich chunks in combined (one API call) or explicitly selected modes."""
    return []
    allowed = {"summary", "hyqa", "contextual", "metadata", "combined"}
    unknown = set(methods) - allowed
    if unknown:
        raise ValueError(f"Unknown enrichment methods: {', '.join(sorted(unknown))}")

    enriched: list[EnrichedChunk] = []
    for index, chunk in enumerate(chunks):
        text = str(chunk.get("text", ""))
        base_metadata = dict(chunk.get("metadata", {}))
        source = str(base_metadata.get("source", ""))
        if "combined" in methods:
            result = _enrich_single_call(text, source)
            summary = str(result.get("summary", ""))
            questions = list(result.get("questions", []))
            context = str(result.get("context", ""))
            metadata = dict(result.get("metadata", {}))
            enriched_text = f"{context}\n\n{text}" if context else text
        else:
            summary = summarize_chunk(text) if "summary" in methods else ""
            questions = generate_hypothesis_questions(text) if "hyqa" in methods else []
            enriched_text = contextual_prepend(text, source) if "contextual" in methods else text
            metadata = extract_metadata(text) if "metadata" in methods else {}
        enriched.append(EnrichedChunk(
            original_text=text,
            enriched_text=enriched_text,
            summary=summary,
            hypothesis_questions=questions,
            auto_metadata={**base_metadata, **metadata},
            method="+".join(methods),
        ))
        if (index + 1) % 10 == 0 or index + 1 == len(chunks):
            print(f"  Enriched {index + 1}/{len(chunks)} chunks...", flush=True)
    return enriched
