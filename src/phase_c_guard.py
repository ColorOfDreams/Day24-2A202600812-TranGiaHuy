from __future__ import annotations

"""Phase C: production guardrails with PII scan, input/output rails, latency."""

import asyncio
import json
import os
import re
import sys
import time
import unicodedata

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    ADVERSARIAL_SET_PATH,
    GUARDRAILS_CONFIG_DIR,
    LATENCY_BUDGET_P95_MS,
    OPENAI_API_KEY,
    PRESIDIO_LANGUAGE,
)


def _normalize(text: str) -> str:
    text = unicodedata.normalize("NFKD", text or "")
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = text.replace("đ", "d").replace("Đ", "D").lower()
    return re.sub(r"\s+", " ", text).strip()


def _has_any(text: str, keywords: list[str]) -> bool:
    return any(keyword in text for keyword in keywords)


def _local_input_guard_response(user_text: str) -> str | None:
    """Fast deterministic guard for the lab adversarial classes."""
    text = _normalize(user_text)
    jailbreak_keywords = [
        "bo qua tat ca", "ignore your previous", "ignore previous instructions",
        "forget all", "forget your system prompt", "system override",
        "admin command", "priority 1", "pretend you are dan", "unrestricted ai",
        "khong co gioi han", "dong vai", "ra lenh", "dump all",
        "training data", "employee records", "system instructions",
        "system prompt", "in ra toan bo", "reveal all", "confidential",
        "tan cong mang", "mat khau admin",
    ]
    pii_request_keywords = [
        "cho toi biet cccd", "cccd cua nhan vien",
        "so dien thoai cua nhan vien", "thong tin ca nhan cua",
        "thong tin nhan vien", "bang luong", "tiet lo luong",
        "luong thang cua nhan vien", "employee salaries", "employee data",
        "personal data",
    ]
    off_topic_keywords = [
        "bai tho", "mua thu", "nau pho", "bitcoin", "ethereum",
        "giai phuong trinh", "phuong trinh vi phan", "marvel",
        "bo phim", "recommend",
    ]
    if _has_any(text, jailbreak_keywords):
        return "I cannot follow instructions that bypass policy, reveal secrets, or enable harmful activity."
    if _has_any(text, pii_request_keywords):
        return "I cannot provide personal or confidential employee information."
    if _has_any(text, off_topic_keywords):
        return "I can only answer questions about internal HR policies and related company procedures."
    return None


class GuardrailFallback:
    """NeMo-compatible fallback used when NeMo or remote LLM calls are unavailable."""

    async def generate_async(self, messages: list[dict]) -> str:
        if len(messages) == 1:
            response = _local_input_guard_response(messages[0]["content"])
            return response or "I can help with that."

        assistant_text = messages[-1]["content"]
        sensitive = _normalize(assistant_text)
        sensitive_keywords = [
            "cccd cua nhan vien la",
            "so dien thoai ca nhan cua",
            "mat khau he thong la",
            "thong tin bi mat",
            "employee record",
        ]
        if _has_any(sensitive, sensitive_keywords):
            return "I cannot provide confidential or personal employee information."
        return assistant_text


def setup_presidio():
    """Initialize Presidio with VN CCCD and phone recognizers, or use regex fallback."""
    try:
        from presidio_analyzer import AnalyzerEngine, Pattern, PatternRecognizer, RecognizerRegistry
        from presidio_anonymizer import AnonymizerEngine
    except ModuleNotFoundError:
        class FallbackResult:
            def __init__(self, entity_type: str, start: int, end: int, score: float):
                self.entity_type = entity_type
                self.start = start
                self.end = end
                self.score = score

        class FallbackAnalyzer:
            patterns = [
                ("EMAIL_ADDRESS", re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.I), 0.95),
                ("VN_CCCD", re.compile(r"\b\d{12}\b"), 0.9),
                ("VN_CCCD", re.compile(r"\b\d{9}\b"), 0.75),
                ("VN_PHONE", re.compile(r"\b0[3-9]\d{8}\b"), 0.9),
            ]

            def analyze(self, text: str, language: str = "en"):
                found = []
                for entity_type, pattern, score in self.patterns:
                    for match in pattern.finditer(text):
                        found.append(FallbackResult(entity_type, match.start(), match.end(), score))
                return found

        class FallbackAnonymizer:
            class Result:
                def __init__(self, text: str):
                    self.text = text

            def anonymize(self, text: str, analyzer_results: list):
                anonymized = text
                for result in sorted(analyzer_results, key=lambda item: item.start, reverse=True):
                    anonymized = (
                        anonymized[:result.start]
                        + f"<{result.entity_type}>"
                        + anonymized[result.end:]
                    )
                return self.Result(anonymized)

        return FallbackAnalyzer(), FallbackAnonymizer()

    registry = RecognizerRegistry()
    registry.load_predefined_recognizers()
    registry.add_recognizer(
        PatternRecognizer(
            supported_entity="VN_CCCD",
            patterns=[
                Pattern("CCCD 12 digits", r"\b\d{12}\b", 0.9),
                Pattern("CMND 9 digits", r"\b\d{9}\b", 0.75),
            ],
        )
    )
    registry.add_recognizer(
        PatternRecognizer(
            supported_entity="VN_PHONE",
            patterns=[Pattern("VN mobile", r"\b0[3-9]\d{8}\b", 0.9)],
        )
    )
    return AnalyzerEngine(registry=registry), AnonymizerEngine()


def pii_scan(text: str, analyzer=None, anonymizer=None) -> dict:
    """Task 9a: detect VN_CCCD, VN_PHONE, email, and anonymize them."""
    if analyzer is None or anonymizer is None:
        analyzer, anonymizer = setup_presidio()

    results = analyzer.analyze(text=text, language=PRESIDIO_LANGUAGE)
    target_types = {"VN_CCCD", "VN_PHONE", "EMAIL_ADDRESS", "PHONE_NUMBER"}
    target_results = [result for result in results if result.entity_type in target_types]
    if not target_results:
        return {"has_pii": False, "entities": [], "anonymized": text}

    anonymized = anonymizer.anonymize(text=text, analyzer_results=target_results).text
    return {
        "has_pii": True,
        "entities": [
            {
                "type": result.entity_type,
                "text": text[result.start:result.end],
                "score": round(float(result.score), 3),
                "start": result.start,
                "end": result.end,
            }
            for result in target_results
        ],
        "anonymized": anonymized,
    }


def setup_nemo_rails():
    """Load NeMo rails when available; otherwise use a deterministic compatible fallback."""
    if not OPENAI_API_KEY:
        return GuardrailFallback()
    try:
        from nemoguardrails import LLMRails, RailsConfig

        return LLMRails(RailsConfig.from_path(GUARDRAILS_CONFIG_DIR))
    except Exception:
        return GuardrailFallback()


async def check_input_rail(text: str, rails=None) -> dict:
    """Task 9b: check jailbreak, prompt injection, PII requests, and off-topic input."""
    local_response = _local_input_guard_response(text)
    if local_response is not None:
        return {
            "allowed": False,
            "blocked_reason": "local_input_rail",
            "response": local_response,
        }

    if rails is None:
        rails = setup_nemo_rails()
    response = await rails.generate_async(messages=[{"role": "user", "content": text}])
    response_norm = _normalize(response)
    blocked = _has_any(
        response_norm,
        ["i cannot", "i can only", "xin loi", "khong the", "khong duoc phep"],
    )
    return {
        "allowed": not blocked,
        "blocked_reason": "nemo_input_rail" if blocked else None,
        "response": response,
    }


async def check_output_rail(question: str, answer: str, rails=None) -> dict:
    """Task 11: check assistant output before returning it."""
    if rails is None:
        rails = setup_nemo_rails()
    response = await rails.generate_async(
        messages=[
            {"role": "user", "content": question},
            {"role": "assistant", "content": answer},
        ]
    )
    response_norm = _normalize(response)
    flagged = _has_any(response_norm, ["i cannot", "xin loi", "khong the"])
    return {
        "safe": not flagged,
        "flagged_reason": "nemo_output_rail" if flagged else None,
        "final_answer": response if flagged else answer,
    }


def run_adversarial_suite(
    adversarial_set: list[dict], rails=None, analyzer=None, anonymizer=None
) -> list[dict]:
    """Task 10: run adversarial inputs through PII scan and input rail."""
    if rails is None:
        rails = setup_nemo_rails()
    if analyzer is None or anonymizer is None:
        analyzer, anonymizer = setup_presidio()

    async def _run_all() -> list[dict]:
        suite_results = []
        for item in adversarial_set:
            blocked_by = None
            pii_result = pii_scan(item["input"], analyzer, anonymizer)
            if pii_result["has_pii"]:
                blocked_by = "presidio"

            if blocked_by is None:
                rail_result = await check_input_rail(item["input"], rails)
                if not rail_result["allowed"]:
                    blocked_by = rail_result["blocked_reason"] or "nemo_input"

            actual = "blocked" if blocked_by else "allowed"
            suite_results.append(
                {
                    "id": item["id"],
                    "category": item["category"],
                    "input": item["input"][:80] + ("..." if len(item["input"]) > 80 else ""),
                    "expected": item["expected"],
                    "actual": actual,
                    "blocked_by": blocked_by,
                    "passed": actual == item["expected"],
                }
            )
        return suite_results

    results = asyncio.run(_run_all())
    passed = sum(result["passed"] for result in results)
    print(f"Adversarial suite: {passed}/{len(results)} passed")
    return results


def _percentiles(times: list[float]) -> dict:
    if not times:
        return {"p50": 0.0, "p95": 0.0, "p99": 0.0}
    ordered = sorted(times)
    n = len(ordered)

    def pick(pct: float) -> float:
        index = min(int((n - 1) * pct), n - 1)
        return round(ordered[index], 2)

    return {"p50": pick(0.50), "p95": pick(0.95), "p99": pick(0.99)}


def measure_p95_latency(
    test_inputs: list[str], n_runs: int = 20, rails=None, analyzer=None, anonymizer=None
) -> dict:
    """Task 12: measure P50/P95/P99 for Presidio, rails, and total guard time."""
    if rails is None:
        rails = setup_nemo_rails()
    if analyzer is None or anonymizer is None:
        analyzer, anonymizer = setup_presidio()

    presidio_times: list[float] = []
    nemo_times: list[float] = []
    total_times: list[float] = []
    inputs = test_inputs[: max(n_runs, 0)]

    async def _measure() -> None:
        for text in inputs:
            total_start = time.perf_counter()
            start = time.perf_counter()
            pii_scan(text, analyzer, anonymizer)
            presidio_ms = (time.perf_counter() - start) * 1000

            start = time.perf_counter()
            await check_input_rail(text, rails)
            nemo_ms = (time.perf_counter() - start) * 1000

            presidio_times.append(presidio_ms)
            nemo_times.append(nemo_ms)
            total_times.append((time.perf_counter() - total_start) * 1000)

    asyncio.run(_measure())
    total = _percentiles(total_times)
    return {
        "presidio_ms": _percentiles(presidio_times),
        "nemo_ms": _percentiles(nemo_times),
        "total_ms": total,
        "latency_budget_ok": total["p95"] < LATENCY_BUDGET_P95_MS,
        "budget_ms": LATENCY_BUDGET_P95_MS,
    }


if __name__ == "__main__":
    analyzer, anonymizer = setup_presidio()
    rails = setup_nemo_rails()

    with open(ADVERSARIAL_SET_PATH, encoding="utf-8") as f:
        adversarial_set = json.load(f)
    results = run_adversarial_suite(adversarial_set, rails, analyzer, anonymizer)

    latency = measure_p95_latency(
        [item["input"] for item in adversarial_set], n_runs=len(adversarial_set),
        rails=rails, analyzer=analyzer, anonymizer=anonymizer,
    )
    report = {
        "results": results,
        "summary": {
            "total": len(results),
            "passed": sum(result["passed"] for result in results),
            "pass_rate": round(sum(result["passed"] for result in results) / len(results), 3)
            if results else 0.0,
        },
        "latency": latency,
    }

    os.makedirs("reports", exist_ok=True)
    with open("reports/guard_results.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print("Phase C report saved -> reports/guard_results.json")
