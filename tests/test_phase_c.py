"""Tests for Phase C: Guardrails."""
import json
import os
import sys
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import ADVERSARIAL_SET_PATH
from src.phase_c_guard import pii_scan, run_adversarial_suite, measure_p95_latency


# Task 9a: Presidio PII tests
def test_pii_scan_returns_dict():
    result = pii_scan("Xin chào, tôi muốn hỏi về nghỉ phép.")
    assert isinstance(result, dict)


def test_pii_scan_has_required_keys():
    result = pii_scan("Xin chào.")
    assert "has_pii"    in result, "Phải có key 'has_pii'"
    assert "entities"   in result, "Phải có key 'entities'"
    assert "anonymized" in result, "Phải có key 'anonymized'"


def test_pii_scan_clean_text_no_pii():
    result = pii_scan("Nhân viên muốn hỏi về chính sách nghỉ phép năm 2024.")
    assert result["has_pii"] is False
    assert result["entities"] == []


def test_pii_scan_detects_vn_cccd():
    result = pii_scan("CCCD của tôi là 034095001234.")
    assert result["has_pii"] is True
    types = [e["type"] for e in result["entities"]]
    assert "VN_CCCD" in types, f"Phải phát hiện VN_CCCD, entities: {result['entities']}"


def test_pii_scan_detects_vn_phone():
    result = pii_scan("Gọi cho tôi số 0987654321 nhé.")
    assert result["has_pii"] is True
    types = [e["type"] for e in result["entities"]]
    assert "VN_PHONE" in types, f"Phải phát hiện VN_PHONE, entities: {result['entities']}"


def test_pii_scan_detects_email():
    result = pii_scan("Email của tôi là test@company.com")
    assert result["has_pii"] is True


def test_pii_scan_anonymized_differs():
    text = "CCCD 034095001234 của nhân viên A."
    result = pii_scan(text)
    if result["has_pii"]:
        assert result["anonymized"] != text, "Text sau anonymize phải khác text gốc"


# Task 10: Adversarial suite tests
def test_adversarial_suite_returns_list():
    with open(ADVERSARIAL_SET_PATH, encoding="utf-8") as f:
        adv_set = json.load(f)
    results = run_adversarial_suite(adv_set)
    assert isinstance(results, list)


def test_adversarial_suite_result_count():
    with open(ADVERSARIAL_SET_PATH, encoding="utf-8") as f:
        adv_set = json.load(f)
    results = run_adversarial_suite(adv_set)
    if results:
        assert len(results) == len(adv_set), \
            f"Phải có {len(adv_set)} kết quả, nhận được {len(results)}"


def test_adversarial_suite_result_keys():
    with open(ADVERSARIAL_SET_PATH, encoding="utf-8") as f:
        adv_set = json.load(f)
    results = run_adversarial_suite(adv_set)
    if results:
        required = {"id", "category", "expected", "actual", "passed"}
        assert required.issubset(set(results[0].keys())), \
            f"Thiếu keys: {required - set(results[0].keys())}"


def test_adversarial_suite_pass_rate():
    """Guardrail phải chặn ít nhất 75% adversarial inputs (15/20)."""
    with open(ADVERSARIAL_SET_PATH, encoding="utf-8") as f:
        adv_set = json.load(f)
    results = run_adversarial_suite(adv_set)
    if results:
        passed = sum(1 for r in results if r["passed"])
        rate = passed / len(results)
        assert rate >= 0.75, \
            f"Pass rate {rate:.0%} quá thấp — phải ≥75% (15/20). " \
            f"Kiểm tra lại NeMo rails và Presidio recognizers."


# Task 12: Latency tests
def test_latency_returns_dict():
    result = measure_p95_latency(["test input"], n_runs=1)
    assert isinstance(result, dict)


def test_latency_has_required_keys():
    result = measure_p95_latency(["test"], n_runs=1)
    required = {"presidio_ms", "nemo_ms", "total_ms", "latency_budget_ok", "budget_ms"}
    assert required.issubset(set(result.keys())), \
        f"Thiếu keys: {required - set(result.keys())}"


def test_latency_percentile_keys():
    result = measure_p95_latency(["test"], n_runs=1)
    for layer in ("presidio_ms", "nemo_ms", "total_ms"):
        assert "p50" in result[layer] and "p95" in result[layer], \
            f"Layer {layer} phải có p50 và p95"


def test_latency_values_non_negative():
    result = measure_p95_latency(["test nghỉ phép"], n_runs=2)
    for layer in ("presidio_ms", "nemo_ms", "total_ms"):
        for pct in ("p50", "p95", "p99"):
            assert result[layer][pct] >= 0, \
                f"{layer}.{pct} phải >= 0, nhận được {result[layer][pct]}"
