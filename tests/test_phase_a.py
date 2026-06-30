"""Tests for Phase A: RAGAS evaluation."""
import json
import os
import sys
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.phase_a_ragas import (
    load_test_set_50q, group_by_distribution, bottom_10,
    cluster_analysis, RagasResult,
)


@pytest.fixture
def test_set():
    return load_test_set_50q()


@pytest.fixture
def dummy_results():
    """Fake RagasResult list for testing logic without calling RAGAS."""
    items = []
    for i, (dist, f, ar, cp, cr) in enumerate([
        ("factual",    0.9, 0.8, 0.7, 0.6),
        ("multi_hop",  0.3, 0.4, 0.5, 0.2),
        ("adversarial",0.1, 0.2, 0.3, 0.4),
        ("factual",    0.8, 0.7, 0.9, 0.8),
        ("multi_hop",  0.2, 0.3, 0.1, 0.2),
    ]):
        items.append(RagasResult(
            question_id=i+1, distribution=dist, question=f"Q{i+1}",
            answer="A", contexts=[], ground_truth="GT",
            faithfulness=f, answer_relevancy=ar,
            context_precision=cp, context_recall=cr,
        ))
    return items


# Task 1 tests
def test_group_by_distribution_keys(test_set):
    groups = group_by_distribution(test_set)
    assert set(groups.keys()) == {"factual", "multi_hop", "adversarial"}, \
        "group_by_distribution phải trả về 3 keys: factual, multi_hop, adversarial"


def test_group_by_distribution_counts(test_set):
    groups = group_by_distribution(test_set)
    assert len(groups["factual"])    == 20, "factual phải có 20 câu"
    assert len(groups["multi_hop"])  == 20, "multi_hop phải có 20 câu"
    assert len(groups["adversarial"]) == 10, "adversarial phải có 10 câu"


def test_group_by_distribution_total(test_set):
    groups = group_by_distribution(test_set)
    total = sum(len(v) for v in groups.values())
    assert total == 50, "Tổng phải là 50 câu hỏi"


# Task 3 tests
def test_bottom_10_length(dummy_results):
    b10 = bottom_10(dummy_results)
    assert len(b10) <= 10, "bottom_10 phải trả về tối đa 10 items"


def test_bottom_10_has_required_keys(dummy_results):
    b10 = bottom_10(dummy_results)
    if b10:
        required = {"rank", "question_id", "distribution", "question",
                    "avg_score", "worst_metric", "diagnosis", "suggested_fix"}
        assert required.issubset(set(b10[0].keys())), \
            f"Thiếu keys: {required - set(b10[0].keys())}"


def test_bottom_10_sorted_ascending(dummy_results):
    b10 = bottom_10(dummy_results)
    if len(b10) >= 2:
        scores = [item["avg_score"] for item in b10]
        assert scores == sorted(scores), "bottom_10 phải sắp xếp theo avg_score tăng dần"


def test_bottom_10_rank_starts_at_1(dummy_results):
    b10 = bottom_10(dummy_results)
    if b10:
        assert b10[0]["rank"] == 1, "rank đầu tiên phải là 1"


# Task 4 tests
def test_cluster_analysis_has_matrix(dummy_results):
    clusters = cluster_analysis(dummy_results)
    assert "matrix" in clusters, "cluster_analysis phải có key 'matrix'"


def test_cluster_analysis_metrics(dummy_results):
    clusters = cluster_analysis(dummy_results)
    if "matrix" in clusters:
        expected_metrics = {"faithfulness", "answer_relevancy",
                            "context_precision", "context_recall"}
        assert set(clusters["matrix"].keys()) == expected_metrics


def test_cluster_analysis_has_insight(dummy_results):
    clusters = cluster_analysis(dummy_results)
    assert "insight" in clusters and len(clusters["insight"]) > 0, \
        "cluster_analysis phải có insight string"
