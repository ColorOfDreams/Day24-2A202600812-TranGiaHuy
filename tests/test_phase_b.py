"""Tests for Phase B: LLM-as-Judge."""
import os
import sys
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.phase_b_judge import (
    pairwise_judge, swap_and_average, cohen_kappa, bias_report, JudgeResult,
)

Q   = "Nhân viên được nghỉ bao nhiêu ngày phép năm?"
A_A = "Nhân viên được nghỉ 15 ngày phép năm theo chính sách v2024."
A_B = "Theo quy định, nhân viên có 12 ngày phép hàng năm."


# Task 5 tests
def test_pairwise_judge_returns_dict():
    result = pairwise_judge(Q, A_A, A_B)
    assert isinstance(result, dict), "pairwise_judge phải trả về dict"


def test_pairwise_judge_has_required_keys():
    result = pairwise_judge(Q, A_A, A_B)
    assert "winner" in result,    "Phải có key 'winner'"
    assert "reasoning" in result, "Phải có key 'reasoning'"
    assert "scores" in result,    "Phải có key 'scores'"


def test_pairwise_judge_winner_valid():
    result = pairwise_judge(Q, A_A, A_B)
    assert result["winner"] in {"A", "B", "tie"}, \
        f"winner phải là 'A', 'B', hoặc 'tie', nhận được: {result['winner']}"


def test_pairwise_judge_scores_in_range():
    result = pairwise_judge(Q, A_A, A_B)
    if "scores" in result:
        for k, v in result["scores"].items():
            assert 0.0 <= v <= 1.0, f"Score {k}={v} phải trong khoảng [0, 1]"


def test_pairwise_judge_reasoning_not_empty():
    result = pairwise_judge(Q, A_A, A_B)
    if result.get("winner") != "tie":  # tie có thể không có reasoning
        assert len(result.get("reasoning", "")) > 0, "Phải có reasoning khi có winner"


# Task 6 tests
def test_swap_and_average_returns_judge_result():
    result = swap_and_average(Q, A_A, A_B)
    assert isinstance(result, JudgeResult), "swap_and_average phải trả về JudgeResult"


def test_swap_and_average_winners_valid():
    result = swap_and_average(Q, A_A, A_B)
    for attr in ("winner_pass1", "winner_pass2", "final_winner"):
        assert getattr(result, attr) in {"A", "B", "tie"}, \
            f"{attr} phải là 'A', 'B', hoặc 'tie'"


def test_swap_and_average_position_consistent_bool():
    result = swap_and_average(Q, A_A, A_B)
    assert isinstance(result.position_consistent, bool), \
        "position_consistent phải là bool"


def test_swap_and_average_inconsistency_detection():
    """Nếu swap đổi kết quả thì position_consistent phải là False."""
    result = swap_and_average(Q, A_A, A_B)
    if result.winner_pass1 != result.winner_pass2 and result.winner_pass1 != "tie":
        assert not result.position_consistent, \
            "Khi 2 passes không đồng ý, position_consistent phải False"


# Task 7 tests
def test_cohen_kappa_perfect_agreement():
    labels = [1, 0, 1, 1, 0, 0, 1, 0, 1, 0]
    kappa = cohen_kappa(labels, labels)
    assert abs(kappa - 1.0) < 0.01, f"Perfect agreement → κ=1.0, nhận được {kappa}"


def test_cohen_kappa_no_agreement():
    labels_a = [1, 0, 1, 0, 1, 0, 1, 0, 1, 0]
    labels_b = [0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
    kappa = cohen_kappa(labels_a, labels_b)
    assert kappa <= 0.0, f"Perfect disagreement → κ≤0, nhận được {kappa}"


def test_cohen_kappa_range():
    judge  = [1, 1, 0, 1, 0, 1, 0, 0, 1, 1]
    human  = [1, 0, 0, 1, 1, 1, 0, 1, 1, 0]
    kappa = cohen_kappa(judge, human)
    assert -1.0 <= kappa <= 1.0, f"κ phải trong [-1, 1], nhận được {kappa}"


# Task 8 tests
def test_bias_report_empty_input():
    result = bias_report([])
    assert result["total_judged"] == 0


def test_bias_report_has_required_keys():
    dummy = JudgeResult(
        question=Q, answer_a=A_A, answer_b=A_B,
        winner_pass1="A", winner_pass2="A", final_winner="A",
        reasoning_pass1="", reasoning_pass2="", position_consistent=True,
    )
    result = bias_report([dummy])
    required = {"total_judged", "position_bias_rate", "verbosity_bias",
                "position_bias_count", "interpretation"}
    assert required.issubset(set(result.keys())), \
        f"Thiếu keys: {required - set(result.keys())}"


def test_bias_report_rates_in_range():
    dummy = JudgeResult(
        question=Q, answer_a=A_A, answer_b=A_B,
        winner_pass1="A", winner_pass2="B", final_winner="tie",
        reasoning_pass1="", reasoning_pass2="", position_consistent=False,
    )
    result = bias_report([dummy])
    assert 0.0 <= result["position_bias_rate"] <= 1.0
    assert 0.0 <= result["verbosity_bias"] <= 1.0
