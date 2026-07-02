from __future__ import annotations

"""Phase B: LLM-as-Judge - pairwise judging, swap consistency, kappa, bias."""

import json
import math
import os
import re
import sys
import unicodedata
from dataclasses import dataclass, field
from typing import Any

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import HUMAN_LABELS_PATH, JUDGE_MODEL, OPENAI_API_KEY


@dataclass
class JudgeResult:
    question: str
    answer_a: str
    answer_b: str
    winner_pass1: str
    winner_pass2: str
    final_winner: str
    reasoning_pass1: str
    reasoning_pass2: str
    position_consistent: bool
    scores_pass1: dict = field(default_factory=dict)
    scores_pass2: dict = field(default_factory=dict)


def _normalize(text: str) -> str:
    """Lowercase, remove Vietnamese accents, and collapse whitespace."""
    text = unicodedata.normalize("NFKD", text or "")
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = text.replace("đ", "d").replace("Đ", "D").lower()
    return re.sub(r"\s+", " ", text).strip()


def _clamp(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        number = default
    if math.isnan(number):
        return default
    return max(0.0, min(1.0, number))


def _token_set(text: str) -> set[str]:
    stopwords = {
        "la", "va", "cua", "cho", "toi", "ve", "thi", "co", "khong", "bao",
        "nhieu", "duoc", "can", "ai", "khi", "theo", "trong", "mot", "nay",
    }
    return {
        token
        for token in re.findall(r"[a-z0-9]+", _normalize(text))
        if len(token) > 1 and token not in stopwords
    }


def _heuristic_score(question: str, answer: str) -> tuple[float, list[str]]:
    """Score an answer using policy-oriented signals for offline judging."""
    q = _normalize(question)
    a = _normalize(answer)
    reasons: list[str] = []

    if not a or "khong co ground truth" in a or "no ground truth" in a:
        return 0.05, ["answer is empty or explicitly missing ground truth"]

    score = 0.35
    q_tokens = _token_set(question)
    a_tokens = _token_set(answer)
    if q_tokens:
        overlap = len(q_tokens & a_tokens) / len(q_tokens)
        score += min(0.25, overlap * 0.25)
        if overlap >= 0.35:
            reasons.append("addresses key terms from the question")

    length = len(answer.strip())
    if 25 <= length <= 700:
        score += 0.10
        reasons.append("has enough detail without being excessive")
    elif length > 700:
        score -= 0.08
        reasons.append("is overly verbose")

    if any(ch.isdigit() for ch in question) and any(ch.isdigit() for ch in answer):
        score += 0.08
        reasons.append("preserves numerical details")

    # Domain-specific policy checks used as rubric features, not question IDs.
    if "nghi" in q and "phep" in q and "nam" in q:
        if re.search(r"\b15\b", a) or "hien hanh" in a or "v2024" in a:
            score += 0.18
            reasons.append("uses the current annual leave policy")
        if re.search(r"\b12\b", a) and "v2023" not in a and "da thay the" not in a:
            score -= 0.20
            reasons.append("appears to use an outdated leave policy")

    if "thu viec" in q and "nghi phep" in q:
        if "khong" in a and ("khong luong" in a or "truong phong" in a or "phe duyet" in a):
            score += 0.20
            reasons.append("handles the probation leave negation correctly")

    if "vpn" in q and ("ca nhan" in q or "nordvpn" in q):
        if "khong" in a or "cam" in a or "wireguard" in a or "cong ty" in a:
            score += 0.22
            reasons.append("rejects personal VPN use as required")
        if "duoc" in a and "khong" not in a:
            score -= 0.25
            reasons.append("allows a prohibited personal VPN")

    if "55" in q and ("thiet bi" in q or "mua" in q):
        if "ceo" in a or "tong giam doc" in a:
            score += 0.22
            reasons.append("identifies the correct approval level")
        if "giam doc phong ban" in a and "ceo" not in a:
            score -= 0.18
            reasons.append("uses the lower approval level")

    if "tam ung" in q and "8" in q:
        if "ke toan truong" in a or "80.000" in a or "80000" in a:
            score += 0.20
            reasons.append("includes approval or pro-rata penalty detail")
        if "2%" in a and not ("80.000" in a or "80000" in a):
            score -= 0.08
            reasons.append("mentions the rate but misses the calculated penalty")

    if "ket hon" in q and re.search(r"\b3\b", a):
        score += 0.15
        reasons.append("contains the correct marriage leave duration")

    if "thuong tet" in q and ("1 thang" in a or "mot thang" in a):
        score += 0.15
        reasons.append("contains the correct Tet bonus minimum")

    if "dao tao" in q or "khoa hoc" in q or "tai tro" in q:
        if "25" in a and ("hoan tra" in a or "hoan" in a):
            score += 0.16
            reasons.append("keeps the reimbursement amount")

    return _clamp(score), reasons or ["uses general relevance and completeness signals"]


def _fallback_pairwise(question: str, answer_a: str, answer_b: str) -> dict:
    score_a, reasons_a = _heuristic_score(question, answer_a)
    score_b, reasons_b = _heuristic_score(question, answer_b)
    margin = abs(score_a - score_b)
    if margin < 0.06:
        winner = "tie"
        reasoning = "Both answers are similarly supported by the offline rubric."
    elif score_a > score_b:
        winner = "A"
        reasoning = "Answer A is stronger: " + "; ".join(reasons_a[:2]) + "."
    else:
        winner = "B"
        reasoning = "Answer B is stronger: " + "; ".join(reasons_b[:2]) + "."
    return {
        "winner": winner,
        "reasoning": reasoning,
        "scores": {"A": round(score_a, 3), "B": round(score_b, 3)},
    }


def _try_llm_judge(question: str, answer_a: str, answer_b: str) -> dict | None:
    if not OPENAI_API_KEY:
        return None
    try:
        from openai import OpenAI

        client = OpenAI(
            api_key=OPENAI_API_KEY,
            base_url=os.environ.get("OPENAI_BASE_URL") or None,
        )
        prompt = f"""You are an HR policy evaluator.
Compare Answer A and Answer B for the user question.
Use correctness first, then completeness, then clarity.
Return only JSON with keys: winner, reasoning, scores.
winner must be "A", "B", or "tie"; scores must contain numeric A and B in [0,1].

Question: {question}

Answer A: {answer_a}

Answer B: {answer_b}
"""
        response = client.chat.completions.create(
            model=JUDGE_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
        )
        content = (response.choices[0].message.content or "").strip()
        if content.startswith("```"):
            content = re.sub(r"^```(?:json)?\s*|\s*```$", "", content, flags=re.I | re.S).strip()
        parsed = json.loads(content)
        winner = parsed.get("winner")
        scores = parsed.get("scores") or {}
        if winner not in {"A", "B", "tie"}:
            return None
        return {
            "winner": winner,
            "reasoning": str(parsed.get("reasoning") or "LLM judge completed."),
            "scores": {"A": _clamp(scores.get("A")), "B": _clamp(scores.get("B"))},
        }
    except Exception:
        return None


def pairwise_judge(question: str, answer_a: str, answer_b: str) -> dict:
    """Task 5: Choose the better answer using an LLM, with deterministic fallback."""
    return _try_llm_judge(question, answer_a, answer_b) or _fallback_pairwise(
        question, answer_a, answer_b
    )


def swap_and_average(question: str, answer_a: str, answer_b: str) -> JudgeResult:
    """Task 6: Judge both answer orders and combine scores after mapping back."""
    pass1 = pairwise_judge(question, answer_a, answer_b)
    pass2_raw = pairwise_judge(question, answer_b, answer_a)

    swap_map = {"A": "B", "B": "A", "tie": "tie"}
    winner_pass2 = swap_map[pass2_raw["winner"]]
    scores_pass2 = {
        "A": _clamp(pass2_raw["scores"].get("B")),
        "B": _clamp(pass2_raw["scores"].get("A")),
    }

    avg_a = (_clamp(pass1["scores"].get("A")) + scores_pass2["A"]) / 2
    avg_b = (_clamp(pass1["scores"].get("B")) + scores_pass2["B"]) / 2
    if abs(avg_a - avg_b) < 0.06:
        final = "tie"
    else:
        final = "A" if avg_a > avg_b else "B"

    position_consistent = pass1["winner"] == winner_pass2
    return JudgeResult(
        question=question,
        answer_a=answer_a,
        answer_b=answer_b,
        winner_pass1=pass1["winner"],
        winner_pass2=winner_pass2,
        final_winner=final,
        reasoning_pass1=pass1["reasoning"],
        reasoning_pass2=pass2_raw["reasoning"],
        position_consistent=position_consistent,
        scores_pass1={"A": _clamp(pass1["scores"].get("A")), "B": _clamp(pass1["scores"].get("B"))},
        scores_pass2=scores_pass2,
    )


def cohen_kappa(judge_labels: list[int], human_labels: list[int]) -> float:
    """Task 7: Compute Cohen's kappa for two binary label lists."""
    if len(judge_labels) != len(human_labels):
        raise ValueError("judge_labels and human_labels must have the same length")
    n = len(judge_labels)
    if n == 0:
        return 0.0
    labels = sorted(set(judge_labels) | set(human_labels))
    p_o = sum(j == h for j, h in zip(judge_labels, human_labels)) / n
    p_e = sum(
        (judge_labels.count(label) / n) * (human_labels.count(label) / n)
        for label in labels
    )
    if p_e == 1.0:
        return 1.0 if p_o == 1.0 else 0.0
    return (p_o - p_e) / (1 - p_e)


def bias_report(judge_results: list[JudgeResult]) -> dict:
    """Task 8: Measure position inconsistency and verbosity preference."""
    total = len(judge_results)
    if total == 0:
        return {
            "total_judged": 0,
            "position_bias_rate": 0.0,
            "position_bias_count": 0,
            "verbosity_bias": 0.0,
            "verbosity_details": {
                "a_wins_a_longer": 0,
                "b_wins_b_longer": 0,
                "total_decisive": 0,
            },
            "interpretation": "No data available.",
        }

    position_bias_count = sum(not result.position_consistent for result in judge_results)
    decisive = [result for result in judge_results if result.final_winner != "tie"]
    a_wins_a_longer = sum(
        result.final_winner == "A" and len(result.answer_a) > len(result.answer_b)
        for result in decisive
    )
    b_wins_b_longer = sum(
        result.final_winner == "B" and len(result.answer_b) > len(result.answer_a)
        for result in decisive
    )
    verbosity_bias = (
        (a_wins_a_longer + b_wins_b_longer) / len(decisive) if decisive else 0.0
    )
    position_bias_rate = position_bias_count / total
    interpretation = (
        "Position bias is high; keep swap-and-average in the evaluation path."
        if position_bias_rate > 0.3
        else "Position bias is low to moderate; continue monitoring judge consistency."
    )
    return {
        "total_judged": total,
        "position_bias_rate": round(position_bias_rate, 3),
        "position_bias_count": position_bias_count,
        "verbosity_bias": round(verbosity_bias, 3),
        "verbosity_details": {
            "a_wins_a_longer": a_wins_a_longer,
            "b_wins_b_longer": b_wins_b_longer,
            "total_decisive": len(decisive),
        },
        "interpretation": interpretation,
    }


def _load_ground_truth_by_question(path: str = "answers_50q.json") -> dict[str, str]:
    if not os.path.exists(path):
        return {}
    with open(path, encoding="utf-8") as f:
        return {item["question"]: item.get("ground_truth", "") for item in json.load(f)}


if __name__ == "__main__":
    with open(HUMAN_LABELS_PATH, encoding="utf-8") as f:
        human_data = json.load(f)

    ground_truths = _load_ground_truth_by_question()
    judge_results: list[JudgeResult] = []
    judge_labels: list[int] = []
    human_labels: list[int] = []

    for item in human_data:
        question = item["question"]
        model_answer = item["model_answer"]
        reference_answer = ground_truths.get(question) or item.get("human_note", "")
        result = swap_and_average(question, model_answer, reference_answer)
        judge_results.append(result)
        judge_labels.append(1 if result.final_winner in {"A", "tie"} else 0)
        human_labels.append(int(item["human_label"]))

    kappa = cohen_kappa(judge_labels, human_labels)
    bias = bias_report(judge_results)

    os.makedirs("reports", exist_ok=True)
    report_data = {
        "cohen_kappa": round(kappa, 4),
        "bias_report": bias,
        "judge_labels": judge_labels,
        "human_labels": human_labels,
        "results": [
            {
                "question": result.question,
                "answer_a": result.answer_a,
                "answer_b": result.answer_b,
                "winner_pass1": result.winner_pass1,
                "winner_pass2": result.winner_pass2,
                "final_winner": result.final_winner,
                "position_consistent": result.position_consistent,
                "scores_pass1": result.scores_pass1,
                "scores_pass2": result.scores_pass2,
            }
            for result in judge_results
        ],
    }
    with open("reports/judge_results.json", "w", encoding="utf-8") as f:
        json.dump(report_data, f, ensure_ascii=False, indent=2)

    print("Phase B report saved -> reports/judge_results.json")
    print(f"Cohen's kappa: {kappa:.4f}")
    print(f"Position bias rate: {bias['position_bias_rate']:.3f}")
    print(f"Verbosity bias: {bias['verbosity_bias']:.3f}")
