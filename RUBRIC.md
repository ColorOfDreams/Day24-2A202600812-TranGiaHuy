# Rubric — Lab 24: Production Eval + Guardrail Stack

**Tổng: 100 điểm + 10 bonus**

---

## Phase A: RAGAS Production Eval (30 điểm)

| Task | Tiêu chí | Điểm |
|---|---|---|
| 1 | `group_by_distribution()` trả về 3 keys đúng số lượng (20/20/10) | 5 |
| 2 | `run_ragas_50q()` trả về 50 `RagasResult` với 4 metric mỗi câu | 10 |
| 3 | `bottom_10()` sort đúng thứ tự, có đủ keys (rank, diagnosis, suggested_fix) | 7 |
| 4 | `cluster_analysis()` có matrix 4×3 đúng và insight string có nghĩa | 8 |

**Bonus Phase A (+4):** Adversarial distribution có avg_score thấp hơn factual (mong đợi) → pipeline detect được version conflicts

---

## Phase B: LLM-as-Judge (35 điểm)

| Task | Tiêu chí | Điểm |
|---|---|---|
| 5 | `pairwise_judge()` trả về dict với winner ∈ {A,B,tie} + reasoning không rỗng | 10 |
| 6 | `swap_and_average()` chạy đúng 2 passes, convert winner_pass2 về space gốc | 10 |
| 7 | `cohen_kappa()` trả về giá trị đúng [-1, 1], perfect agreement → 1.0 | 10 |
| 8 | `bias_report()` tính đúng position_bias_rate và verbosity_bias | 5 |

**Bonus Phase B (+3):** κ > 0.6 (substantial agreement giữa LLM judge và human labels)

---

## Phase C: NeMo Guardrails (35 điểm)

| Task | Tiêu chí | Điểm |
|---|---|---|
| 9a | `pii_scan()` detect VN_CCCD (12 số) và VN_PHONE (0[3-9]xxxxxxxx) | 10 |
| 9b | `check_input_rail()` async, trả về dict với allowed + blocked_reason | 5 |
| 10 | Adversarial suite ≥ 15/20 passed (≥75% pass rate) | 10 |
| 11 | `check_output_rail()` trả về safe + final_answer | 5 |
| 12 | `measure_p95_latency()` trả về đúng structure, P95 có nghĩa | 5 |

**Task 13 — Blueprint (đánh giá riêng, không tính vào 100 điểm trên):**
- Blueprint điền đầy đủ 4 sections: +2 điểm mỗi section = 8 điểm
- P95 latency thực tế được điền từ kết quả đo: +2 điểm

**Bonus Phase C (+3):** Adversarial suite ≥ 18/20 passed (≥90%)

---

## Trừ điểm

| Lỗi | Trừ |
|---|---|
| Còn `# TODO` trong phase_*.py | -5/module |
| Test fail | -3/test |
| `reports/blueprint.md` trống | -10 |
| `answers_50q.json` không tồn tại | Phase A không chạy được → 0 |

---

## Bonus Summary

| Bonus | Điều kiện | Điểm |
|---|---|---|
| Phase A | Adversarial avg_score < factual avg_score | +4 |
| Phase B | Cohen's κ > 0.6 với human labels | +3 |
| Phase C | Adversarial pass rate ≥ 18/20 | +3 |
| **Tổng bonus** | | **+10** |
