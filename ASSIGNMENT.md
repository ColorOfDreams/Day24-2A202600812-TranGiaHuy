# Lab 24: Production Eval + Guardrail Stack

**Thời gian:** 90 phút (3 phases × 30 phút) + 15 phút setup  
**Điểm:** 100 + 10 bonus

---

## Tổng quan

Xây dựng **complete eval + guardrail stack** trên RAG pipeline từ Day 18.

```
[Day 18 Pipeline]  →  Phase A: RAGAS 50q  →  ragas_50q.json
                   →  Phase B: LLM Judge   →  judge_results.json
                   →  Phase C: NeMo Guard  →  guard_results.json + blueprint.md
```

---

## Setup (15 phút — TRƯỚC khi bắt đầu tính giờ)

```bash
# 1. Copy src/ từ Day 18 của bạn
cp -r <path-to-day18>/src/m*.py src/
cp -r <path-to-day18>/src/pipeline.py src/

# 2. Start Qdrant
docker compose up -d

# 3. Install dependencies
pip install -r requirements.txt
python -m spacy download en_core_web_lg   # cho Presidio

# 4. Copy .env và điền API key
cp .env.example .env  # → điền OPENAI_API_KEY

# 5. Chạy setup để generate answers (mất ~5-10 phút)
python setup_answers.py
```

Sau bước 5, file `answers_50q.json` được tạo. **Đây là input cho Phase A.**

---

## Phase A: RAGAS Production Eval (30 phút)

**File:** `src/phase_a_ragas.py` · **Test:** `pytest tests/test_phase_a.py`

### Test set mới: 50 câu hỏi, 3 distributions

| Distribution | Số câu | Đặc điểm |
|---|---|---|
| `factual` | 20 | Single-doc lookup, câu hỏi thẳng |
| `multi_hop` | 20 | Cross-doc, tính toán, suy luận |
| `adversarial` | 10 | Version conflicts, negation traps, policy contradictions |

### Tasks

| Task | Hàm | Mô tả |
|---|---|---|
| 1 | `group_by_distribution()` | Group 50q theo distribution |
| 2 | `run_ragas_50q()` | Gọi `evaluate_ragas()` từ m4_eval.py của bạn trên 50 answers |
| 3 | `bottom_10()` | Sort theo avg_score, lấy 10 câu tệ nhất + diagnosis |
| 4 | `cluster_analysis()` | Matrix worst_metric × distribution, tìm dominant failure |

**Pass criteria:**
- [ ] 3 distributions đúng số lượng (20/20/10)
- [ ] `run_ragas_50q()` trả về 50 `RagasResult`
- [ ] `bottom_10()` có đủ keys: rank, diagnosis, suggested_fix
- [ ] `cluster_analysis()` có matrix + insight string

---

## Phase B: LLM-as-Judge (30 phút)

**File:** `src/phase_b_judge.py` · **Test:** `pytest tests/test_phase_b.py`

### Tasks

| Task | Hàm | Mô tả |
|---|---|---|
| 5 | `pairwise_judge()` | Gọi LLM chọn answer A hoặc B, trả về winner + reasoning + scores |
| 6 | `swap_and_average()` | Chạy 2 lần (swap order), phát hiện position bias |
| 7 | `cohen_kappa()` | So sánh judge labels vs 10 human labels trong `human_labels_10q.json` |
| 8 | `bias_report()` | Đo position bias rate + verbosity bias |

**Công thức Cohen's κ:**
```
p_o = observed agreement rate
p_e = expected agreement by chance
κ   = (p_o - p_e) / (1 - p_e)
```
Thang đo: `>0.6 = substantial`, `>0.8 = almost perfect`

**Pass criteria:**
- [ ] `pairwise_judge()` trả về winner ∈ {"A", "B", "tie"} + reasoning
- [ ] `swap_and_average()` phát hiện đúng position inconsistency
- [ ] `cohen_kappa()` trả về giá trị ∈ [-1, 1]
- [ ] `bias_report()` có position_bias_rate + verbosity_bias

---

## Phase C: NeMo Guardrails (30 phút)

**File:** `src/phase_c_guard.py` · **Test:** `pytest tests/test_phase_c.py`

### Stack kiến trúc

```
User Input
    │
    ▼
[Presidio]         ← Task 9a: Detect & anonymize PII (CCCD, phone, email)
    │ block if PII
    ▼
[NeMo Input Rail]  ← Task 9b: Block off-topic / jailbreak / prompt injection
    │ allow if OK
    ▼
[RAG Pipeline]     ← (Day 18 code — không cần thay đổi)
    │
    ▼
[NeMo Output Rail] ← Task 11: Check response trước khi trả về user
    │
    ▼
User Response
```

### Tasks

| Task | Hàm | Mô tả |
|---|---|---|
| 9a | `pii_scan()` | Presidio: detect VN_CCCD, VN_PHONE, EMAIL + anonymize |
| 9b | `check_input_rail()` | NeMo: async call, trả về allowed/blocked |
| 10 | `run_adversarial_suite()` | Chạy 20 inputs qua full stack, pass rate ≥ 15/20 |
| 11 | `check_output_rail()` | NeMo output rail: flag sensitive response |
| 12 | `measure_p95_latency()` | Đo P50/P95/P99 cho Presidio và NeMo riêng lẻ |

### NeMo Guardrails

Config đã được chuẩn bị sẵn trong `guardrails/`:
- `config.yml` — model (gpt-4o-mini) + rails config
- `rails.co` — Colang flows: jailbreak, off-topic, PII request, output check

Bạn **có thể mở rộng** `rails.co` để cải thiện pass rate trong Task 10.

### Task 13: CI/CD Blueprint (điền vào `reports/blueprint.md`)

Điền bảng sau vào `reports/blueprint.md`:

```markdown
## CI/CD Blueprint: RAG Eval + Guardrail Stack

### Guard Stack Pipeline
| Layer           | Tool          | Latency P95 | Failure Action |
|-----------------|---------------|-------------|----------------|
| PII Detection   | Presidio      | <10ms       | Reject + log   |
| Topic/Jailbreak | NeMo Input    | <300ms      | 503 + reason   |
| RAG Pipeline    | Day 18        | <2000ms     | Fallback       |
| Output Check    | NeMo Output   | <300ms      | Block + log    |

### CI Gates (phải pass trước khi merge to main)
- [ ] RAGAS faithfulness ≥ 0.75 (measured on 50q test set)
- [ ] Adversarial suite pass rate ≥ 90% (18/20)
- [ ] P95 total guard latency < 500ms

### Monitoring (điền dựa trên kết quả của bạn)
- P95 latency thực tế: ___ms
- Adversarial pass rate: ___/20
- Worst RAGAS metric: ___
- Dominant failure distribution: ___
```

**Pass criteria:**
- [ ] `pii_scan()` detect VN_CCCD và VN_PHONE
- [ ] Adversarial suite ≥ 15/20 passed
- [ ] `measure_p95_latency()` trả về đúng structure
- [ ] `reports/blueprint.md` được điền đầy đủ

---

## Deliverables (push lên GitHub)

```
Day24-Track3-Eval-Guard/
├── src/
│   ├── m1_chunking.py       ← copy từ Day 18
│   ├── m2_search.py         ← copy từ Day 18
│   ├── m3_rerank.py         ← copy từ Day 18
│   ├── m4_eval.py           ← copy từ Day 18 (★ cần implement xong)
│   ├── m5_enrichment.py     ← copy từ Day 18
│   ├── pipeline.py          ← copy từ Day 18
│   ├── phase_a_ragas.py     ★ implement Tasks 1-4
│   ├── phase_b_judge.py     ★ implement Tasks 5-8
│   └── phase_c_guard.py     ★ implement Tasks 9-12
├── reports/
│   ├── ragas_50q.json       ★ auto-generated (Phase A)
│   ├── judge_results.json   ★ auto-generated (Phase B)
│   ├── guard_results.json   ★ auto-generated (Phase C)
│   └── blueprint.md         ★ Task 13 (điền tay)
└── answers_50q.json         (generated by setup_answers.py)
```

### Trước khi nộp

```bash
pytest tests/ -v                   # Tất cả tests pass?
grep -r "# TODO" src/phase_*.py    # 0 TODOs remaining?
python check_lab.py                # Final check
```
