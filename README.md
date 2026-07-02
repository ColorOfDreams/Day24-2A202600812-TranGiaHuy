# Lab 24 - Production Eval + Guardrail Stack

**Sinh viên:** Trần Gia Huy  
**Ngày hoàn thiện:** 2026-07-01  
**Trạng thái:** Hoàn thành, `pytest` pass toàn bộ và `check_lab.py` đạt `22/22`.

## Tổng Quan

Dự án xây dựng một stack đánh giá và bảo vệ cho RAG pipeline HR policy tiếng Việt. Pipeline gốc từ Day 18 được mở rộng với ba lớp chính:

| Phase | Mục tiêu | Output |
|---|---|---|
| Phase A | Đánh giá RAG bằng RAGAS trên 50 câu hỏi thuộc 3 phân phối | `reports/ragas_50q.json` |
| Phase B | LLM-as-Judge, swap-and-average, Cohen's kappa và bias report | `reports/judge_results.json` |
| Phase C | Guardrails bằng PII scan, input/output rail và adversarial suite | `reports/guard_results.json` |

Ngoài code, dự án có thêm phân tích lỗi và blueprint triển khai production trong `analysis/` và `reports/blueprint.md`.

## Cấu Trúc Dự Án

```text
.
├── src/
│   ├── m1_chunking.py
│   ├── m2_search.py
│   ├── m3_rerank.py
│   ├── m4_eval.py
│   ├── m5_enrichment.py
│   ├── pipeline.py
│   ├── phase_a_ragas.py
│   ├── phase_b_judge.py
│   └── phase_c_guard.py
├── tests/
│   ├── test_phase_a.py
│   ├── test_phase_b.py
│   └── test_phase_c.py
├── data/
├── guardrails/
├── reports/
│   ├── blueprint.md
│   ├── ragas_50q.json
│   ├── judge_results.json
│   └── guard_results.json
├── analysis/
│   ├── failure_clusters.md
│   └── bias_report.md
├── answers_50q.json
├── check_lab.py
├── requirements.txt
└── docker-compose.yml
```

## Cài Đặt

Yêu cầu:

- Python 3.11+
- Docker để chạy Qdrant
- OpenAI-compatible API key nếu muốn gọi model thật

```bash
python -m pip install -r requirements.txt
python -m spacy download en_core_web_lg
docker compose up -d
```

Tạo file `.env` từ mẫu nếu cần chạy các bước có gọi LLM:

```bash
cp .env.example .env
```

Sau đó điền `OPENAI_API_KEY` trong `.env`.

## Cách Chạy

Sinh câu trả lời cho bộ 50 câu hỏi:

```bash
python setup_answers.py
```

Chạy từng phase:

```bash
python src/phase_a_ragas.py
python src/phase_b_judge.py
python src/phase_c_guard.py
```

Kiểm tra toàn bộ trước khi nộp:

```bash
python check_lab.py
```

## Chạy Test

Chạy toàn bộ test suite:

```bash
python -m pytest tests -q
```

Chạy theo từng phase:

```bash
python -m pytest tests/test_phase_a.py -q
python -m pytest tests/test_phase_b.py -q
python -m pytest tests/test_phase_c.py -q
```

Kết quả kiểm tra gần nhất:

```text
40 passed
check_lab.py: 22/22 checks passed
```

## Kết Quả Chính

### Phase A - RAGAS

| Distribution | Count | Avg score |
|---|---:|---:|
| factual | 20 | 0.8850 |
| multi_hop | 20 | 0.7375 |
| adversarial | 10 | 0.4475 |

Nhận xét chính:

- Nhóm factual đạt điểm tốt nhất.
- Nhóm adversarial yếu nhất và chiếm toàn bộ bottom-10.
- Worst metric tổng thể là `context_precision`, cho thấy retriever còn lấy nhiều context nhiễu, đặc biệt khi corpus có nhiều phiên bản policy cũ/mới.

Chi tiết nằm trong:

- `reports/ragas_50q.json`
- `analysis/failure_clusters.md`

### Phase B - LLM-as-Judge

| Metric | Value |
|---|---:|
| Cohen's kappa | 0.8000 |
| Position bias rate | 0.00 |
| Verbosity bias | 1.00 |

Nhận xét chính:

- Judge đã align tốt hơn với human labels sau khi bỏ hardcode và dùng rubric fallback.
- Position bias hiện bằng 0 trên bộ 10 nhãn kiểm thử.
- Verbosity bias vẫn cao, nên khi production cần tiếp tục ưu tiên correctness/citation hơn độ dài.

Chi tiết nằm trong:

- `reports/judge_results.json`
- `analysis/bias_report.md`

### Phase C - Guardrails

| Metric | Value |
|---|---:|
| Adversarial pass rate | 20/20 |
| Guard P95 latency | 0.21 ms |
| Latency budget | 500 ms |

Nhận xét chính:

- Guardrail chặn đúng toàn bộ 20/20 adversarial cases trong bộ kiểm thử.
- PII scan phát hiện được CCCD, số điện thoại Việt Nam và email.
- Local input rail chạy trước NeMo/LLM nên P95 latency nằm dưới budget.

Chi tiết nằm trong:

- `reports/guard_results.json`
- `reports/blueprint.md`

## Deliverables

Các phần đã hoàn thành:

- `src/phase_a_ragas.py`: group distribution, chạy RAGAS 50 câu, bottom-10 và cluster analysis.
- `src/phase_b_judge.py`: pairwise judge, swap-and-average, Cohen's kappa và bias report.
- `src/phase_c_guard.py`: PII scan, input/output rail, adversarial suite và latency measurement.
- `reports/blueprint.md`: CI/CD blueprint cho RAG eval và guardrail stack.
- `analysis/failure_clusters.md`: phân tích lỗi Phase A.
- `analysis/bias_report.md`: phân tích bias Phase B.
- `answers_50q.json` và các report JSON đã được sinh sẵn để kiểm tra lại.

## Ghi Chú Production

Dự án đã pass test lab, nhưng nếu đưa vào production nên ưu tiên cải thiện:

- Thêm metadata filter theo version/effective date để giảm nhiễu retrieval.
- Tách rule-based guard cho intent rủi ro cao trước khi gọi LLM rail.
- Cache kết quả guardrail cho các truy vấn lặp lại.
- Cải thiện judge bằng rubric rõ hơn, citation checking và consensus nhiều prompt/model.
