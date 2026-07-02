# CI/CD Blueprint: RAG Eval + Guardrail Stack

**Sinh viên:** Trần Gia Huy  
**Ngày:** 2026-07-01

## Guard Stack Pipeline

| Layer | Tool | P95 Latency | Failure Action |
|---|---|---:|---|
| PII Detection | Presidio + VN regex fallback | 0.08 ms | Reject + log |
| Topic/Jailbreak | Local normalized rule rail, then NeMo-compatible rail | 0.16 ms | Block + safe refusal |
| RAG Pipeline | Day 18 pipeline | Target < 2000 ms | Fallback answer |
| Output Check | NeMo-compatible output rail | Target < 300 ms | Replace with safe response |
| Total Guard | PII + input rail | 0.21 ms | Must stay < 500 ms |

## CI Gates

- RAGAS total questions must equal 50.
- RAGAS report must include `per_distribution`, `failure_clusters`, and `bottom_10`.
- `pytest tests/` must pass.
- Adversarial suite pass rate must be at least 15/20; current result is 20/20.
- Guard P95 latency must be less than 500 ms; current result is 0.21 ms.
- Cohen's kappa should be monitored; current result is 0.8000.

## Monitoring

| Metric | Current | Alert Threshold | Action |
|---|---:|---:|---|
| RAGAS avg score | 0.7385 | < 0.65 | Review retrieval and prompts |
| Worst RAGAS metric | context_precision | n/a | Add metadata filter/reranking |
| Dominant weak distribution | adversarial | n/a | Add version and negation tests |
| Adversarial pass rate | 20/20 | < 18/20 | Add/adjust guard rules |
| Guard P95 latency | 0.21 ms | > 500 ms | Profile rail stack |
| Cohen's kappa | 0.8000 | < 0.60 | Review judge rubric |

## Production Notes

The final guard stack now uses a deterministic normalized rule rail before any NeMo or LLM-backed rail. This keeps latency low and catches the known adversarial classes reliably: PII injection, jailbreak, off-topic requests, and prompt injection.

For production, keep the local rail as the fast first pass, then send uncertain inputs to a stronger NeMo/LLM classifier. RAG quality should be improved separately by adding metadata filters for policy version/effective date and strengthening reranking for adversarial or multi-hop questions.
