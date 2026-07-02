# LLM Judge Bias Report - Phase B

**Sinh viên:** Trần Gia Huy  
**Ngày:** 2026-07-01  
**Judge path:** LLM judge when API is available, deterministic rubric fallback when offline.

## Summary

| Metric | Value |
|---|---:|
| Total judged | 10 |
| Cohen's kappa | 0.8000 |
| Position bias rate | 0.000 |
| Position bias count | 0 |
| Verbosity bias | 1.000 |

## Judge Labels vs Human Labels

| # | Human | Judge | Agreement |
|---:|---:|---:|---|
| 1 | 1 | 1 | Yes |
| 2 | 0 | 0 | Yes |
| 3 | 1 | 0 | No |
| 4 | 1 | 1 | Yes |
| 5 | 1 | 1 | Yes |
| 6 | 0 | 0 | Yes |
| 7 | 1 | 1 | Yes |
| 8 | 0 | 0 | Yes |
| 9 | 1 | 1 | Yes |
| 10 | 0 | 0 | Yes |

Agreement is strong overall. The one mismatch is a case where the short model answer is correct but the reference answer is more complete, so the rubric prefers the longer answer.

## Position Bias

The current swap-and-average implementation maps the swapped pass back to the original answer IDs and averages scores. On the 10 labeled examples, both passes agree after mapping.

```text
position_bias_rate = 0 / 10 = 0.000
```

## Verbosity Bias

Verbosity bias remains high because all decisive wins favor the longer answer.

```text
verbosity_bias = 5 / 5 = 1.000
```

This is not automatically wrong because many references are longer and include missing policy conditions. Still, in production the judge should explicitly prioritize correctness and cited evidence over length.

## Production Recommendation

- Keep swap-and-average enabled.
- Keep rubric fallback deterministic for offline CI.
- Add citation/evidence checks so a concise but correct answer is not penalized simply for being short.
- Review low-confidence cases manually or with a second judge prompt.
