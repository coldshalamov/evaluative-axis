# Axis Correction Experiment — June 23, 2026

## Question

The convergence test (June 22) showed the harm_reduction axis is orthogonal
to the general evaluative direction (cosine -0.123). Does switching to the
general evaluative axis improve HH-RLHF preference prediction?

## Result: No. The opposite.

### Response-only scoring (300 pairs, BGE-small, HH-RLHF test split)

| Axis | Agreement | z | p |
|---|---|---|---|
| harm_reduction | **56.3%** | 2.19 | 0.016* |
| domain_parenting | 51.0% | 0.35 | 0.39 |
| domain_friendship | 50.0% | 0.00 | 1.00 |
| mean_evaluative | 48.7% | -0.46 | n.s. |
| simple_good_bad | 48.3% | -0.58 | n.s. |
| domain_code_quality | 45.3% | -1.62 | n.s. |

### Prompt+response scoring (300 pairs, BGE-small, HH-RLHF test split)

| Axis | Agreement | z | p |
|---|---|---|---|
| harm_reduction | **55.7%** | 1.96 | 0.028* |
| domain_engineering | 51.7% | 0.58 | 0.30 |
| mean_evaluative | 49.0% | -0.35 | n.s. |
| simple_good_bad | 49.3% | -0.23 | n.s. |
| domain_ethics | 45.0% | -1.73 | n.s. |

## Interpretation

The harm_reduction axis outperforms the general evaluative axis on HH-RLHF.
The general evaluative axes (simple good/bad, mean of 10 domains) score at
or below chance.

This makes sense: HH-RLHF is primarily a **safety** dataset. Most preference
pairs distinguish between refusal/safety-conscious responses and
compliance/harmful responses. The harm_reduction axis directly targets this
distinction. The general evaluative axis captures broad "quality" — but on
HH-RLHF, both responses often have similar general quality, differing
specifically on safety behavior.

## Implications

1. **The harm_reduction axis was not "the wrong axis" for HH-RLHF.** It was
   the right axis for a safety-focused dataset. The general evaluative axis
   captures a different dimension that HH-RLHF doesn't discriminate on.

2. **Safety and general quality are partially orthogonal dimensions.** The
   convergence test showed safety is geometrically independent from general
   evaluation. This experiment confirms they also discriminate on different
   datasets.

3. **HH-RLHF cannot validate the general evaluative axis.** The dataset
   measures safety preference, not general quality. To validate the general
   evaluative axis, we need datasets that discriminate on quality rather than
   safety (e.g., writing quality, code quality, reasoning correctness).

4. **A multi-axis approach may be justified.** Safety and general quality
   provide independent signals. An alignment reward could use both:
   `score = w1 * evaluative + w2 * safety`.

## Note on earlier convergence test HH results

The session summary from June 22 reported that domain axes (writing: 56.7%,
cooking: 56.0%, etc.) outperformed harm_reduction (48.7%) on HH-RLHF. This
current experiment finds the opposite pattern. Possible explanations:
- Different 300-pair sample (random sampling vs. sequential test split)
- Different axis definitions
- Small-sample variance (300 pairs, effect sizes near zero)

The current results are reproducible from the first 300 pairs of the test
split. The earlier results should be treated with caution until reproduced.
