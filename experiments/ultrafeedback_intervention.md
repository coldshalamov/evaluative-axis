# UltraFeedback Intervention Test — June 23, 2026

## Design

UltraFeedback dataset: 4 model responses per prompt, each with GPT-4 quality
scores (1-10). We score each response with the embedding axis and test whether
higher axis scores predict higher GPT-4 scores.

100 prompts, 399 responses. Model: BGE-small-en-v1.5 (384-dim, 33M params).
Response text truncated to 800 chars (~BGE context limit).

## Results

### Rank correlation: embedding score vs GPT-4 quality score

| Axis | Spearman rho | p-value |
|---|---|---|
| simple_good_bad | **0.224** | <0.0001 |
| mean_evaluative | **0.197** | 0.0001 |
| harm_reduction | 0.074 | 0.14 (n.s.) |

The general evaluative axis significantly correlates with GPT-4 quality
scores. The harm_reduction axis does not.

### Best-of-N selection: does the axis pick the best response?

| Method | Exact match (picks GPT-4's best) | Mean GPT-4 score of pick |
|---|---|---|
| Oracle (best possible) | 100% | 7.73 |
| simple_good_bad | 31% | 6.64 |
| mean_evaluative | 30% | 6.66 |
| harm_reduction | 26% | 6.66 |
| Random | 25% (expected) | 6.64 |

The axis picks the GPT-4-best response 31% of the time vs 25% random
baseline (with 4 choices). However, the selected response does not have a
meaningfully higher average GPT-4 score than random selection.

### Pairwise accuracy

| Axis | Accuracy | z | p |
|---|---|---|---|
| simple_good_bad | 52.0% | 0.86 | 0.21 (n.s.) |
| mean_evaluative | 51.5% | 0.68 | 0.26 (n.s.) |
| harm_reduction | 51.8% | 0.77 | 0.23 (n.s.) |

Not significant for pairwise preference prediction.

## Key finding: cross-dataset axis specificity

| Dataset | Best axis | Signal | Worst axis |
|---|---|---|---|
| UltraFeedback (quality) | simple_good_bad (rho=0.224***) | General quality | harm_reduction (rho=0.074, n.s.) |
| HH-RLHF (safety) | harm_reduction (56.3%, p=0.016) | Safety | simple_good_bad (48.3%, n.s.) |

Each axis works on the dataset matching its domain. The general evaluative
axis captures quality. The harm reduction axis captures safety. They are
geometrically orthogonal (cosine = -0.124) and discriminate on different
datasets.

## Interpretation

1. **The evaluative signal exists in BGE-small but is too weak for practical
   best-of-N selection.** Rank correlation is significant but doesn't translate
   into meaningful quality improvement at n=4. This is consistent with the
   model capacity findings: BGE-small (33M params, 384-dim) scored ~56% on
   controlled pairs while Gemini Embedding 2 (3072-dim) scored 70.5%. A
   stronger embedding model should produce stronger rank correlation.

2. **Safety and quality are independent dimensions.** This is the most
   important finding of the session. The evaluative axis and harm axis
   measure different things, work on different datasets, and are geometrically
   orthogonal. An alignment reward signal may benefit from combining both.

3. **UltraFeedback validates the general evaluative axis on a non-safety
   dataset.** Unlike HH-RLHF (which is primarily safety), UltraFeedback
   measures general response quality across diverse tasks. The significant
   rank correlation (rho=0.224) confirms the axis captures general quality,
   not just safety.

## Next steps

- Re-run with Gemini Embedding 2 when API quota resets. If rho scales from
  0.224 to >0.4, best-of-N selection becomes practically useful.
- Test combined scoring: `w1 * evaluative + w2 * safety`. The independence
  suggests they provide complementary information.
