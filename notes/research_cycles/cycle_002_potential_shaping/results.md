# Cycle 002 Results

Date: June 23, 2026

## What Changed From The Attachment

The attachment reframed the research correctly:

- The broad evaluative-geometry hypothesis is plausible.
- The HH disagreement audit is suggestive, not decisive accuracy.
- The table-backed audit count is 63 `EMBEDDING_RIGHT`, 45 `HH_RIGHT`, and
  123 `EXCLUDE`, not the earlier 65/44/122 prose summary.
- Among table-backed gradeable disagreements, embedding wins 63/108 = 58.3%.
- Exact two-sided binomial p-value vs 50%: 0.101.
- The strongest original direction is cumulative full-context potential
  shaping, not naive final-answer scoring.

## Controlled Battery V0

File:

- `controlled_evaluative_axis_battery.jsonl`

Size:

- 23 hand-authored minimal pairs.

Length diagnostics:

- Mean absolute length gap: 3.57 words.
- Max absolute length gap: 12 words.
- Better answer longer in 20/23 cases.

Result:

- Length baseline: 89.1%.
- FastEmbed/BGE-small direct combined: 26.1%.
- FastEmbed/BGE-small decomposition category axis: 52.2%.

Interpretation:

V0 still leaked length. It is useful as a failure of experimental design, not as
evidence for the model or thesis.

## Controlled Battery V2: Exact Length-Matched

File:

- `controlled_evaluative_axis_battery_v2_length_balanced.jsonl`

Size:

- 12 hand-authored minimal pairs.

Length diagnostics:

- Mean absolute length gap: 0.00 words.
- Max absolute length gap: 0 words.
- All 12 pairs have exact word-count ties.

FastEmbed/BGE-small results:

| Method | Accuracy |
| --- | ---: |
| Length | 50.0% |
| Sentiment | 50.0% |
| Refusal heuristic | 58.3% |
| Direct combined | 8.3% |
| Direct broad evaluative | 0.0% |
| Direct anti-sycophancy | 66.7% |
| Decomposition combined | 41.7% |
| Decomposition category axis | 58.3% |
| Decomposition harm reduction | 58.3% |

Lexical debugging results:

| Method | Accuracy |
| --- | ---: |
| Length | 50.0% |
| Sentiment | 50.0% |
| Refusal heuristic | 58.3% |
| Direct combined | 87.5% |
| Decomposition combined | 75.0% |

The lexical result is a debugging artifact because the battery text and lexical
keywords overlap by construction. It shows the harness can detect the intended
labels. It is not model evidence.

## Oracle-Decomposition Sanity Check After Category-Map Fix

While extending the no-quota tests, the harness exposed a real bug: the v2
battery used category names such as `truthfulness`, `harm_reduction`,
`reasoning_rigor`, `context_binding`, and `helpfulness`, but the scorer's
category-axis map still covered mostly older category names such as
`factuality`, `harmful_request`, and `general_helpfulness`. As a result, some
earlier `category_axis` numbers silently fell back to the broad
`general_evaluative` axis.

The map was fixed in `scripts/run_cycle001_intervention.py`, then the exact
same 12-case length-balanced battery was rerun across 8 FastEmbed models. Model
weights were cached outside the repo in `C:\Users\93rob\.cache\codex-fastembed`.

File:

- `fastembed_model_sweep_v2/summary.md`

Corrected key results on the oracle-decomposition interface:

| Model | Best axis | Best accuracy | Direct combined | Direct category | Decomp category |
| --- | --- | ---: | ---: | ---: | ---: |
| `BAAI/bge-small-en-v1.5` | `decomposition_anti_sycophancy` | 66.7% | 8.3% | 25.0% | 58.3% |
| `BAAI/bge-base-en-v1.5` | `decomposition_category_axis` | 58.3% | 8.3% | 41.7% | 58.3% |
| `thenlper/gte-base` | `direct_harm_reduction` | 75.0% | 8.3% | 41.7% | 50.0% |
| `snowflake/snowflake-arctic-embed-m` | `direct_persona_honesty` | 75.0% | 33.3% | 50.0% | 58.3% |
| `jinaai/jina-embeddings-v2-small-en` | `decomposition_category_axis` | 91.7% | 25.0% | 50.0% | 91.7% |
| `jinaai/jina-embeddings-v2-base-en` | `decomposition_harm_reduction` | 50.0% | 25.0% | 41.7% | 41.7% |
| `nomic-ai/nomic-embed-text-v1.5-Q` | `decomposition_harm_reduction` | 75.0% | 16.7% | 25.0% | 33.3% |
| `mixedbread-ai/mxbai-embed-large-v1` | `decomposition_anti_sycophancy` | 66.7% | 8.3% | 25.0% | 66.7% |

The high number is `jinaai/jina-embeddings-v2-small-en` at 11/12 = 91.7% on
decomposition plus category-specific axes. This should not be interpreted as
evidence that the embedding model inferred answer quality from the raw answer.
The decomposition fields were hand-authored and included explicit "Good parts"
and "Bad parts" language, so this is an oracle/leakage sanity check.

The valid interpretation is much weaker:

> If explicit evaluative labels are put into the input, the embedding projection
> can read that evaluative language. This is a plumbing/upper-bound check, not
> a measurement result for the proposed evaluator.

## 50-Prompt Proxy Pilot With Jina

The same Jina model was also run on the existing 50-prompt, 4-candidate pilot:

- `cycle_001_next/pilot_50_fastembed_jina_v2_small/summary.md`

Result:

- Length baseline: 33/50 = 66.0%.
- Best embedding method: `decomposition_persona_honesty`, 23/50 = 46.0%.
- `decomposition_category_axis`: 15/50 = 30.0%.

Interpretation:

This does not refute the controlled-battery result because the 50-prompt pilot
uses proxy labels and known length-biased/generated candidates. It does show
that the old pilot is not a clean validation target. The next candidate
intervention set must be length-balanced and blind-adjudicated before its hit
rates mean much.

## Naive Trajectory-Potential Probe

A first process-scoring probe was added:

- `scripts/run_trajectory_potential_test.py`
- `trajectory_potential_bge_small/summary.md`
- `trajectory_potential_jina_v2_small/summary.md`

It scores cumulative trajectory states: strategy, evaluation, and final answer.

Results:

| Model | Category-axis integral | Combined integral |
| --- | ---: | ---: |
| `BAAI/bge-small-en-v1.5` | 25.0% | 0.0% |
| `jinaai/jina-embeddings-v2-small-en` | 33.3% | 0.0% |

Interpretation:

The naive process-state version fails. That matters because it prevents a
sloppy conclusion that "any reasoning-looking scratchpad makes the axis work."
The current oracle-decomposition result is not positive evidence for the
evaluator because it leaks the intended judgment into the scored text. A serious
potential-shaping test needs natural model-generated trajectories,
injected-error/repair cases, and controlled stage texts, not the generic
template used here.

## Interpretation

This is a real negative diagnostic for the current small-model/direct-axis
setup:

- exact length control removed the length shortcut;
- local BGE-small broad good/bad scoring failed badly;
- narrow anti-sycophancy did better than the broad evaluative axis;
- decomposition/category scoring with hand-authored "Good parts"/"Bad parts"
  text became high for Jina-small, but this is label leakage and must not be
  promoted as evaluator evidence.

This supports the scalar-plus-basis framing and argues against claiming that a
small open embedding model plus a naive broad axis is already an adequate
universal evaluator.

## Decision

The project should proceed, but the next gate is not another HH agreement run.

Next credible steps:

1. Expand v2 to 50 exact or near-exact length-matched cases.
2. Add more phenomena: sunk cost, contradiction, quotation vs endorsement,
   irony, tool-result overclaiming, generic-template reward hacking, and repair
   after error.
3. Run the same battery with Gemini when quota is available.
4. Build a better trajectory version where a known bad step is injected and
   later repaired, then test cumulative potential deltas.
5. Do not use hand-authored oracle decompositions as evidence except as a
   leakage/plumbing sanity check.

The strongest current claim is:

> Broad evaluative geometry is plausible and supported by prior work, but local
> BGE-small direct broad-axis scoring fails under exact length control. The
> 91.7% Jina-small oracle-decomposition number is contaminated by explicit
> good/bad labels and should not be used as evidence of independent evaluation.
> The next valid question is whether blind, label-free decomposition or raw
> answer scoring beats cheap baselines on a larger held-out battery.
