# The Evaluative Axis: Embedding Geometry as a Quality Signal for LLM Alignment

Robin Gattis, June 2026

## What This Is

Evaluation (good/bad) is the primary dimension of human semantic judgment
(Osgood et al., 1957), and this structure is recoverable from embedding
geometry (Grand et al., 2022; Kozlowski et al., 2025). We test whether
projecting text onto evaluative directions in embedding space can serve as
a cheap alignment signal --- without training a classifier, without labeled
preference data, and without LLM inference.

## Current Results

**Objective reranking** (verifiable end metrics, Gemini Embedding 2):

| Domain | N | Random | Length | Embedding | Cheap OSS best |
|---|---:|---:|---:|---:|---:|
| Code | 6 | 50.0% | 50.0% | 83.3% | 50.0% |
| Math | 8 | 37.5% | 50.0% | 100.0% | 62.5% |
| Tool | 8 | 37.5% | 37.5% | 87.5% | 50.0% |

All suites are 3-way selection; p-values against 1/3 random: code p=0.018,
math p < 0.001, tool p=0.003. Length baseline is tiebreak-sensitive on tasks
with equal-word-count candidates.

Length confound analysis on the code suite: r(length, quality) = 0.19;
r(score, quality) = 0.60. Length selection scores 3/6 (50%), equal to random.
On the 2 tasks with a uniquely longest wrong candidate, embedding succeeds on
both.

**50-case length-balanced conflict battery** (exact word-count matching):

| Method | Accuracy |
|---|---:|
| Raw `good/bad` | 26% |
| Best proxy word (`useful/useless`) | 42% |
| Targeted axis (anti-sycophancy) | 98% |
| Targeted axis (persona honesty) | 96% |
| Targeted axis (harm reduction) | 94% |
| Targeted axis (truthfulness) | 90% |
| Combined targeted axes | 86% |

**Signal concentration gap**: On individual targeted axes, local OSS models
(384--1024 dim) score 50--74% vs Gemini's 86--98%, and this does not improve
with model size: Qwen3-Embedding-0.6B (600M params, 1024d) performs comparably
to 33M-parameter models. Multi-axis PCA does not yield a usable evaluative
direction — no method-internal orientation rule recovers the correct sign on
local models (principled orientation gives 16--28%). The frontier advantage
does not close with scale in the 33M–600M range; its cause is unidentified
(Gemini's parameter count is undisclosed). On expanded
objective suites (48 math, 32 tool), individual-axis OSS scoring is at
or below baseline.

**Random-axis null control**: Individual targeted axes are mostly in the noise
band on local models (only Snowflake-M at 99.5th percentile). Multi-axis PCA
yields a negative result: while centered-PC1 of targeted axes separates from
a matched null (PC1-of-5-random-axes) on 3/5 models, no method-internal
orientation rule recovers the correct sign — principled orientation gives
16--28% accuracy, below chance. The 72--84% numbers depend on post-hoc sign
selection using labels. Local models do not yield a usable zero-shot evaluative
direction even with PCA.

**Process-potential scoring**: On 12 injected error/repair traces, Gemini
category-axis scoring detects 91.7% of error drops and 83.3% of repair
rises, beating length (0%/100%), sentiment (42%/17%), and final-answer-only
(0%/0%). Dense localization score is 50%, below the frozen 65% training gate.

**HH-RLHF disagreement audit**: 231 cases where embedding disagreed with
dataset labels. Among gradeable disagreements, embedding was right 58.3%
of the time. Corrected agreement: 83--88%.

## Honest Negatives

- Raw one-word `good/bad` **fails** as a zero-shot evaluator (26% on the
  50-case battery). Targeted multi-sentence axes are required.
- Cheap OSS embedders fail on individual targeted axes regardless of model
  size (33M--600M params). Multi-axis PCA is also a negative result: no
  method-internal orientation rule recovers the correct sign on local models.
- The training-readiness gate has not been met. Process scoring is promising
  but not yet sharp enough for dense reward.
- Sycophancy and honesty-hedging are structural blind spots for any
  surface-text evaluation method.
- The HH disagreement audit was not blind; provisional until multi-annotator
  adjudication.

## Reproducing

Requirements: Python 3.10+, a Google API key for Gemini experiments.

```bash
# Install all dependencies
pip install -r requirements.txt

# Or install only what you need for local experiments (free, no API key)
pip install fastembed numpy

# Battery sweep (8 local models, ~5 minutes)
python scripts/sweep_fastembed_battery.py --interfaces direct

# Word-stripping ablation
python scripts/run_word_stripping_ablation.py --backend fastembed

# Random-axis null control (verifies signal is axis-specific)
python scripts/run_random_axis_control.py --n-random 200

# Anchor perturbation (tests axis robustness to anchor rewording)
python scripts/run_anchor_perturbation.py --backend fastembed

# Cross-category transfer and PCA of axis geometry
python scripts/run_cross_category_transfer.py --backend fastembed

# Larger open models via sentence-transformers (PC1 analysis)
pip install sentence-transformers einops
python scripts/run_large_open_model_battery.py --model BAAI/bge-large-en-v1.5
python scripts/run_large_open_model_battery.py --model nomic-ai/nomic-embed-text-v1.5
python scripts/run_large_open_model_battery.py --model Qwen/Qwen3-Embedding-0.6B

# PC1 validation (sign orientation + matched null checks)
python scripts/validate_pc1_claims.py

# Statistical significance tests
python scripts/compute_significance.py

# Gemini experiments (requires GOOGLE_API_KEY in .env.local; subject to quota)
pip install google-genai python-dotenv
python scripts/run_evaluative_axis_battery.py --backend gemini
python scripts/run_objective_code_reranking.py   # uses curated candidate sets; see cycle_003 results.md
```

## Repository Structure

```
paper/
  draft.md                     Full paper draft

methodology/
  DECISIVE_EVIDENCE_PLAN.md    What counts as proof, what does not
  SERIOUS_RESEARCH_SYSTEM_V1.md  Frozen research program with 4 claims
  NO_LEAKAGE_DECOMPOSITION_PROTOCOL.md  Leakage prevention rules

scripts/
  run_evaluative_axis_battery.py    Score battery with any backend
  sweep_fastembed_battery.py        Sweep 8 local models on battery
  run_objective_code_reranking.py   Code reranking with hidden tests
  run_word_stripping_ablation.py    Evaluative word stripping ablation
  run_random_axis_control.py        Random-axis null control
  run_anchor_perturbation.py        Anchor perturbation robustness test
  run_cross_category_transfer.py    Cross-category transfer and axis PCA
  run_large_open_model_battery.py    Full battery on larger open models (sentence-transformers)
  validate_pc1_claims.py             PC1 sign orientation + matched null validation
  compute_significance.py           Binomial tests and Wilson CIs
  run_good_vs_proxy_conflicts.py    Raw good/bad vs proxy words
  run_process_potential_error_repair.py  Error/repair trace scoring

notes/research_cycles/
  cycle_002_*    Battery development and potential shaping
  cycle_003_*    Objective code reranking (Gemini + OSS)
  cycle_009_*    8-model OSS battery sweep
  cycle_010_*    Good-vs-proxy conflict sweep
  cycle_013_*    Expanded objective suites (48 math, 32 tool)

experiments/research_system_v1/   Objective benchmark specs and results
```

## Theoretical Chain

1. **Osgood et al. (1957)**: Evaluation is the primary dimension of human
   semantic judgment, cross-culturally universal.
2. **Grand et al. (2022)**: Semantic projection recovers human knowledge
   from embedding geometry (Nature Human Behaviour).
3. **Kozlowski et al. (2025)**: Osgood's dimensions exist in LLM embeddings;
   features are entangled along shared directions.
4. **Cho et al. (2026)**: LLMs conflate moral, grammatical, and economic
   "good" --- for alignment, this entanglement is the mechanism, not a bug.

## Key Claim Ladder

1. **Geometry exists** (supported): evaluative structure is present in
   embedding space across domains.
2. **Selection works** (supported on small suites): embedding scoring beats
   baselines on objective reranking with a capable model.
3. **Signal concentration gap** (supported): individual targeted axes fail on
   all local models tested (33M--600M params) regardless of scale. Multi-axis
   PCA is a negative result (sign not orientable). The frontier advantage
   does not close with scale in the 33M–600M range; cause is unidentified
   (Gemini's parameter count is undisclosed).
4. **Training readiness** (not yet met): process scoring is promising but
   below the frozen gate.

---

*Independent research. No institutional affiliation.*
