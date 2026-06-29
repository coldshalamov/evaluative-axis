# CLAUDE.md

## Session Start

Read these files before doing anything:
1. `methodology/RESEARCH_CONTEXT.md` — Environment, pitfalls, experiment template
2. `methodology/RESEARCH_DIRECTIONS.md` — Open questions and priorities
3. `methodology/EXPERIMENT_SPECS.md` — Ready-to-run experiment designs

The paper draft is at `paper/draft.md`.

## Environment

- Windows 11, use PowerShell for Python commands
- Python venv: `C:\Users\93rob\.cache\codex-embedding-venv\Scripts\python.exe`
- Only Google API key available (in `.env.local`), frequently quota-limited
- No GPU, 32GB RAM, all embedding models run on CPU
- Zero budget — no paid APIs, no hired annotators

## Three Golden Rules

1. **Balanced battery**: Always test on original 50 + warmth 20. Report both
   splits. The original battery alone is firmness-biased (64%).

2. **All three models**: snowflake/snowflake-arctic-embed-m, BAAI/bge-m3,
   nomic-ai/nomic-embed-text-v1.5. Always all three.

3. **Independent scoring**: Never average axis vectors. Score independently,
   sum or vote.

## User preferences

- Explain with concrete examples from data, not jargon
- Don't dump percentages without saying what they mean
- Don't theorize when you could run an experiment
- Don't ask permission — do the work
- Don't suggest arXiv, cold-emailing researchers, or anything requiring money
- Negative results are fine — report them honestly

## Key files

- `scripts/run_cycle001_intervention.py` — AXES dict (imported by many scripts)
- `scripts/run_centroid_deep.py` — Core centroid experiments (learning curve, bootstrap, permutation, etc.)
- `scripts/run_centroid_splits.py` — Cross-type transfer, firmness/warmth anti-correlation
- `scripts/run_centroid_confounds.py` — Length, norm, random direction confound checks
- `scripts/run_gemini_centroid.py` — Gemini centroid replication
- `notes/research_cycles/cycle_002_potential_shaping/controlled_evaluative_axis_battery_v3_50_cases.jsonl` — 50-case battery
- `notes/research_cycles/battery_rebalancing/warmth_cases.jsonl` — 20 warmth cases
- `notes/research_cycles/battery_expansion/` — 61 expansion test cases (5 JSONL files)
- `notes/research_cycles/centroid_deep/` — All centroid experiment results (JSON)

## Current state of knowledge

**The centroid approach is the central finding:**
- Supervised centroid (mean better - mean worse) gets 66-80% OOS on 61 cases (5 models), 86% Gemini (35 cases)
- Best: Jina v5-text-small 80.3% (p=0.003, N=61), Snowflake 77%, Jina v3 77% (p=0.008), BGE-M3 75%, Nomic 66%
- On original 35-case subset: 77-80% OOS, permutation p=0.003 (Snowflake), 0.053 (BGE-M3), 0.032 (Nomic)
- Quality direction is ORTHOGONAL to all tested words (cosine < 0.10 with "good", "careful", etc.)
- 10 labeled pairs suffice for BGE-M3 (78% OOS)
- Length ruled out as confound; bootstrap stable (cosine 0.77-0.83)

**Quality is multi-dimensional:**
- Firmness and warmth quality are anti-correlated: cosine -0.35 (Snowflake) to -0.49 (Gemini)
- Training on firmness alone → 5-25% on warmth (worse than coin flip)
- The centroid is a weighted compromise between opposing quality dimensions

**Words fail because they're ambiguous:**
- All 30 senses of "good" collapse into the same warmth-dominated direction
- No synonym cluster escapes the warmth bias — all positive-valence words are in the same neighborhood
- "Careful" works partially because it's warmth-independent, not because it points at quality

**The five-term tree's 89-94% is inflated:**
- OR logic with 5 binary classifiers has a chance baseline of 1 - 0.5^5 = 97%
- The centroid's 77-86% is a more honest measure of what embeddings encode

**Per-category strengths and weaknesses (centroid on 61 OOS cases):**
- Anti-sycophancy: 93% all models — strongest signal
- Nuance/context: 77% all models — reads social subtext well
- Factual accuracy: 33-67% — can't fact-check, worse on smaller models
- Creative quality: 54-77% — model-dependent

**Meta-validity battery confirms method is sound:**
- Response-only BEATS full-format on 2/3 models (prompt adds noise)
- Label flip perfectly symmetric; LOO very stable (cosine 0.995+)
- Absolute scoring meaningless — pairwise comparison only
- Additional models (gte, e5, mxbai) show signal (66-72%) but fail permutation

**The direction is unnameable and style-tolerant:**
- No quality phrase exceeds cosine 0.30 (46 phrases, 22 words tested)
- Between-prompt variance > within-prompt variance (ratio 1.8-2.7x)
- Formal/technical style penalized; no single style dominates cross-model
