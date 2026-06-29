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
- `notes/research_cycles/cycle_002_potential_shaping/controlled_evaluative_axis_battery_v3_50_cases.jsonl` — 50-case battery
- `notes/research_cycles/battery_rebalancing/warmth_cases.jsonl` — 20 warmth cases
- `notes/research_cycles/battery_rebalancing/rebalance_results.json` — Rebalancing results

## Current state of knowledge

- No single word survives both firmness and warmth on all three models
- "Good/Bad" is bipolar: 85% on warmth, 16% on firmness (BGE-M3)
- Subtracting warmth from "good" leaves noise at chance — quality IS in the children
- 5-term tree (careful, honest, helpful, thorough, restrained) with OR logic: 89-94% OOS across all 3 models
- This is the decomposed "good" signal: each term catches different quality dimensions
- Gemini Embedding 2 dramatically outperforms local models (cause unknown)
- The "good" tree decomposition is confirmed: neighborhood analysis shows 40% warmth+emotion vs 12% competence+restraint
