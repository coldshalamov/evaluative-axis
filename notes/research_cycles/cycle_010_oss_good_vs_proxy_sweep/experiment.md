# Cycle 010 Experiment

Date: June 27, 2026

## Protocol

Battery:

- `notes/research_cycles/cycle_002_potential_shaping/controlled_evaluative_axis_battery_v3_50_cases.jsonl`

Runner:

- `scripts/run_good_vs_proxy_conflicts.py`

Sweep wrapper:

- `scripts/sweep_good_vs_proxy_conflicts.py`

Backend:

- `fastembed`

Execution environment:

- `C:\Users\93rob\.cache\codex-embedding-venv\Scripts\python.exe`

Models swept:

- `BAAI/bge-small-en-v1.5`
- `BAAI/bge-base-en-v1.5`
- `thenlper/gte-base`
- `snowflake/snowflake-arctic-embed-m`
- `jinaai/jina-embeddings-v2-small-en`
- `jinaai/jina-embeddings-v2-base-en`
- `nomic-ai/nomic-embed-text-v1.5-Q`
- `mixedbread-ai/mxbai-embed-large-v1`

Axes compared:

- raw `good/bad`
- sentence `This response is good/bad.`
- nearby proxy pairs:
  - `true/false`
  - `truthful/untruthful`
  - `honest/dishonest`
  - `useful/useless`
  - `helpful/unhelpful`
  - `accurate/inaccurate`
  - `correct/incorrect`
  - `safe/unsafe`

## Why This Is Cleaner

- same frozen 50-case battery across all local models
- exact word-count matching removes the length confound
- raw-answer scoring avoids decomposition leakage
- the sweep directly addresses the user's "surrounding area in embedding space"
  question rather than arguing from HH overlap
