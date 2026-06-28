# Cycle 009 Experiment

Date: June 27, 2026

## Protocol

Battery:

- `notes/research_cycles/cycle_002_potential_shaping/controlled_evaluative_axis_battery_v3_50_cases.jsonl`

Runner:

- `scripts/sweep_fastembed_battery.py`

Interfaces:

- `direct`

Output root:

- `notes/research_cycles/cycle_009_oss_direct_battery_v3_sweep/`

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

Baselines carried inside the battery:

- `length`
- `sentiment`
- `refusal`

Operational note:

- the first attempt under the thread's default `python.exe` failed because that
  interpreter did not have `fastembed`
- the sweep was then rerun under the cached embedding venv and completed

## Why This Is Cleaner

- same frozen 50-case battery for every local model
- direct-only scoring avoids decomposition leakage from hand-authored notes
- cheap baselines remain visible in the same table
