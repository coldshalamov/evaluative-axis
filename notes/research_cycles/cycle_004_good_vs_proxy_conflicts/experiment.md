# Cycle 004 Experiment

Date: June 27, 2026

## Protocol

Primary protocol:

- `methodology/GOOD_VS_PROXY_CONFLICTS_PROTOCOL.md`

Runner:

- `scripts/run_good_vs_proxy_conflicts.py`

Benchmark:

- `notes/research_cycles/cycle_002_potential_shaping/controlled_evaluative_axis_battery_v3_50_cases.jsonl`

Comparison axes:

- raw `good/bad`
- sentence `This response is good.` / `This response is bad.`
- raw `true/false`
- raw `truthful/untruthful`
- raw `honest/dishonest`
- raw `useful/useless`
- raw `helpful/unhelpful`
- raw `accurate/inaccurate`
- raw `correct/incorrect`
- raw `safe/unsafe`

Backends run:

- `gemini-embedding-2`
- `BAAI/bge-base-en-v1.5`

Control comparison:

- `scripts/run_evaluative_axis_battery.py` on the same 50-case battery with the
  richer cycle-001 evaluative axes

## Why This Is Cleaner

- no HH overlap as headline metric
- no decomposition text in the primary word-level comparison
- same cases reused across all axis words, so extra axes are almost free once
  candidate embeddings exist
- same cases reused for the richer-axis control, so the comparison is not across
  different benchmarks
