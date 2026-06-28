# Cycle 011 Experiment

Date: June 27, 2026

## Protocol

Frozen battery:

- `notes/research_cycles/cycle_002_potential_shaping/controlled_evaluative_axis_battery_v3_50_cases.jsonl`

Run 1: direct routed evaluative battery

- runner: `scripts/run_evaluative_axis_battery.py`
- backend: `gemini`
- interfaces: `direct`
- output:
  `notes/research_cycles/cycle_011_battery_v3_gemini_reproduction/battery_direct_only`

Run 2: raw `good/bad` versus nearby proxy words

- runner: `scripts/run_good_vs_proxy_conflicts.py`
- backend: `gemini`
- output:
  `notes/research_cycles/cycle_011_battery_v3_gemini_reproduction/good_vs_proxy`

Gemini settings:

- model probe: `gemini-embedding-2`
- batch size: `16`
- max workers: `1`

Operational note:

- both runs hit HTTP 429 quota retries mid-run
- both completed successfully without changing protocol, file, or model

## Why This Is Cleaner

- exact word-count matching is preserved: mean gap `0.00`, max gap `0`
- raw-answer scoring avoids decomposition leakage
- the reproduction checks the current live file, not just an older saved folder
- the paired runs show both the positive targeted-axis result and the negative
  broad-word result on the same frozen battery
