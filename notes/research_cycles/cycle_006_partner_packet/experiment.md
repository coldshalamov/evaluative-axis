# Cycle 006 Experiment

Date: June 27, 2026

## Protocol

Build a reproducible partner packet directly from saved `research_system_v1`
artifacts.

Inputs:

- `notes/research_system_v1/report/report.json`
- `notes/research_system_v1/good_vs_proxy_conflicts_gemini_v1/summary.json`
- `notes/research_system_v1/good_vs_proxy_conflicts_bge_v1/summary.json`
- `notes/research_system_v1/battery_v3_gemini_direct_v1/summary.json`
- `notes/research_system_v1/process_potential_error_repair_v1/summary.json`
- `notes/research_system_v1/process_potential_error_repair_bge_v1/summary.json`

Output generator:

- `scripts/build_partner_packet.py`

Outputs:

- `paper/partner_packet_v1/brief.md`
- `paper/partner_packet_v1/packet_summary.json`
- `paper/partner_packet_v1/figures/*.svg`

## Why This Is Cleaner

- figures are generated from current saved results rather than copied by hand
- the packet keeps positive, negative, and pending claims together
- the process gate status is preserved exactly as frozen in the report
