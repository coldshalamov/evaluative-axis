# AGENTS.md

## Scope

This repo is a research workspace for testing whether evaluative embedding
geometry can act as a cheap answer-selection or training signal.

Keep claims proportional to evidence. This repo is specifically trying to avoid
reward-hacky benchmarks, label leakage, and false confidence from weak proxies.

## Read First

- `COLLABORATOR_BRIEF.md`
- `methodology/DECISIVE_EVIDENCE_PLAN.md`
- `methodology/RIGOR_GUARDRAILS.md`
- `notes/research_log.md`

## Core Research Rules

- Treat HH, PKU, SHP, and LLM judges as sensors, not ground truth.
- Prefer objective end metrics or blind adjudication over proxy overlap.
- Do not present length-biased, leakage-prone, or judge-contaminated wins as
  decisive evidence.
- Negative results are useful here. Record them honestly.
- After each substantive research cycle or phase, append the outcome,
  interpretation, fallback conditions, and next steps to `notes/research_log.md`.

## Colab And Chrome Workflow

- Assume the real operating constraints are: one laptop, free-tier Google/Colab
  access, and intermittent quota issues.
- If `mcp__colab_mcp.open_colab_browser_connection` returns `false`, do **not**
  immediately conclude that the Colab connector is unavailable.
- First use the Chrome plugin with the user's existing Chrome profile to handle
  the browser approval prompt for Colab MCP, then retry the Colab connection.
- Only fall back to local execution after the Chrome approval path has been
  tried or is unavailable.
- Do not report "Colab didn't work" unless the approval step was attempted.
- If approval succeeds but the MCP transport still closes, use the live Colab
  browser session directly as a fallback surface before abandoning Colab
  entirely.
- Record whether a run used:
  - direct Colab connection;
  - Chrome-assisted Colab approval;
  - Chrome-approved direct browser Colab execution;
  - or local fallback.
- Treat connector approval failures and HTTP 429 quota blocks as operational
  issues, not as evidence about the research thesis.

## Repo Map

- `methodology/`: research protocols, rigor rules, evidence framing
- `notes/research_cycles/`: cycle-by-cycle idea, experiment, results, decision
- `notes/research_log.md`: chronological experiment log
- `scripts/`: runnable harnesses and analysis tools
- `disagreement_audit/`: HH disagreement grading and related artifacts
- `phase*/`: earlier staged experiments and outputs
- `paper/`: draft external writeup

## Good Default Next Steps

- When a proxy result is ambiguous, move toward an objective reranking task.
- When a positive result depends on decomposition text, check for leakage before
  trusting it.
- When a local or quota-limited run blocks the ideal test, document the
  fallback explicitly rather than silently changing standards.
