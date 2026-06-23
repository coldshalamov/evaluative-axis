# Decision Mode: Cycle 001

Date: June 23, 2026
Cycle: `cycle_001_next`

## Decision

Switch to intervention test.

Do not spend the next serious cycle trying to squeeze more meaning out of raw
HH-RLHF agreement. Use HH only as one noisy sensor and as a source of hard
examples. The next decisive artifact is a no-training candidate-selection
benchmark with blind review and example autopsy.

## Evidence Used

- Full HH disagreement grading:
  - Raw agreement: 269/500 = 53.8%.
  - Gradeable disagreement split: embedding better 65, HH better 44.
  - Excluded low-signal/both-bad cases: 122.
  - Corrected gradeable agreement: 334/378 = 88.4%.
- Controlled Gemini Phase 1:
  - 70.5% on controlled statement pairs, 86.7% excluding known hard
    sycophancy/honesty categories.
- Context polarity test:
  - Generic axes failed on polarity/context.
  - Contextual harm-reduction improved substantially.
  - Oracle decomposition showed the interface matters.
- Literature:
  - Semantic projection supports axis-based scoring.
  - Process-supervision work shows intermediate feedback matters.
  - DPO shows preference pairs are a natural training interface if pairs can be
    generated or cleaned.
  - Constitutional AI shows critique/decomposition text is a useful object of
    evaluation.

## Remaining Uncertainty

- Whether blind reviewers prefer embedding-selected outputs over random,
  length, sentiment, and LLM judge selections.
- Whether Gemini-class embeddings materially outperform BGE-small in the
  intervention setting.
- Whether decomposition scoring improves because it exposes real evaluative
  structure or because it leaks labels.
- Whether the signal is most useful for selection, rejection, weighting,
  routing, or dense process reward.
- Whether long-context embedding models are required for the strongest version.

## What Would Reverse This Decision?

- A blinded candidate-selection pilot where embedding selection loses to random
  and length, and autopsy shows the losses are true embedding failures rather
  than bad prompts, bad candidate sets, or missing context.
- A decomposition-scoring pilot where decompositions fail to improve selection
  and mainly amplify bias or label leakage.
- Evidence that the signal is not stable across stronger embedding models or
  across prompt categories after proper axis routing.

## Next Action

Concrete next action:

Run the cycle-001 scaffold in lexical smoke mode to verify plumbing, then run a
Gemini-backed pilot on 50 prompts when quota is available.

Owner:

Codex for scripts, protocol, generated packets, and summaries. Human or
external reviewer for blind adjudication if publication-grade claims are made.

Artifact to produce:

- `scripts/run_cycle001_intervention.py`
- `notes/research_cycles/cycle_001_next/seed_candidates.json`
- `notes/research_cycles/cycle_001_next/smoke_results/summary.md`
- Later: `notes/research_cycles/cycle_001_next/pilot_results/summary.md`

Completion evidence:

- Script compiles.
- Smoke run completes without API or GPU.
- Score tables, selections, and blind packet are written.
- Research log records what was done, numerical outputs, interpretation, and
  next decision.
