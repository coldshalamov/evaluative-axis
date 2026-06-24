# Decision Mode: Cycle 002

Date: June 23, 2026
Cycle: `cycle_002_potential_shaping`

## Decision

Promote cumulative full-context potential shaping to the strongest research
formulation.

Do not lead with "83-88% corrected HH accuracy" as if it were validated
accuracy. Use the HH audit as qualitative and triage evidence until blind
adjudication samples both agreements and disagreements.

## Evidence Used

- New prior work verifies that unified evaluative/truth-related representation
  is plausible.
- Embedding reward has precedent in PGSRM, but not with a sparse general
  good/bad external potential.
- Trajectory geometry has evidence as reward signal through TRACE.
- Cycle 001 exposed length confounding in the first intervention harness.
- HH table-backed disagreement audit gives 63/108 = 58.3% embedding wins among
  gradeable disagreements; exact two-sided binomial p = 0.101, suggestive but
  not decisive.

## Remaining Uncertainty

- Whether controlled length-matched pairs are separable by current embeddings.
- Whether potential deltas localize injected reasoning errors.
- Whether stronger embeddings materially improve this.
- Whether broad valence creates motivated hallucination or factual subordination.

## Next Action

Run the controlled evaluative-axis battery locally with lexical and FastEmbed
BGE-small backends, then inspect failures before designing trajectory tests.

Owner: Codex.

Completion evidence:

- Battery JSONL exists.
- Runner exists.
- Lexical and FastEmbed summaries exist.
- Research log records numbers and interpretation.
