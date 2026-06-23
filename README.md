# Evaluative Geometry as Reward Signal

**Claim**: The evaluative axis (good/bad) in embedding space encodes a general-purpose quality signal that can serve as a direct reward for LLM alignment — without training a reward model, without human labelers, and without LLM-as-judge inference.

**Status**: Exploratory research with strong preliminary evidence. The key finding so far is that embedding-axis scoring agrees with corrected human preference on 83–88% of gradeable cases, and in many disagreements with existing labels, the embedding anticipates where human preference norms later converged.

## The Idea in 30 Seconds

Osgood (1957) showed that evaluation (good/bad) is the primary axis of human semantic judgment. Kozlowski et al. (2025) confirmed this axis exists in LLM embedding geometry. We project text onto this axis using 12 anchor sentences and a dot product. The result is a quality score that costs a fraction of a cent, is perfectly deterministic, and captures helpfulness, honesty, safety, and correctness simultaneously — because those properties are geometrically entangled along the evaluative direction.

## Key Results

- **Controlled validation**: 70.5% accuracy on 61 statement pairs (Gemini Embedding 2); 86.7% excluding structurally hard sycophancy/honesty cases
- **HH-RLHF preference prediction**: 55.8% raw agreement with 2022 labels (BGE-small)
- **Full disagreement audit**: Of 231 embedding-vs-HH disagreements, 65 were embedding-right, 44 were HH-right, 122 were both-bad/trivial. Corrected gradeable agreement: **88.4%**
- **Label noise detection**: Embedding scoring caught fabricated persona claims, doxxing compliance, misinformation, and racist content that HH-RLHF rewarded

## Why This Matters

Current alignment approaches require either expensive human annotation (RLHF: ~$100/annotation-hour) or full LLM inference for every judgment (RLAIF). Embedding-axis scoring costs near-zero, runs at millions of evaluations per minute, and captures evaluative structure that appears to be deeper than any single dataset's labeling policy.

The "anticipating later norms" finding — where the embedding preferred responses that align with modern safety standards over responses that HH-RLHF labeled as preferred in 2022 — suggests the geometry isn't just approximating existing labels but recovering underlying evaluative structure from language itself.

## Repository Structure

```
paper/
  draft.md              # Paper draft (argument + evidence)
  references.md         # Bibliography

methodology/
  RESEARCH_LOOP_PROTOCOL.md   # Idea/Literature/Experiment/Autopsy/Forest/Decision loop
  RESEARCH_OPERATING_MODE.md  # Prevents premature metric-chasing
  MECHANISM_MAP.md            # Where embedding evaluation could fit in the pipeline
  RIGOR_GUARDRAILS.md   # Self-imposed experimental discipline rules
  experiment_roadmap.md # Remaining experiments needed
  templates/            # Fill-in templates for each research mode

experiments/            # All experimental results, by phase
  phase1/               # Axis validation (controlled statement pairs)
  phase1_gemini/        # Gemini rerun of Phase 1
  phase2/               # HH-RLHF preference prediction
  phase2_gemini/        # Gemini rerun (quota-blocked)
  phase2_open_source/   # BGE model comparisons
  phase3/               # DPO fine-tuning (not yet run)
  phase4/               # Original report + figures
  phase5_verification/  # Context binding test + HH verification
  phase6_gemini_200/    # Partial Gemini multi-sensor
  phase6_multi_sensor/  # Multi-sensor across HH, PKU, SHP

disagreement_audit/     # Full 231-case HH disagreement grading
  full_grading.md       # All cases with grades + reasoning
  manual_grading.md     # Initial 30-case audit
  all_disagreements.json

scripts/                # All experimental scripts

notes/                  # Analysis and logs
  RESEARCH_NOTES.md     # Detailed analytical notes
  research_log.md       # Chronological experiment log
  PHASE2_DIAGNOSIS.md   # Phase 2 failure analysis

infrastructure/         # Implementation artifacts (Codex prompts, setup)
```

## Theoretical Foundation

1. **Osgood (1957)**: Evaluation is the primary dimension of human semantic judgment, cross-culturally universal
2. **Grand et al. (2022)**: Semantic projection onto antonym-defined directions recovers human knowledge from embeddings
3. **Kozlowski et al. (2025)**: Osgood's dimensions exist in LLM embedding geometry; features are entangled
4. **Value Entanglement (2026)**: LLMs conflate moral, grammatical, and economic "good" — which for alignment is a feature, not a bug

## What's Next

The decisive experiment is an **intervention test**: generate multiple candi