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
- **Controlled no-quota sweep**: On a 12-case exact word-count-matched battery, broad direct scoring failed. A later 11/12 = 91.7% Jina result used hand-authored "Good parts"/"Bad parts" decompositions and is now classified as oracle-label leakage: useful as a plumbing sanity check, not evidence that embeddings independently inferred answer quality.

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

The decisive experiment is an **intervention test**: generate multiple candidate responses, score them with the embedding axis, and blind-judge whether embedding-selected outputs beat random, length, sentiment, and LLM-judge baselines. This tests the practical claim directly.

Cycle 001 now contains the first serious version of that plan:

- [cycle_001_next/experiment.md](notes/research_cycles/cycle_001_next/experiment.md):
  frozen protocol for the no-training candidate-selection benchmark
- [cycle_001_next/autopsy.md](notes/research_cycles/cycle_001_next/autopsy.md):
  required example-reading taxonomy
- [cycle_001_next/forest.md](notes/research_cycles/cycle_001_next/forest.md):
  broad mechanism synthesis
- [cycle_001_next/decision.md](notes/research_cycles/cycle_001_next/decision.md):
  decision to switch from HH agreement to intervention testing
- [scripts/run_cycle001_intervention.py](scripts/run_cycle001_intervention.py):
  runnable scaffold with lexical smoke mode and Gemini embedding mode
- [cycle_001_next/smoke_results/summary.md](notes/research_cycles/cycle_001_next/smoke_results/summary.md):
  verified no-API smoke output
- [cycle_001_next/quota_free_results.md](notes/research_cycles/cycle_001_next/quota_free_results.md):
  50-prompt pilot, cheap baselines, local BGE-small run, and length-bias
  diagnosis
- [cycle_002_potential_shaping/results.md](notes/research_cycles/cycle_002_potential_shaping/results.md):
  controlled minimal-pair results, exact length-balanced v2, and potential
  shaping reframe

Gemini embedding mode was attempted on the seed fixture and blocked by API
quota. See
[cycle_001_next/gemini_smoke_results/quota_blocked.md](notes/research_cycles/cycle_001_next/gemini_smoke_results/quota_blocked.md).

The current quota-free pilot is not a positive intervention result. Local
BGE-small did not beat the length baseline on the proxy key. That is useful:
it shows the next pilot must length-balance candidates and rely on blind review
rather than proxy labels before making practical claims.

Cycle 002 goes one step further: on a 12-case exact word-count-matched battery,
length and sentiment drop to chance, and local BGE-small broad good/bad scoring
fails. After fixing a category-axis mapping bug, an 11/12 = 91.7% Jina result
appeared on answer-plus-decomposition category-axis scoring, but that interface
used hand-authored "Good parts"/"Bad parts" text. The result is therefore
classified as oracle-label leakage. The old 50-prompt proxy pilot did not
validate the method, and a naive cumulative trajectory probe also failed. The
next clean test is blind, label-free decomposition or raw answer scoring on a
larger held-out battery.

Before running it, follow `methodology/RESEARCH_LOOP_PROTOCOL.md`: Idea Mode,
Literature Mode, Experiment Mode, Autopsy Mode, Forest Mode, and Decision Mode.
The goal is to avoid getting trapped inside a single proxy metric again.

See `methodology/experiment_roadmap.md` for the full plan.

---

*Research by Robin Gattis. June 2026.*
