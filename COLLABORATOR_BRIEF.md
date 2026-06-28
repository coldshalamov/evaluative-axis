# Collaborator Brief

This is an active research project exploring whether the evaluative axis in embedding space — the geometric direction corresponding to "good" versus "bad" — can serve as a general-purpose quality signal for language model alignment, data curation, and training.

## Quick Links

- **Paper draft**: [paper/draft.md](paper/draft.md) — the full argument, evidence, and limitations
- **Current serious cycle**: [notes/research_cycles/cycle_001_next](notes/research_cycles/cycle_001_next) — intervention-test protocol, autopsy, forest, decision, seed fixture, and smoke outputs
- **Quota-free results**: [notes/research_cycles/cycle_001_next/quota_free_results.md](notes/research_cycles/cycle_001_next/quota_free_results.md) — 50-prompt pilot, cheap baselines, local BGE-small, and length-bias diagnosis
- **Potential-shaping cycle**: [notes/research_cycles/cycle_002_potential_shaping](notes/research_cycles/cycle_002_potential_shaping) — controlled minimal pairs, exact length-balanced v2, and cumulative-context potential framing
- **Research loop**: [methodology/RESEARCH_LOOP_PROTOCOL.md](methodology/RESEARCH_LOOP_PROTOCOL.md) — Idea → Literature → Experiment → Autopsy → Forest → Decision
- **Mechanism map**: [methodology/MECHANISM_MAP.md](methodology/MECHANISM_MAP.md) — pipeline uses beyond HH agreement
- **Experiment roadmap**: [methodology/experiment_roadmap.md](methodology/experiment_roadmap.md) — what's needed next
- **Decisive evidence plan**: [methodology/DECISIVE_EVIDENCE_PLAN.md](methodology/DECISIVE_EVIDENCE_PLAN.md) — what would actually count as proof vs suggestive evidence
- **Rigor guardrails**: [methodology/RIGOR_GUARDRAILS.md](methodology/RIGOR_GUARDRAILS.md) — self-imposed experimental discipline
- **Research operating mode**: [methodology/RESEARCH_OPERATING_MODE.md](methodology/RESEARCH_OPERATING_MODE.md) — prevents premature metric-chasing and forces broad interpretation
- **Central result**: [disagreement_audit/full_grading.md](disagreement_audit/full_grading.md) — all 231 HH disagreement cases graded
- **Concept notes**: [RESEARCH_CONCEPT_NOTES.md](RESEARCH_CONCEPT_NOTES.md) — human framing, dense supervision, persona honesty, and scalar-plus-basis ideas

## The Claim

Evaluative judgment is geometrically encoded in embedding space. A single direction — defined by 12 anchor sentences and a dot product — captures helpfulness, honesty, safety, and quality simultaneously because those properties are geometrically entangled along the evaluative axis. The signal costs near-zero, is deterministic, and in several cases anticipates where human preference norms later converged.

## Current Evidence

- 88.1% table-backed corrected agreement on gradeable HH-RLHF cases, with the prior 88.4% summary now flagged for count reconciliation
- Embedding catches label noise in HH-RLHF: fabricated personas, doxxing compliance, misinformation
- Known limitations: 0% on sycophancy, 40% on honesty-hedging (documented honestly)
- Quota-free 50-prompt intervention harness now exists; local BGE-small did not beat length on the proxy key, revealing a length-bias design problem to fix before making intervention claims
- Exact-length controlled v2 battery now exists. Local BGE-small broad good/bad scoring failed on it. After fixing a category-axis mapping bug, `jinaai/jina-embeddings-v2-small-en` reached 11/12 = 91.7% on answer-plus-decomposition category-axis scoring, but that interface used hand-authored "Good parts"/"Bad parts" decompositions. It is now classified as oracle-label leakage: a plumbing sanity check, not evidence that embeddings independently inferred answer quality.
- A naive cumulative-trajectory probe failed for both BGE-small and Jina-small, so the process-supervision hypothesis should be tested with better natural/injected-error trajectories rather than generic strategy templates.

## What's Missing

The blinded intervention test — does embedding-axis selection actually produce
better outputs under blind review? Cycle 001 now contains the frozen protocol,
a runnable scaffold, a 50-prompt pilot packet, cheap baseline results, and a
local BGE-small run. The current pilot is length-biased, so the next version
must length-balance constructed candidates before claiming lift. The Gemini
smoke run is currently blocked by API quota, not by local tooling.

The deeper training hypothesis is also not yet adequately tested: score the
full context after each reasoning step, then use score deltas as cheap dense
supervision. A first generic trajectory probe failed, which is useful discipline:
the next process test needs natural trajectories, injected errors, repairs, and
controlled stage texts. Cycle 002 no longer treats hand-authored
answer-plus-decomposition scoring as evidence. The valid next protocol is
blind, label-free decomposition or raw answer scoring on held-out cases, with
cumulative potential-shaping as a separate mechanism to test carefully.

## Help Needed

Experimental design review, statistical review, blind human adjudication, access to stronger embedding APIs, and eventually DPO/RL implementation if the intervention test is positive.

---

*Research by Robin Gattis. June 2026.*
