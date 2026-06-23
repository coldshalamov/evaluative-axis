# Collaborator Brief

This is an active research project exploring whether the evaluative axis in embedding space — the geometric direction corresponding to "good" versus "bad" — can serve as a general-purpose quality signal for language model alignment, data curation, and training.

## Quick Links

- **Paper draft**: [paper/draft.md](paper/draft.md) — the full argument, evidence, and limitations
- **Current serious cycle**: [notes/research_cycles/cycle_001_next](notes/research_cycles/cycle_001_next) — intervention-test protocol, autopsy, forest, decision, seed fixture, and smoke outputs
- **Research loop**: [methodology/RESEARCH_LOOP_PROTOCOL.md](methodology/RESEARCH_LOOP_PROTOCOL.md) — Idea → Literature → Experiment → Autopsy → Forest → Decision
- **Mechanism map**: [methodology/MECHANISM_MAP.md](methodology/MECHANISM_MAP.md) — pipeline uses beyond HH agreement
- **Experiment roadmap**: [methodology/experiment_roadmap.md](methodology/experiment_roadmap.md) — what's needed next
- **Rigor guardrails**: [methodology/RIGOR_GUARDRAILS.md](methodology/RIGOR_GUARDRAILS.md) — self-imposed experimental discipline
- **Research operating mode**: [methodology/RESEARCH_OPERATING_MODE.md](methodology/RESEARCH_OPERATING_MODE.md) — prevents premature metric-chasing and forces broad interpretation
- **Central result**: [disagreement_audit/full_grading.md](disagreement_audit/full_grading.md) — all 231 HH disagreement cases graded
- **Concept notes**: [RESEARCH_CONCEPT_NOTES.md](RESEARCH_CONCEPT_NOTES.md) — human framing, dense supervision, persona honesty, and scalar-plus-basis ideas

## The Claim

Evaluative judgment is geometrically encoded in embedding space. A single direction — defined by 12 anchor sentences and a dot product — captures helpfulness, honesty, safety, and quality simultaneously because those properties are geometrically entangled a