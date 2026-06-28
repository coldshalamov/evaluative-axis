# Collaborator Brief

This is an active research project exploring whether the evaluative axis in
embedding space can serve as a general-purpose quality signal for language
model alignment.

## Quick Links

- **Paper draft**: `paper/draft.md`
- **Research context** (environment, pitfalls, template): `methodology/RESEARCH_CONTEXT.md`
- **Open research questions**: `methodology/RESEARCH_DIRECTIONS.md`
- **Experiment specs** (ready to implement): `methodology/EXPERIMENT_SPECS.md`
- **Rigor guardrails**: `methodology/RIGOR_GUARDRAILS.md`

## The Core Idea

Define a direction in embedding space (e.g., embed("Careful") - embed("Reckless"),
normalized). Project any response onto this direction. The dot product is the
quality score. No classifier training, no labeled data, no LLM inference.

## Current State (June 2026)

### What works
- Gemini Embedding 2 scores 86-98% on targeted axes (50-case battery)
- Objective reranking: code 83%, math 100%, tool 88% (Gemini, small suites)
- "Careful/Reckless" is the most cross-model stable single word on firmness
  cases (80% anti-sycophancy on all 3 local models)
- Different single words capture genuinely different geometric dimensions
  (cosine 0.01-0.20 between axis vectors)

### What fails
- Raw "good/bad" fails on firmness cases (16-48%) but succeeds on warmth (60-85%)
- No single word survives BOTH firmness and warmth splits on all 3 local models
- Local models (33M-600M params) never match Gemini; scale doesn't help
- Compositing multiple axes into one averaged vector always degrades
- Multi-sentence ML-jargon axes are less cross-model stable than single words

### Key insight (recent)
The original 50-case battery is 64% firmness-biased. All previous single-word
results are confounded. "Hard/Soft" was a battery artifact, not a finding.
Battery rebalancing (adding 20 warmth cases) showed that no tested axis is
truly universal. The research program is now about finding the right SET of
terms that together cover all dimensions of quality.

### Leading theory
"Good" sits atop a semantic tree with thousands of child meanings. Each child
(careful, honest, kind...) is more specific and gives a cleaner signal in its
domain. But no single child covers everything. The optimal evaluative system
is a SET of child terms at the right tree depth, scored independently and
combined. Finding this set is the main open question.

## What's Needed

Experiments, not theory. The open questions are all in
`methodology/RESEARCH_DIRECTIONS.md`. Priority experiments are spec'd in
`methodology/EXPERIMENT_SPECS.md`. Pick one and run it.

---

*Research by Robin Gattis. June 2026.*
