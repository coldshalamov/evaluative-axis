# Experiment Roadmap

What remains to make the argument publishable, in priority order.

Before running any experiment here, use
`methodology/RESEARCH_OPERATING_MODE.md`: inspect examples, treat disagreement
as data, and ask what training mechanism the result suggests. Do not stop at a
single dataset-correlation number.

For each serious cycle, create a folder under `notes/research_cycles/` and fill
the mode templates from `methodology/templates/`:

- `idea.md`
- `literature.md`
- `experiment.md`
- `autopsy.md`
- `forest.md`
- `decision.md`

The experiment is not complete until `autopsy.md` and `forest.md` exist.

---

## 1. Controlled Minimal-Pair Battery [DO BEFORE MORE INTERVENTION CLAIMS]

**The question**: Can the evaluator distinguish good/right from bad/wrong when
length, tone, and verbosity are controlled?

**Why it matters**: Cycle 001 showed that candidate selection can be dominated
by length. A benchmark that does not control length cannot tell whether the
embedding sees quality or verbosity.

**Current status (June 23, 2026)**: Cycle 002 added a 23-case controlled
battery and a 12-case exact word-count-matched v2 battery. On v2, length and
sentiment fell to 50%, but local BGE-small broad evaluative scoring failed
badly: direct combined 8.3%, broad evaluative 0.0%, while direct
anti-sycophancy reached 66.7%. This is a negative diagnostic for the current
BGE-small/direct-axis setup and supports testing stronger embeddings plus a
scalar-plus-basis design.

**Next step**: Expand v2 to 50 length-balanced cases, then rerun FastEmbed and
Gemini when quota exists.

---

## 2. Emergent Decomposition Reward Simulation [STRONGEST ORIGINAL MECHANISM]

**Protocol file**:
`methodology/EMERGENT_DECOMPOSITION_REWARD_PROTOCOL.md`

**The question**: Does optimizing against an evaluative embedding reward make
decomposition emerge as a reasoning strategy?

**Why it matters**: The strongest hypothesis is not that an experimenter can
write `Good parts` / `Bad parts` and an embedding model can read those words.
The useful training claim is that a model seeking higher evaluative reward may
learn to separate a situation into good-making and bad-making factors, preserve
the good, reduce the bad, and handle tradeoffs. This should show up most clearly
in scratchpad-like traces or plans, not necessarily in the final answer alone.

**Protocol**:

1. Build 50-100 tasks where good performance requires tradeoff recognition,
   false-premise correction, risk containment, or repair after noticing an
   error.
2. Generate multiple scratchpad-style traces per task under neutral
   instructions. Do not ask for `good` / `bad` decomposition.
3. Score cumulative contexts with embedding potential.
4. Compare high-scoring and low-scoring traces under blind review for
   spontaneous decomposition, tradeoff handling, task quality, and reward
   hacking.
5. Run an evolutionary feedback loop: keep top-scoring traces, ask for variants
   with only scalar/rank feedback, and test whether traces become more
   decompositional without explicit instruction.
6. Include context-sensitivity cases such as `I should avoid introducing this
   bug`, `I should not lie`, and `The user's premise is wrong, so I should
   correct it`.

**Success criteria**: High reward should correlate with better blind-rated
traces, more spontaneous decomposition, and better contextual-negation handling
without merely increasing positive-tone words, refusals, or avoidance of
negative vocabulary.

---

## 3. Cumulative Context Potential-Shaping Simulation

**The question**: Can evaluative embedding scores provide useful dense
supervision when used as full-context potential deltas?

**Why it matters**: Raw final-answer or prefix scoring accumulates local
goodness and rewards verbosity. Potential deltas test whether a new reasoning
step improves or degrades the entire current trajectory.

**Protocol**:

1. Build short trajectories where one step introduces a known error and a later
   step may repair it.
2. Score `Phi_t = axis dot embed(full context through step t)`.
3. Compute `Delta_t = Phi_t - Phi_(t-1)`.
4. Test whether deltas localize the injected error and recognize repair.
5. Compare final-only, isolated-step, cumulative-context, and summary-context
   scoring.

**Success criteria**: Error steps should create negative deltas and repair
steps positive deltas more reliably than isolated-step or final-only scoring.

---

## 4. No-Leakage Decomposition Intervention Test [DECISIVE PRACTICAL CLAIM]

**Protocol file**:
`methodology/NO_LEAKAGE_DECOMPOSITION_PROTOCOL.md`

**The question**: Does embedding-axis scoring actually improve output selection, or does it just correlate with noisy labels?

**Why it matters**: Everything so far is correlation — the embedding agrees with human preference at some rate. The field has enough correlation studies. What's missing is a demonstration that the signal *does something useful*.

**Protocol**:

1. Sample 100–200 prompts from mixed sources (HH-RLHF, Alpaca, WildChat, or similar).
2. Generate 4–8 candidate responses per prompt using a capable model (Gemini Flash, GPT-4o-mini, or similar).
3. Score each candidate with:
   - Random selection (baseline)
   - Length (prefer longest)
   - Sentiment (prefer most positive)
   - Direct embedding-axis score
   - Embedding-scored blind feature report. The LLM describes strengths,
     weaknesses, risks, factual claims, uncertainty, and practical usefulness
     without answer labels and without choosing a winner in the same pass.
   - Standard LLM-as-judge (for comparison, not as ground truth)
4. Select the winner under each method.
5. Blind-judge the winners pairwise (human or strong LLM judge, order-randomized).
6. Report win rates and cost per evaluation.

**Success criteria**: Embedding-based selection beats random, length, and sentiment. If it also approaches LLM-as-judge quality at a fraction of the cost, that's the paper's practical claim.

**Resources needed**: An LLM API for candidate generation and judging (Gemini Flash free tier should work). Embedding API or local model for scoring. No GPU needed.

**Estimated time**: 1–2 days.

**Current status (June 23, 2026)**: Cycle 001 has the frozen protocol,
autopsy/forest/decision memos, seed fixture, and runnable scaffold:
`scripts/run_cycle001_intervention.py`. Lexical smoke mode ran successfully on
5 prompts / 15 candidates. Gemini smoke mode was attempted and blocked by API
quota during the embedding probe.

---

## 5. Blind Disagreement Adjudication [STRENGTHENS CENTRAL CLAIM]

**The question**: Does the HH disagreement audit hold up under blind review?

**Protocol**:

1. Take the 108 table-backed gradeable disagreement cases (63 embedding-right + 45 HH-right).
2. Present each case to 2–3 reviewers (or a strong LLM judge) blind — they see the prompt and both responses but don't know which the embedding preferred.
3. Collect independent grades.
4. Compute inter-annotator agreement and the fraction that agree with the embedding vs. HH.

**Why it matters**: The current grading was done by one reviewer who knew which response the embedding preferred. Blind adjudication eliminates that bias. Also sample agreement cases, because assuming all 269 agreements are correct is not a valid corrected-accuracy estimate.

**Resources**: Could use Gemini Flash as a blind judge (cheap, fast) with a sample of human-verified cases to validate the LLM judge's reliability.

**Estimated time**: Half a day.

---

## 6. Gemini Embedding Full Run [MODEL SCALING]

**The question**: How much does embedding model quality improve the signal?

**What we know**: Gemini Embedding 2 scored 70.5% on controlled pairs vs ~56% for all-mpnet and BGE-small. But the full 500+ pair HH-RLHF run was never completed due to API quota limits.

**Protocol**:

1. Check if Gemini API quota has reset.
2. If yes: run the full 500-pair HH-RLHF preference prediction with Gemini Embedding 2.
3. Run the full disagreement audit on Gemini's disagreements.
4. Compare corrected agreement: BGE-small vs Gemini.

**Why it matters**: If Gemini pushes corrected agreement from 88% to 93%+, the argument that model quality directly improves signal quality is strong. If it doesn't improve much, the signal may be more about axis construction than model capacity.

**Resources**: Gemini API with sufficient quota. No GPU needed.

**Estimated time**: A few hours (mostly API calls).

---

## 7. Cross-Domain Validation [BROADENS THE CLAIM]

**The question**: Does the evaluative axis work beyond safety/helpfulness?

**Domains to test**:

- **Code quality**: Use HumanEval or MBPP outputs. Score with embedding axis. Compare against pass@1 (verifiable ground truth).
- **Writing quality**: Use essay scoring datasets. Score with embedding axis. Compare against human quality ratings.
- **Summarization**: Use SummEval or similar. Score summaries with embedding axis. Compare against human coherence/relevance ratings.
- **Reasoning**: Use GSM8K or similar with multiple solution attempts. Score with embedding axis. Compare against correctness.

**Why it matters**: If the axis works across domains, the "general-purpose quality signal" claim is supported. If it only works for safety, it's a narrower contribution.

**Resources**: Public datasets. Embedding model. No GPU needed.

**Estimated time**: 1–2 days.

---

## 8. Adversarial / Negation Testing [CHARACTERIZES LIMITS]

**The question**: How does the axis handle adversarial cases?

**Tests**:

1. **Negation pairs**: Construct pairs where a single negation flips quality ("This is helpful" vs "This is not helpful"). Measure error rate.
2. **Confident lies vs hedged truths**: "The answer is definitely Paris" (wrong) vs "I'm not sure, but I think it might be Lyon" (right). Does the axis reward confidence over correctness?
3. **Helpful harmful content**: Text that is well-written, clear, and substantive but provides dangerous instructions. Does the evaluative axis catch the harmful content or reward the quality of expression?
4. **Sycophancy at scale**: Generate a larger set of sycophantic vs genuine pairs. Confirm the 0% finding. Test whether anti-sycophancy anchors help.

**Why it matters**: Honest characterization of failure modes makes the paper more credible, not less. Reviewers will test these cases themselves if we don't.

**Resources**: Manual construction of test cases. Embedding model. No GPU needed.

**Estimated time**: Half a day.

---

## 9. Cumulative Context Process-Scoring Simulation [TRAINING MECHANISM]

**The question**: Can embedding-axis scores provide dense supervision over a
reasoning or response trajectory, rather than only a final-answer score?

**Why it matters**: Outcome-only reward compresses all supervision into one
number at the end. The user's stronger hypothesis is that reasoning itself is
evaluative decomposition: a model improves when its intermediate text separates
good-making and bad-making factors. If embeddings can score the cumulative
context after each step, they may provide cheap process reward without training
a separate process reward model.

**Protocol**:

1. Collect or generate reasoning traces with known good and bad final outcomes.
2. Split each trace into steps, paragraphs, or tagged reasoning segments.
3. For each step `t`, embed the full context so far: prompt, previous turns,
   prior reasoning, and partial answer.
4. Score the full context against the broad evaluative axis and selected
   diagnostic axes.
5. Compute deltas: score at `t` minus score at `t-1`.
6. Inspect whether score drops identify bad turns in reasoning and whether good
   trajectories trend upward more reliably than bad trajectories.

**Variants**:

- raw reasoning stream vs. final answer only;
- cumulative full context vs. isolated step;
- broad good/bad axis vs. scalar-plus-basis;
- generated critique/decomposition text vs. hidden-style scratchpad text.

**Success criteria**: Cumulative-context deltas should locate meaningful
reasoning improvements/degradations better than isolated-step scoring and
better than final-answer-only scoring. A strong result would show that the axis
supplies useful credit assignment before any model training.

**Resources needed**: Saved reasoning traces and an embedding model/API. No GPU
required.

**Estimated time**: 1 day.

---

## 10. DPO Training Experiment [INTERVENTION - LATER]

**The question**: Does embedding-axis reward actually improve a model through DPO training?

**Protocol**:

1. Generate candidate responses from a small model (Gemma 2B or similar).
2. Score with embedding axis. Construct preference pairs from the scores.
3. Train with rDPO (label_smoothing calibrated to estimated noise rate).
4. Evaluate: LLM judge win rate, sycophancy check, response length distribution.

**Why it matters**: This is the end-to-end test of the original thesis. But it requires GPU time and is less informative than the intervention test (Experiment 1) if we haven't established the signal quality first.

**Prerequisite**: Experiments 1–3 should show positive results before investing GPU time.

**Resources**: Colab T4 GPU (~2–4 hours). Model download. Embedding API.

**Estimated time**: 1–2 days.

---

## 11. Pretraining Data Curation Pilot [EXTENDS APPLICATIONS]

**The question**: Can the embedding axis serve as a quality signal for pretraining data?

**Protocol**:

1. Sample 10K–100K documents from a pretraining corpus (C4, The Pile, or similar).
2. Score each with the embedding axis.
3. Examine the top and bottom 1% manually. Do high-scoring documents look "better"?
4. Correlate with existing quality metrics (perplexity, document length, domain).
5. If promising: train two small models — one on the full sample, one on the top 50% by embedding score — and compare on downstream benchmarks.

**Why it matters**: If this works, the practical value extends far beyond alignment. Pretraining data quality is widely considered the single biggest lever on model quality, and nobody has a great automated way to measure "quality" at trillion-token scale.

**Resources**: Dataset access. Embedding model. Small model training for the comparison (optional, needs GPU).

**Estimated time**: 1–3 days depending on scale.

---

## Order of Operations

```text
Experiment 1 (Controlled battery)    <- First gate: remove confounds.
  |
Experiment 2 (Emergent decomposition) <- Strongest original mechanism.
  |
Experiment 3 (Potential shaping)      <- Dense reward mechanics.
  |
Experiment 4 (Intervention test)      <- Practical claim after controls.
  |
  +-- Experiment 5 (Blind adjudication) <- Can run in parallel.
  |
  +-- Experiment 6 (Gemini full run)    <- Run when quota is available.
  |
Experiment 7 (Cross-domain)          <- After controlled/intervention signal.
  |
Experiment 8 (Adversarial testing)   <- Expand continuously.
  |
Experiment 9 (Process scoring)       <- Broader dense-supervision mechanism.
  |
Experiment 10 (DPO training)         <- Only after 1-4 and/or 9 are positive.
  |
Experiment 11 (Pretraining curation) <- Extension, after core argument is solid.
```

The controlled battery is now the gate. Cycle 001 showed that intervention
tests can be length-confounded; Cycle 002 showed that local BGE-small broad
good/bad scoring fails on a tiny exact-length battery. The next credible claim
requires passing controlled, length-balanced tests before spending Gemini quota
or GPU time.

---

*Last updated: June 23, 2026*
