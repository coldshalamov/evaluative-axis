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

## 1. Intervention Test [DECISIVE - DO THIS FIRST]

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
   - Embedding-scored LLM critique (have an LLM write a natural-language evaluation, score the evaluation text with the embedding axis)
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

## 2. Blind Disagreement Adjudication [STRENGTHENS CENTRAL CLAIM]

**The question**: Does the 88.4% corrected agreement hold up under blind review?

**Protocol**:

1. Take the 109 gradeable disagreement cases (65 embedding-right + 44 HH-right).
2. Present each case to 2–3 reviewers (or a strong LLM judge) blind — they see the prompt and both responses but don't know which the embedding preferred.
3. Collect independent grades.
4. Compute inter-annotator agreement and the fraction that agree with the embedding vs. HH.

**Why it matters**: The current grading was done by one reviewer who knew which response the embedding preferred. Blind adjudication eliminates that bias and makes the 88.4% number publishable.

**Resources**: Could use Gemini Flash as a blind judge (cheap, fast) with a sample of human-verified cases to validate the LLM judge's reliability.

**Estimated time**: Half a day.

---

## 3. Gemini Embedding Full Run [MODEL SCALING]

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

## 4. Cross-Domain Validation [BROADENS THE CLAIM]

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

## 5. Adversarial / Negation Testing [CHARACTERIZES LIMITS]

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

## 6. Cumulative Context Process-Scoring Simulation [TRAINING MECHANISM]

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

## 7. DPO Training Experiment [INTERVENTION — LATER]

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

## 8. Pretraining Data Curation Pilot [EXTENDS APPLICATIONS]

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

```
Experiment 1 (Intervention test)     <- DO THIS FIRST. Practical claim.
    │
    ├── Experiment 2 (Blind adjudication) <- Can run in parallel
    │
    ├── Experiment 3 (Gemini full run)    <- Can run in parallel if quota available
    │
Experiment 4 (Cross-domain)          <- After intervention test confirms signal
    │
Experiment 5 (Adversarial testing)   <- Can run anytime
    │
Experiment 6 (Process scoring)       <- Tests dense-supervision mechanism
    │
Experiment 7 (DPO training)          <- Only after 1-3 and/or 6 are positive
    │
Experiment 8 (Pretraining curation)  <- Extension, after core argument is solid
```

The intervention test is the single most important experiment. If embedding-axis selection beats cheap baselines under blind review, the paper has a practical contribution. If it doesn't, no amount of correlation studies will make the argument.

---

*Last updated: June 23, 2026*
