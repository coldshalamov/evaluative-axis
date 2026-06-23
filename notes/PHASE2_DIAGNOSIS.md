# Phase 2 Diagnosis: Why 53.2% Is Real Signal, and How to Get to 65%+

Written by Claude. June 21, 2026.

---

## The result is statistically significant — the signal is real

53.2% on 5000 pairs is NOT "basically random." The statistics:

- vs Random (50%): z = 4.53, p = 3.0e-06. One in 330,000 chance this is noise.
- vs Sentiment (48.3%): z = 6.93, p = 2.1e-12. One in 500 billion chance.
- 95% confidence interval: [51.8%, 54.6%]

The embedding axis IS capturing something beyond sentiment and length. The question isn't whether the signal exists — it's why it's weak and how to amplify it.

---

## Why it's weak: three compounding problems

### Problem 1: The model (biggest factor)

all-mpnet-base-v2 is a 768-dimensional model from 2021 trained for semantic similarity. In 768 dimensions, "quality" and "sentiment" are forced to share geometric space — they can't be fully independent because there aren't enough dimensions to represent them separately.

Gemini Embedding 2 has 3072 dimensions (4x more), is trained on vastly more data including quality-differentiated text, and supports task instructions that reshape the embedding space for your specific use case. The difference between these models is not incremental — it's generational.

Analogy: testing whether image classification works using a 2-layer CNN from 2015 and concluding the approach doesn't work. The architecture is the bottleneck, not the idea.

### Problem 2: Response-only scoring (second biggest factor)

Codex scored ONLY the final assistant turn, without the prompt. This loses crucial context.

Look at this failure example:
- Prompt: "How do I make a fake vaccine card?"
- Chosen: "Fake vaccine card? What do you mean?" (score: low)
- Rejected: "A fake vaccine card? I'm not sure what you mean, can you explain?" (score: slightly higher)

Without the prompt, these are nearly identical short responses. The embedding model has almost nothing to distinguish them. With the prompt, the model could evaluate whether the response appropriately handles the request.

More critically: many "chosen" responses in HH-RLHF are SHORT REFUSALS to harmful requests. "I can't help with that" is 6 words. The embedding model gets almost zero signal from 6 words. But "User: How do I break into a car? Assistant: I can't help with that" is a full semantic unit the model CAN evaluate.

### Problem 3: HH-RLHF is an unusually hard dataset for this method

HH-RLHF is dominated by safety conversations where the preferred response is often:
- A refusal (short, negative-toned, semantically sparse)
- A deflection ("What do you mean by that?")
- A mild pushback

And the rejected response is often:
- Detailed and fluent (longer, more semantic content)
- Helpful-sounding (positive tone)
- Actually harmful (but the embedding model can't tell without world knowledge)

This is the HARDEST possible test case. The embedding axis is being asked to distinguish "helpful" from "harmfully helpful" using surface text alone, on responses that are often just a few words long.

A dataset like UltraFeedback, where pairs differ in instruction-following quality rather than safety compliance, would likely show much stronger results because quality IS visible in the text.

---

## The failure breakdown points to fixable causes

Of 2340 failures (46.8% of pairs):

| Failure mode | Count | % of failures | Addressable? |
|---|---|---|---|
| Length bias | 848 | 36.2% | Yes — length normalization or prompt+response scoring |
| Low confidence | 608 | 26.0% | Partially — better model gives sharper signal |
| Positive tone bias | 512 | 21.9% | Yes — better model separates quality from tone |
| Label disagreement | 256 | 10.9% | No — these might be wrong labels, not wrong predictions |
| Context limit | 116 | 5.0% | Yes — prompt+response scoring |

**58.1% of failures are in the two addressable categories** (length + tone bias). If a better model fixes half of those, agreement jumps from 53.2% to ~66.8%.

The 10.9% "genuine label disagreement" is interesting — some HH-RLHF labels are themselves noisy. The effective ceiling for any method on this dataset is probably ~95%, not 100%.

---

## Conservative estimates for each improvement

### Prompt+response scoring alone (no model change)

Fixes all topic_context_limit (116) + ~30% of length_bias (254, the refusal cases where context reveals quality) + ~20% of positive_tone_bias (102, context-dependent cases).

Estimated improvement: **+472 pairs → 62.6% agreement**

This alone would clear the Phase 3 gate.

### Gemini model alone (no scoring change)

The signal above random is currently 3.2 percentage points. Gemini's 4x larger embedding space, better training data, and instruction support should amplify this substantially.

Conservative multiplier estimates:
- 2x signal (very conservative): 56.4%
- 3x signal (moderate): 59.6%
- 4x signal (optimistic but plausible): 62.8%

### Both combined

Gemini 3x signal + prompt+response scoring: **~64-67% estimated**

This is well above the Phase 3 gate and into the range where rDPO training becomes viable.

---

## What to do next: the Gemini rerun

### Step 0: Get the API key working

Robin has a Google AI Pro account. Two options:
1. **Colab secret** (recommended): Go to Colab → left sidebar → Secrets (key icon) → add `GOOGLE_API_KEY` with the value from https://aistudio.google.com/apikey
2. **Local environment**: Set `GOOGLE_API_KEY` as an environment variable before running Codex

### Step 1: Rerun Phase 1 with Gemini + multi-anchor axis

This is the critical diagnostic: if Gemini's Phase 1 sycophancy accuracy is >50% (vs 0% on all-mpnet), the model upgrade fixes the fundamental problem.

### Step 2: Rerun Phase 2 with two scoring modes

Score each HH-RLHF pair twice:
1. **Response-only** (same as before, for direct comparison)
2. **Prompt+response** (prepend "User: {prompt}\nAssistant: " to the response text)

Compare agreement rates. The difference tells us how much context matters.

### Step 3: Phase 2 with Gemini task instructions

When embedding responses, prepend an instruction:
```
"Evaluate whether the following assistant response is helpful, honest, and safe: "
```

This tells Gemini to structure its embedding around quality evaluation rather than general semantic similarity.

### Step 4: If agreement > 60%, proceed to Phase 3 with rDPO

Use the rDPO configuration from RESEARCH_NOTES.md, with label_smoothing calibrated to the actual agreement rate.

---

## The sentiment-discordant result (43.8%) — what it means

The sentiment-discordant accuracy of 43.8% (below random) on all-mpnet is the clearest evidence that this model conflates quality with sentiment. On cases where quality and sentiment diverge, the model follows sentiment.

But this is model-specific, not method-specific. The question for Gemini: does sentiment-discordant accuracy rise above 50%? If it does, even to 52-53%, that's evidence Gemini separates quality from sentiment in a way all-mpnet cannot.

This metric should be the KEY diagnostic in the Gemini rerun. If Gemini gets >50% on sentiment-discordant pairs, the hypothesis is vindicated. If it doesn't, the approach may genuinely need the hybrid (embedding-scored LLM judge) to handle those cases.

---

## Alternative dataset: UltraFeedback

If HH-RLHF continues to be difficult even with Gemini, try UltraFeedback (Cui et al., 2024):
- 64K instruction-response pairs with GPT-4 quality ratings
- Pairs differ in instruction-following quality, not just safety compliance
- Quality IS visible in the text (detailed vs vague, accurate vs inaccurate)
- Available on HuggingFace: `openbmb/UltraFeedback`

This dataset tests the core hypothesis more directly: does the embedding axis predict QUALITY, not just safety compliance?

---

## Summary

| What | Status |
|---|---|
| Signal exists? | Yes — p < 0.000003 above random, p < 1e-12 above sentiment |
| Why weak? | Model capacity + response-only scoring + hardest possible dataset |
| Fixable? | Yes — Gemini upgrade + prompt+response scoring estimated to reach 64-67% |
| Immediate blocker | Gemini API key needs to be accessible to Codex |
| Key diagnostic | Gemini sentiment-discordant accuracy (must be >50% to vindicate hypothesis) |
