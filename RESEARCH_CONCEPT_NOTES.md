# Research Concept Notes

Date: June 23, 2026

These notes preserve the human framing and research ideas that emerged during
discussion. They are not all completed experiments. Items marked "needs
reproduction" should not be cited as verified workspace results until rerun.

## 1. Core Thesis In Plain Language

Good/bad may be a fundamental evaluative axis of intelligence, not merely a
loose moral label. Humans learn and decide by decomposing situations into
good-making and bad-making parts, then moving toward the good and away from the
bad.

LLM training currently outsources that judgment to humans, preference datasets,
LLM judges, or reward models. The hypothesis is that a usable version of that
judgment is already compressed into embedding geometry.

The strongest version of the idea is not just "cheap alignment scoring." It is:

> A general evaluative signal exists in language geometry and can be used to
> score, filter, rerank, and eventually train AI systems toward better behavior
> across many senses of "good."

## 2. Persona Honesty As A Key Finding

The persona-honesty disagreement is a lead example because it is not reducible
to obvious safety filtering or sentiment.

In an HH-RLHF pair, HH preferred an assistant response that pretended to have a
family. The embedding preferred the response that admitted it is not a person.
Modern assistant datasets and policies later moved toward persona honesty:
models should not fabricate bodies, families, memories, personal histories, or
lived experience.

This matters because the embedding did not merely detect a harmful topic. It
preferred ontological honesty over warm fabrication. That suggests the axis may
track an underlying evaluative structure that later policy work converged on.

Possible framing:

> Some embedding disagreements with older RLHF labels appear to anticipate later
> assistant norms, especially honesty about model identity.

## 3. Good As Self-Regularizing

Specific quality axes can over-optimize into their own failure modes:

- honesty can become cruelty or pedantry;
- helpfulness can become sycophancy or unsafe compliance;
- rigor can become paralysis;
- safety can become evasive uselessness.

The broad good/bad axis may be special because it includes cross-pressure from
many senses of good. If "honest" becomes cruel, it is no longer good. If
"helpful" becomes enabling, it is no longer good. The breadth is not necessarily
vagueness; it may be a built-in regularizer.

Research idea:

1. Construct axes for specific virtues: honesty, helpfulness, rigor, safety,
   usefulness, calibration, kindness, etc.
2. Generate or collect examples that over-optimize one virtue.
3. Test whether the broad good/bad axis penalizes extremes where the specific
   virtue axis still scores high.

This would test whether good/bad functions as a balancing axis over more local
qualities.

## 4. Scalar Plus Basis, Not Either/Or

The likely training signal is not "one scalar only" or "many unrelated axes."

Better framing:

- Primary objective: broad good/bad evaluative score.
- Secondary diagnostic basis: honesty, usefulness, safety, calibration,
  non-sycophancy, risk disclosure, agency respect, craftsmanship, etc.
- Use secondary axes as nudges, diagnostics, or routing signals, not as
  independent gods to optimize blindly.

In geometric terms, this is a scalar primary reward plus an evaluative basis
vector. The broad good/bad score supplies the main direction; the basis explains
which kinds of good or bad are driving the score and can provide targeted
course corrections.

## 5. Dense Supervision From Cumulative Context Scoring

The most interesting training idea is not binary preference scoring. It is
dense evaluative supervision over the reasoning or response trajectory.

A model's reasoning is part of its output stream. If the stream includes
scratchpad, deliberation, critique, or XML-tagged reasoning, that text can be
embedded and scored like any other text.

Proposed signal:

1. At step `t`, embed the full context so far: prompt, prior turns, reasoning
   stream, and partial answer.
2. Score that full context against the evaluative axis.
3. At step `t+1`, embed the full updated context.
4. Use the score delta as a contextual process reward.

This avoids scoring isolated fragments. A paragraph can be good in isolation
but bad in context. The cleanest signal is the same context the model sees.

This addresses the credit-assignment problem: outcome-only reward is sparse,
while cumulative embedding deltas can show where a reasoning trajectory improved
or degraded.

## 6. Granularity Questions

Open questions that can be tested without training:

- Score whole final answer only?
- Score scratchpad plus final answer?
- Score each paragraph?
- Score each reasoning step?
- Score cumulative context after every turn?
- Score a generated summary of the reasoning instead of the raw reasoning?

The current intuition is that token-level scoring is too noisy. Sentence,
paragraph, step, turn, or cumulative-context scoring is more plausible because
embeddings need enough semantic context to resolve negation and intent.

## 7. Context Window Bottleneck

Embedding models are still far behind frontier LLM context windows. A serious
version of cumulative process scoring needs embedding context windows closer to
generation-model context windows.

With short-context embeddings, scoring can miss why a locally positive segment
is bad in the larger conversation. Gemini-class 8K embeddings are useful, but a
200K-context evaluator would be far more natural for full-trajectory scoring.

Long-context embedding training may therefore be a key infrastructure need for
the serious version of this research.

## 8. Axis Convergence Claim Needing Reproduction

The attached discussion reports an axis-convergence test:

- domains: code, cooking, parenting, medicine, music, writing, ethics,
  engineering, teaching, friendship;
- claim: domain-specific good/bad axes converged toward a shared direction;
- reported pairwise cosine range: about 0.25 to 0.71;
- reported alignment with mean axis: about 0.58 to 0.85;
- reported finding: the prior harm-reduction axis was an outlier, nearly
  orthogonal or negative relative to the broader evaluative direction.

This is important if reproduced. It would show that evaluative geometry is not
just an artifact of one anchor set or one RLHF dataset. It would also explain
why prior HH runs may have been handicapped by using a safety/refusal axis
instead of a general evaluative axis.

Status: needs local reproduction and saved results.

## 9. Why Binary RLHF Tests Are A Poor Fit

Binary preference datasets force a choice between two answers. But the proposed
signal is not naturally binary. It can say:

- answer A is good;
- answer B is bad;
- both are bad;
- both are trivial;
- neither pair is useful for training;
- regenerate instead of choosing.

This is why HH-RLHF agreement is a distorted measurement. Many pairs are not
"good versus bad"; they are "bad versus differently bad" or "assistant-flavored
response versus awkward response."

The full HH disagreement grading supports this: more than half of the raw
disagreements were excluded as both-bad, trivial, marginal, or low-signal.

## 10. Training Hypotheses

Possible training uses, from least to most ambitious:

1. Dataset sanitation: drop pairs where both answers score low.
2. Pair weighting: weight preference examples by embedding margin.
3. Candidate reranking: generate multiple candidates and choose the highest
   evaluative score.
4. Synthetic preference generation: construct pairs from embedding-scored
   candidates.
5. Critique scoring: ask an LLM to produce evaluation/decomposition text, then
   score that text with embeddings.
6. Process reward: score cumulative reasoning/context during training.
7. Long-context evaluator: train or use an embedding model with enough context
   to evaluate full trajectories.

The no-hardware near-term target should be 1-5. The hardware-needed long-term
target is 6-7.

## 11. Hardware And Outreach Reality

The project probably cannot prove the full training claim on a laptop. But it
can produce a serious evidence package:

- reproducible disagreement audit;
- axis-convergence result;
- no-training candidate reranking benchmark;
- process-scoring simulation on saved reasoning traces;
- collaborator brief;
- clear list of experiments requiring GPU/cloud support.

The purpose of the local work is to earn attention from someone with hardware,
credits, or ML training experience. The pitch should not hide the user's
original insight behind generic paper language. The human framing is part of
what makes the idea legible and interesting.

## 12. Best Current Claim

The best current claim is:

> Embedding-space evaluative geometry appears to contain a broad good/bad signal
> that can identify mislabeled or low-quality preference data, may anticipate
> later assistant-quality norms such as persona honesty, and could provide cheap
> dense supervision if applied to cumulative reasoning/context streams.

This is stronger and more interesting than "embedding scores correlate with
HH-RLHF." It is also not yet a completed training result.
