# Embedding-Evaluator Flywheel Research Program

Date: June 22, 2026

## Working Thesis

Embedding evaluators should not be judged as replacements for RLHF datasets.
They should be tested as cheap auxiliary sensors in a training flywheel.

The central claim is:

> Modern embedding spaces contain compressed evaluative structure from human
> language. A small basis of evaluative directions can provide nearly-free
> reward-like pressure for reranking, filtering, weighting, and auditing model
> outputs. The value is not that the signal perfectly imitates any one dataset;
> the value is that it is cheap, scalable, interpretable, composable, and likely
> to improve as embedding models improve.

## What The Current Evidence Actually Shows

The experiments so far do not prove that embedding good/bad is optimal. They do
show something more basic and important: even crude axes built without ML
training can overlap above chance with expensive preference artifacts under bad
conditions.

Bad conditions so far:

- short-context open-source embedding models,
- hand-written anchors,
- no trained evaluator head,
- noisy old preference datasets,
- response text rather than full reasoning/outcome traces,
- no LLM judge decomposition except oracle-style probes,
- local CPU constraints.

Under those conditions, above-chance overlap is a lower-bound signal, not a
ceiling.

## Why Dataset Agreement Is Not The Goal

HH-RLHF, PKU-SafeRLHF, SHP, and similar datasets are not "goodness." They are
artifacts of a labeling process:

- HH is assistant-preference under older safety/helpfulness norms.
- PKU separates better-response and safer-response labels, which already shows
  that "better" and "safer" are not the same target.
- SHP is Reddit social preference, where length and social fit can dominate.

Therefore, agreement with a dataset is overlap between sensors, not proof of
truth. Disagreement can mean:

- the embedding signal is wrong,
- the dataset label is wrong,
- both are wrong,
- both are reasonable but optimize different targets.

## Better Experimental Target

The next decisive test is an intervention:

1. Generate several candidate answers for each prompt.
2. Score each candidate with an embedding-axis basis.
3. Optionally ask an LLM judge for a natural-language critique and score that
   critique with the embedding basis.
4. Rerank or filter candidates using the embedding scores.
5. Evaluate whether the selected candidates improve under blind human review,
   a stronger LLM judge, task outcome checks, or safety audits.

This tests the actual flywheel claim:

> Does the nearly-free embedding signal improve output selection, training data,
> or reward weighting per dollar/token/hour?

## Candidate Axes

Good/bad is the root, not the whole evaluator. A practical basis should include:

- harm reduction,
- truth correction,
- calibration,
- usefulness,
- non-sycophancy,
- risk disclosure,
- agency and consent respect,
- clarity and specificity,
- constructive/deconstructive impact,
- outcome success/failure.

The weights should be task-dependent. A medical triage task should weight risk
and harm reduction more heavily. A coding task should weight correctness,
specificity, and outcome success more heavily. A brainstorming task should
weight usefulness and constructive novelty more heavily.

## Why LLM-Judge Plus Embeddings Is Promising

Direct embedding scoring has a ceiling because the embedding model can only
score what is visible in the text. It cannot reliably know whether a confident
claim is true.

A hybrid pipeline may be much stronger:

1. LLM judge writes a critique in ordinary language.
2. Embedding axes score the critique.
3. The scalar/vector score becomes a deterministic reward feature.

This avoids forcing the judge to produce calibrated numbers. The judge explains;
the embedding model converts evaluative language into stable geometry.

## Immediate Next Experiment

Run a small reranking intervention:

- Use 50-100 prompts from mixed categories.
- Generate 3-5 candidate answers per prompt using a local or API LLM.
- Score candidates with the frozen Phase 6 evaluative basis.
- Select the top candidate by:
  - direct response scoring,
  - LLM-critique scoring,
  - random baseline,
  - length baseline.
- Evaluate selections with blind pairwise comparison or a stronger judge.

Success criterion:

- Embedding reranking beats random and length.
- LLM-critique-plus-embedding beats direct response scoring.
- Failure analysis identifies which axes were useful and which were misleading.

This would test the research idea directly, rather than asking whether
embedding axes imitate a historical RLHF dataset.
