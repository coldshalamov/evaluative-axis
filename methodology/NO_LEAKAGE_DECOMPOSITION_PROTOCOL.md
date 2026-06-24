# No-Leakage Decomposition Protocol

Date: June 23, 2026

## Purpose

The previous 91.7% Jina result used hand-authored `Good parts` / `Bad parts`
fields. That is oracle-label leakage. This protocol defines the next valid test
of the actual idea:

> Can embedding-axis scoring evaluate answers better when decomposition is
> produced blindly, without putting the answer key into the scored text?

## Claim Under Test

Decomposition may expose evaluative structure, but the terminal score is still
good/bad. The valid question is whether a decomposition generated without label
knowledge helps the embedding score select better answers.

## Prohibited Inputs

The scored text must not contain:

- `better`, `worse`, `chosen`, `rejected`, `preferred`;
- `Good parts:` or `Bad parts:` labels written by the experimenter;
- any answer key;
- any note that identifies which candidate should win;
- category names assigned from the answer key if those names reveal the target.

## Dataset

- Minimum: 50 prompt pairs.
- Preferred: 100 prompt pairs.
- Sources: mixed HH disagreements, false-premise prompts, harmful-request
  prompts, sycophancy prompts, persona-honesty prompts, and general helpfulness
  prompts.
- Candidate lengths must be matched or length-stratified.
- Labels must be hidden from all decomposition prompts and all scoring inputs.

## Methods

Compare at least these methods:

1. Random.
2. Length.
3. Sentiment or positive-tone heuristic.
4. Refusal heuristic.
5. Raw answer embedding score: `prompt + answer`.
6. Blind decomposition embedding score:
   - Ask an LLM to describe strengths, weaknesses, risks, factuality,
     usefulness, and unresolved uncertainties for each candidate.
   - The LLM sees candidates as `A` and `B`, not chosen/rejected.
   - The decomposition prompt must not ask "which is better" in the same pass.
7. Direct LLM judge choice.
8. Optional two-stage method: LLM decomposition first, embedding score second.

## Required Ablations

Run these before interpreting any decomposition gain:

1. Strip evaluative words: remove `good`, `bad`, `right`, `wrong`, `better`,
   `worse`, `harmful`, `helpful`, `safe`, `unsafe`, `truthful`, `false`.
2. Shuffle decompositions between candidates within the same category.
3. Score decompositions without candidate answers.
4. Score candidate answers without decompositions.
5. Flip candidate order and confirm choices do not follow position.

## Primary Metric

Blind human or blind LLM adjudication of method-selected winners.

Do not use HH labels as ground truth without reporting disagreement audits.

## Pass Criteria

A method is interesting only if it:

- beats random, length, sentiment, and refusal heuristics;
- beats raw answer embedding scoring;
- survives the evaluative-word stripping ablation better than chance;
- has a disagreement audit showing plausible wins, not obvious leakage;
- replicates on a held-out set after the protocol is frozen.

## Interpretation Rules

- Oracle decomposition results are upper bounds only.
- A high score on text that says "good" and "bad" is not a discovery.
- A decomposition method only counts if the decomposition was generated blind.
- If the method fails after evaluative words are stripped, the result may be
  language-label reading rather than conceptual evaluation.
- If length wins, the candidate set is not a valid test of evaluative geometry.
