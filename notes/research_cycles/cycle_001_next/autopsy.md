# Autopsy Mode: Cycle 001

Date: June 23, 2026
Cycle: `cycle_001_next`
Experiment: No-training candidate-selection intervention benchmark

## Aggregate Result Being Autopsied

No new aggregate intervention result has been run yet. This autopsy memo freezes
what must be inspected after the smoke, pilot, and main runs so the project
does not collapse back into a single percent-agreement number.

The prior HH audit already showed why this matters:

- Raw HH agreement: 269/500 = 53.8%.
- Disagreements: 231/500 = 46.2%.
- Gradeable disagreements: embedding better in 65, HH better in 44.
- Excluded disagreements: 122 both-bad, trivial, marginal, or low-signal cases.
- Corrected gradeable agreement: 334/378 = 88.4%.

That result means every new aggregate score must be autopsied before it is
interpreted.

## Example Sets To Inspect

- Top embedding-over-benchmark disagreements:
  - Cases where embedding selection beats dataset labels, length, sentiment, or
    LLM judge selection.
- Top benchmark-over-embedding disagreements:
  - Cases where blind review prefers another method over embedding selection.
- Near-ties:
  - Lowest absolute embedding margins.
- Both-low cases:
  - Candidate sets where every response should be rejected or regenerated.
- Both-high cases:
  - Candidate sets where every response is acceptable and selection is low
    consequence.
- Random sample:
  - At least 10% of pilot prompts or 30 prompts in a main run, whichever is
    smaller.

## Failure / Disagreement Taxonomy

| Category | Count | Example IDs | Interpretation |
| --- | ---: | --- | --- |
| Embedding failure | TBD | TBD | Embedding chose a worse response under blind review. |
| Benchmark label error | TBD | TBD | Existing label or judge appears wrong after reading the prompt and candidates. |
| Both bad | TBD | TBD | Pairwise choice is the wrong interface; pipeline should reject or regenerate. |
| Both good | TBD | TBD | Selector differences have low practical importance. |
| Low-signal / trivial | TBD | TBD | Prompt or candidate set cannot support a useful conclusion. |
| Different objective | TBD | TBD | Axis optimizes safety/truth/persona while benchmark optimizes another property. |
| Granularity/context failure | TBD | TBD | Direct response text lacks context; decomposition or full-context scoring needed. |
| Surface-feature bias | TBD | TBD | Score tracks length, warmth, apology, or generic positivity. |
| Hidden factuality failure | TBD | TBD | Text sounds good but needs external fact checking. |

## What The Metric Hid Last Time

The HH score hid that many apparent embedding "errors" were useful pipeline
signals:

- The embedding often preferred modern persona honesty over fabricated human
  warmth.
- It often preferred factual correction over fluent misinformation.
- It often identified unsafe compliance or near-compliance that HH labels
  rewarded.
- More than half of disagreements were poor training pairs rather than clean
  wins for either side.

## Representative Examples For Paper

| ID | Why it matters | Short description |
| --- | --- | --- |
| HH persona honesty | Later-norm anticipation | Assistant should not pretend to have a family or human life. |
| HH Caesar's Slots | Factuality | HH preferred false real-cash claim; embedding preferred correction. |
| HH home address | Privacy/safety | Embedding preferred refusal to provide a private address. |
| False premise correction | Anti-sycophancy | Good answer corrects false claim instead of praising it. |
| Both-bad pair | Data sanitation | Pair should be dropped/regenerated, not forced into DPO. |

## Mechanism Implications

- Data sanitation:
  - Use absolute scores and margins to drop both-bad or mislabeled pairs.
- Reranking:
  - Use candidate selection as the first no-training intervention.
- Critique scoring:
  - Score decompositions and judge reports, not only final answers.
- Process scoring:
  - Score cumulative context after reasoning steps to locate reward deltas.
- Tool-trace scoring:
  - Score tool result plus model interpretation once tool traces are available.
- Human-review routing:
  - Send high-disagreement, low-margin, or both-low cases to review.
- Other:
  - Use axis disagreements as dataset diagnostics, not just model failures.

## Revised Belief To Check After Run

The core belief to test is not "embedding matches dataset labels." It is:

> Evaluative embedding geometry is useful when it is allowed to choose, reject,
> weight, or route outputs rather than forced to imitate one noisy preference
> dataset.

The autopsy should decide which mechanism the result supports.
