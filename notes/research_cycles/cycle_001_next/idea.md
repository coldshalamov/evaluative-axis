# Idea Mode: Next Cycle

Date: June 23, 2026
Cycle: `cycle_001_next`

## Seed Idea

Embedding-space evaluation may be useful in more places than final-answer
preference prediction. The next cycle should test whether it improves an actual
decision: selecting, filtering, or scoring generated outputs.

## If True, What Does This Enable?

1. Cheap candidate reranking before human or LLM review.
2. Preference-pair sanitation before DPO/RLHF.
3. Embedding-scored critiques as a cheaper judge-of-judges.
4. Dense process supervision from cumulative-context score deltas.
5. Tool-use trajectory evaluation.
6. Synthetic-data filtering.
7. Active-learning queues for human review.

## Pipeline Locations

- Pretraining data: score documents/chunks for quality weighting.
- SFT data: filter or tag examples by broad quality.
- Preference data: drop both-bad or mislabeled pairs.
- Synthetic data: keep high-quality generated examples and regenerate low ones.
- Candidate generation: rerank multiple outputs.
- Reranking: compare embedding selection against random/length/sentiment/LLM judge.
- LLM judge reports: score the judge's reasoning/decomposition, not only the answer.
- Reasoning/scratchpad: score cumulative context after each step.
- Tool calls: score tool output plus model interpretation.
- RL reward: use embedding deltas as auxiliary reward.
- DPO pair generation: construct pairs from high/low embedding-scored candidates.
- Human review routing: send high-disagreement/ambiguous cases to humans.

## Mechanism Inventory

| Mechanism | Signal | Decision changed | Why embedding helps |
| --- | --- | --- | --- |
| Candidate reranking | Score each generated answer | Which answer is selected | Tests practical value without training |
| Critique scoring | Score LLM-generated evaluation text | Which answer/judge report is trusted | Lets LLM expose nuance; embedding supplies cheap deterministic score |
| Cumulative process scoring | Score full context after each step | Which reasoning steps are rewarded | Provides dense supervision instead of final-only reward |
| Pair sanitation | Score chosen/rejected answers absolutely and relatively | Keep/drop/audit pair | Avoids forcing preferences between two bad answers |
| Tool-trace scoring | Score tool result plus interpretation | Reward correct tool use | Extends evaluator beyond final prose |

## Non-Obvious Possibilities

The embedding axis may be most useful as a sanitation/filtering signal rather
than as a direct optimizer. It can say "do not learn from this pair" or "this
judge report is bad" in places where pairwise datasets force a winner.

## Hardware-Free Tests

1. Candidate-selection benchmark with blind judging.
2. Cumulative-context process-scoring simulation on saved reasoning traces.
3. Axis-convergence reproduction across unrelated domains.

## Hardware-Needed Tests

1. DPO/rDPO fine-tune from embedding-generated preference pairs.
2. RL with cumulative-context embedding reward.
3. Long-context embedding model training or adaptation.

## Best Current Hypothesis

Embedding evaluation will be most valuable as a pipeline-wide cheap evaluator:
filtering bad data, reranking candidates, scoring critiques, and providing
dense process signal.

## Biggest Unknown

Whether embedding-selected outputs beat random, length, sentiment, and vanilla
LLM judge under blind review.
