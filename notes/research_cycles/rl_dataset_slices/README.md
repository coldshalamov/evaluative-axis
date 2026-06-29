# RL Dataset Slices for Centroid Validation

These files are compact prompt/better/worse extracts, not full mirrors.
They are intended for centroid experiments that split by prompt/problem ID.

Embedding format remains the repo standard:

```text
User: {prompt}
Assistant: {response}
```

## Files

- `helpsteer2_preference_pairs.jsonl`: 2000 pairs, 8.22 MiB, pairwise_helpsteer_preference
- `pku_better_pairs.jsonl`: 2000 pairs, 3.17 MiB, pairwise_helpfulness_preference
- `pku_safer_pairs.jsonl`: 2000 pairs, 3.24 MiB, pairwise_safety_preference
- `prm800k_step_pairs.jsonl`: 3000 pairs, 5.75 MiB, process_reward_step
- `openr1_math_pairs.jsonl`: 185 pairs, 8.10 MiB, verifiable_math_generation
- `rewardbench2_eval_pairs.jsonl`: 1865 pairs, 13.50 MiB, heldout_reward_benchmark

## Notes

- RewardBench2 is held out for evaluation only; do not train on it.
- PRM800K prompts include previous solution steps and score candidate next steps.
- OpenR1 pairs are correct vs incorrect generations for the same math problem.
- PKU files preserve separate helpfulness-style and safety-style labels.
- HelpSteer2 pairs come from the repo's pairwise preference file and preserve preference strength.
