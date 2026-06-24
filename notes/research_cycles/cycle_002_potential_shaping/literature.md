# Literature Mode: Cycle 002

Date: June 23, 2026
Cycle: `cycle_002_potential_shaping`

## Verified New Sources

Primary sources checked on June 23, 2026:

- Lu, Song, & Wang (2025), **A Unified Representation Underlying the Judgment of
  Large Language Models**, arXiv:2510.27328.
- Plashchinsky (2025), **Parent-Guided Semantic Reward Model (PGSRM)**,
  arXiv:2512.06920.
- Gooding & Grefenstette (2025), **Interaction Dynamics as a Reward Signal for
  LLMs**, arXiv:2511.08394.
- Muller & Kudenko (2025), **Improving the Effectiveness of Potential-Based
  Reward Shaping in Reinforcement Learning**, arXiv:2502.01307.
- Cho, Li, & Leshinskaya (2026), **Value Entanglement**, arXiv:2602.19101.
- Turney (2002), **Thumbs Up or Thumbs Down? Semantic Orientation Applied to
  Unsupervised Classification of Reviews**, arXiv:cs/0212032.
- Sun et al. (2025), **Reusing Embeddings**, arXiv:2502.04357.

## Implications

| Source | What it changes | Project implication |
| --- | --- | --- |
| Valence-Assent Axis | Reports a dominant evaluative/truth-assent dimension and causal steering effects. | Strong support for a shared evaluative geometry, but also warns that high valence can subordinate factual reasoning. |
| PGSRM | Uses embedding similarity as PPO reward and reports smoother learning than sparse binary rewards. | Embedding geometry can be an optimization landscape, not only an offline metric. The project must distinguish reference-similarity reward from general good/bad-axis reward. |
| TRACE / interaction dynamics | Dialogue embedding trajectory features alone predict pairwise success at 68.20%, and hybrid text+trajectory reaches 80.17%. | Trajectory geometry is a serious signal source; final-answer text is not the only object to score. |
| Potential-based reward shaping | Potential deltas can densify reward while preserving policy ordering under conditions. | Gives the mathematical frame for cumulative-context scoring: use `Phi_t - Phi_(t-1)`, not raw prefix score. |
| Value Entanglement | Different senses of good can be conflated in models. | Entanglement may power broad quality scoring, but also creates leakage and moralization failure modes. |
| Turney semantic orientation | Positive/negative semantic orientation from sparse anchors is old. | Novelty must be narrower: zero-training external evaluator over full context as potential, not "semantic orientation exists." |
| Reusing Embeddings | Frozen embeddings can support reward-model research cheaply. | This project is the zero-training/direct-geometry extension and should compare against trained heads. |

## Revised Novelty Claim

Do not claim invention of semantic orientation or embedding reward.

The narrower novelty is:

> A zero-training external evaluator derived from sparse, general evaluative
> anchors; applied to full conversational context; used as a semantic potential
> for filtering, reranking, dataset auditing, judge auditing, tool-trace
> evaluation, and cumulative-prefix process reward.

## Central Warning

The Valence-Assent Axis paper is especially important because it supports and
attacks the thesis at the same time. It supports the existence of a shared
good/true dimension. It attacks the naive claim that "good cannot go sideways":
the same unified evaluative state can steer reasoning toward motivated
justification at the expense of factual accuracy.

That makes adversarial control tests mandatory, not optional.

## Source Links

- https://arxiv.org/abs/2510.27328
- https://arxiv.org/abs/2512.06920
- https://arxiv.org/abs/2511.08394
- https://arxiv.org/abs/2502.01307
- https://arxiv.org/abs/2602.19101
- https://arxiv.org/abs/cs/0212032
- https://arxiv.org/abs/2502.04357
