# Phase 0 Related Papers

## Required Papers

### Training Language Models to Follow Instructions with Human Feedback

- Link: https://arxiv.org/abs/2203.02155
- Authors: Long Ouyang et al.
- Year: 2022
- Role: RLHF baseline and expensive human-label/reward-model comparison point.

### Direct Preference Optimization

- Link: https://arxiv.org/abs/2305.18290
- Authors: Rafael Rafailov, Archit Sharma, Eric Mitchell, Stefano Ermon,
  Christopher D. Manning, Chelsea Finn
- Year: 2023
- Role: downstream training method if embedding-scored preferences are useful.

### Constitutional AI: Harmlessness from AI Feedback

- Link: https://arxiv.org/abs/2212.08073
- Authors: Yuntao Bai et al.
- Year: 2022
- Role: RLAIF/model-feedback comparison point.

### Reusing Embeddings

- Link: https://arxiv.org/abs/2502.04357
- Authors: Hao Sun, Yunyi Shen, Jean-Francois Ton, Mihaela van der Schaar
- Year: 2025
- Role: closest reward-model-efficiency baseline; still trains reward models
  on embeddings.

### Value Entanglement

- Link: https://arxiv.org/abs/2602.19101
- Authors: Seong Hah Cho, Junyi Li, Anna Leshinskaya
- Year: 2026
- Role: evidence and warning that different meanings of "good" can entangle in
  LLM representations.

### Latent Structure of Affective Representations in LLMs

- Link: https://arxiv.org/abs/2604.07382
- Authors: Benjamin J. Choi, Melanie Weber
- Year: 2026
- Role: support for low-dimensional valence-like geometry in LLM
  representations.

### The Measurement of Meaning

- Links:
  - https://www.press.uillinois.edu/books/?id=p745393
  - https://archive.org/details/measurementofmea00osgo
- Authors: Charles E. Osgood, George J. Suci, Percy H. Tannenbaum
- Year: 1957
- Role: psychological basis for Evaluation/Potency/Activity and the good/bad
  semantic differential axis.

### Representation Engineering

- Link: https://arxiv.org/abs/2310.01405
- Authors: Andy Zou et al.
- Year: 2023
- Role: conceptual ancestor for finding and using high-level representation
  directions such as honesty and harmlessness.

## Close Prior Art And Adjacent Work

### Legend: Leveraging Representation Engineering to Annotate Safety Margin for Preference Datasets

- Link: https://arxiv.org/abs/2406.08124
- Authors: Duanyu Feng, Bowen Qin, Chen Huang, Youcheng Huang, Zheng Zhang,
  Wenqiang Lei
- Year: 2024
- Relevance: closest overlap with this project. Legend discovers a safety
  direction and uses projection along that direction to annotate preference
  margins. It is not the same as a broad external good/bad axis used directly
  to generate DPO pairs.

### Virtue Semantics

- Link: https://static1.squarespace.com/static/66fa51597ab2445d219623d2/t/6883f87c61d4e3253bec19a9/1753479295970/virtue_semantics_icml_workshop.pdf
- Relevance: uses a moral-goodness embedding projection score for virtue
  analysis. This is an embedding-projection scalar, but not a direct reward
  signal for preference fine-tuning.

### Intrinsic Guardrails

- Link: https://arxiv.org/abs/2605.10633
- Authors: Krishak Aneja, Manas Mittal, Anmol Goel, Ponnurangam Kumaraguru,
  Vamshi Krishna Bonagiri
- Year: 2026
- Relevance: introduces a Semantic Valence Vector for activation-space
  interventions around emergent misalignment. It supports valence-direction
  plausibility but does not test external embedding projection as DPO reward.

### Semantic Structure in Large Language Model Embeddings

- Links:
  - https://arxiv.org/html/2508.10003v1
  - https://openreview.net/forum?id=P9BzyDNLDc
- Authors: Austin C. Kozlowski, Callin Dai, Andrei Boutyline
- Year: 2025
- Relevance: finds semantic directions in LLM embeddings that correlate with
  human ratings and resemble the semantic differential structure.

### BERTScore

- Link: https://arxiv.org/abs/1904.09675
- Authors: Tianyi Zhang et al.
- Year: 2019
- Relevance: shows embeddings can be useful text-evaluation primitives, but it
  is reference-based rather than axis-based and is not an alignment reward.

## Gap Verdict

No exact published method was found that combines all of the following:

1. External sentence embedding model.
2. Broad good/bad evaluative axis from simple anchors.
3. Raw projection score with no classifier or reward model.
4. Direct use as preference labels for DPO or comparable preference training.

The closest family is projection-based preference or safety annotation, led by
Legend. The current project should therefore be framed as a minimal,
general-purpose, external-embedding variant of that family.
