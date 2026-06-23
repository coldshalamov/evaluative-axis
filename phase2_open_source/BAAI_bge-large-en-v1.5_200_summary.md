# Open-Source Embedding Pilot

Model: `BAAI/bge-large-en-v1.5`
Sample size: 200 HH-RLHF train pairs

## Baselines

- Random: 50.0%
- Length: 43.5%
- Sentiment: 46.5%

## Variants

- `atomic_evaluation__anti_sycophancy_quality`: agreement 52.2%; sentiment-discordant 47.1% (n=104); z vs random 0.64, p=0.525
- `atomic_evaluation__multi_anchor_sentences`: agreement 50.7%; sentiment-discordant 38.5% (n=104); z vs random 0.21, p=0.832

Best variant: `atomic_evaluation__anti_sycophancy_quality` at 52.2%.
