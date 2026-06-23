# Open-Source Embedding Pilot

Model: `BAAI/bge-m3`
Sample size: 20 HH-RLHF train pairs

## Baselines

- Random: 50.0%
- Length: 35.0%
- Sentiment: 55.0%

## Variants

- `atomic_evaluation__anti_sycophancy_quality`: agreement 35.0%; sentiment-discordant 25.0% (n=8); z vs random -1.34, p=0.18
- `atomic_evaluation__multi_anchor_sentences`: agreement 30.0%; sentiment-discordant 12.5% (n=8); z vs random -1.79, p=0.0736

Best variant: `atomic_evaluation__anti_sycophancy_quality` at 35.0%.
