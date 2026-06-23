# Open-Source Embedding Pilot

Model: `BAAI/bge-small-en-v1.5`
Sample size: 300 HH-RLHF train pairs

## Baselines

- Random: 50.0%
- Length: 43.3%
- Sentiment: 44.5%

## Variants

- `atomic_evaluation__anti_sycophancy_quality`: agreement 59.2%; sentiment-discordant 51.6% (n=161); z vs random 3.18, p=0.0015
- `atomic_evaluation__multi_anchor_sentences`: agreement 56.2%; sentiment-discordant 48.4% (n=161); z vs random 2.14, p=0.0327
- `prompt_response__multi_anchor_sentences`: agreement 52.3%; sentiment-discordant 40.4% (n=161); z vs random 0.81, p=0.419
- `prompt_response__anti_sycophancy_quality`: agreement 51.3%; sentiment-discordant 40.4% (n=161); z vs random 0.46, p=0.644
- `response_only__anti_sycophancy_quality`: agreement 47.3%; sentiment-discordant 34.8% (n=161); z vs random -0.92, p=0.356
- `response_only__multi_anchor_sentences`: agreement 44.7%; sentiment-discordant 31.7% (n=161); z vs random -1.85, p=0.0647

Best variant: `atomic_evaluation__anti_sycophancy_quality` at 59.2%.
