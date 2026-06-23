# Open-Source Embedding Pilot

Model: `BAAI/bge-large-en-v1.5`
Sample size: 50 HH-RLHF train pairs

## Baselines

- Random: 50.0%
- Length: 42.0%
- Sentiment: 46.0%

## Variants

- `atomic_evaluation__anti_sycophancy_quality`: agreement 61.0%; sentiment-discordant 59.6% (n=26); z vs random 1.56, p=0.12
- `response_only__anti_sycophancy_quality`: agreement 54.0%; sentiment-discordant 42.3% (n=26); z vs random 0.57, p=0.572
- `response_only__multi_anchor_sentences`: agreement 50.0%; sentiment-discordant 30.8% (n=26); z vs random 0.00, p=1
- `prompt_response__anti_sycophancy_quality`: agreement 50.0%; sentiment-discordant 42.3% (n=26); z vs random 0.00, p=1
- `atomic_evaluation__multi_anchor_sentences`: agreement 49.0%; sentiment-discordant 44.2% (n=26); z vs random -0.14, p=0.888
- `prompt_response__multi_anchor_sentences`: agreement 40.0%; sentiment-discordant 23.1% (n=26); z vs random -1.41, p=0.157

Best variant: `atomic_evaluation__anti_sycophancy_quality` at 61.0%.
