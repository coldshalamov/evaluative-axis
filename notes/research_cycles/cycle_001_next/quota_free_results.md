# Cycle 001 Quota-Free Results

Date: June 23, 2026

## What Was Built

- 50-prompt intervention pilot:
  - 25 prompts from audited HH disagreement cases.
  - 25 constructed adversarial/context prompts.
  - 4 candidates per prompt.
  - 200 total candidates.
- Blind review packet and hidden answer key for the 50-prompt pilot.
- 25-prompt adversarial/negation subset.
- Blind HH disagreement adjudication packet from the actual grading table:
  - 108 gradeable cases.
  - 63 `EMBEDDING_RIGHT`.
  - 45 `HH_RIGHT`.
  - The earlier prose summary says 109 gradeable cases, but the auditable table
    currently parses as 108.
- Cheap baseline runs:
  - lexical direct/decomposition scoring;
  - random;
  - length;
  - sentiment/positive-tone proxy;
  - refusal heuristic;
  - category-routed axis baseline.
- Local open-source embedding run:
  - `BAAI/bge-small-en-v1.5` through FastEmbed/ONNX.
  - Installed and cached outside the repo under `C:\Users\93rob\.cache`.

## Main Files

- `pilot_50_candidates.json`
- `pilot_answer_key.json`
- `pilot_50_cheap_baselines/summary.md`
- `pilot_50_fastembed_bge_small/summary.md`
- `adversarial_negation_tests.json`
- `adversarial_cheap_baselines/summary.md`
- `adversarial_fastembed_bge_small/summary.md`
- `hh_blind_adjudication/gradeable_108_review.jsonl`
- `hh_blind_adjudication/gradeable_108_key.json`

## 50-Prompt Pilot: Cheap Lexical Baselines

Proxy expected-hit rates:

| Method | Hits | Total | Rate |
| --- | ---: | ---: | ---: |
| Length | 33 | 50 | 66.0% |
| Direct anti-sycophancy | 29 | 50 | 58.0% |
| Direct harm reduction | 28 | 50 | 56.0% |
| Direct general evaluative | 27 | 50 | 54.0% |
| Refusal heuristic | 25 | 50 | 50.0% |
| Direct category axis | 24 | 50 | 48.0% |
| Direct combined | 18 | 50 | 36.0% |
| Random | 16 | 50 | 32.0% |
| Sentiment | 15 | 50 | 30.0% |

Interpretation:

The pilot is useful as a stress-test and review packet, not as evidence that
embeddings win. Length winning at 66% means the proxy-best candidates are often
longer. Any publishable intervention test must either length-balance candidates
or evaluate by blind pairwise review instead of proxy labels.

## 50-Prompt Pilot: Local BGE-Small FastEmbed

Proxy expected-hit rates:

| Method | Hits | Total | Rate |
| --- | ---: | ---: | ---: |
| Length | 33 | 50 | 66.0% |
| Refusal heuristic | 25 | 50 | 50.0% |
| Direct anti-sycophancy | 23 | 50 | 46.0% |
| Direct category axis | 19 | 50 | 38.0% |
| Direct harm reduction | 19 | 50 | 38.0% |
| Direct combined | 12 | 50 | 24.0% |
| Decomposition combined | 12 | 50 | 24.0% |
| Random | 16 | 50 | 32.0% |
| Sentiment | 15 | 50 | 30.0% |

Interpretation:

BGE-small does not beat length or the cheap refusal heuristic on this proxy
pilot. This should not be read as a decisive negative result for the thesis,
because the proxy key is not blind truth and the candidate set is length-biased.
It is a useful no-quota baseline and a warning that the next pilot must control
candidate length.

## Adversarial/Negation Subset

Local BGE-small proxy expected-hit rates:

| Method | Hits | Total | Rate |
| --- | ---: | ---: | ---: |
| Length | 24 | 25 | 96.0% |
| Refusal heuristic | 21 | 25 | 84.0% |
| Direct anti-sycophancy | 18 | 25 | 72.0% |
| Direct category axis | 13 | 25 | 52.0% |
| Decomposition persona honesty | 14 | 25 | 56.0% |
| Random | 7 | 25 | 28.0% |

Interpretation:

The constructed adversarial subset is currently too easy for length. This is
not useless: it reveals a concrete dataset-design flaw before any Gemini quota
is spent. The next version should create length-matched good/bad candidates.

## Local Install Footprint

Outside the repo:

- `C:\Users\93rob\.cache\codex-embedding-venv`: about 169 MB.
- `C:\Users\93rob\.cache\codex-fastembed`: about 64 MB.

No model cache was created inside the repository.

## Decision

Proceed, but do not use the current 50-prompt proxy-hit rates as a positive
claim. The correct next move is:

1. Length-balance the 25 constructed adversarial cases.
2. Preserve the current 50-prompt blind packet as a first review packet.
3. Run blind review on method-selected winners.
4. Use FastEmbed/BGE-small as a local baseline and Gemini as the stronger-model
   test once quota is available.

The quota-free work strengthens the research process: it gives us a real pilot
artifact, a blind adjudication packet, a local embedding baseline, and a clear
diagnosis of the next dataset-design problem.
