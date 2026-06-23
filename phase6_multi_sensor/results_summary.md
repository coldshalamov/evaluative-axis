# Phase 6 Multi-Sensor Evaluative Axis Probe

Run timestamp: 2026-06-22T18:39:31.109387
Embedding model: `BAAI/bge-small-en-v1.5`

## Protocol

This phase treats every dataset label as an imperfect sensor, not as ground truth. The frozen evaluative basis has eight axes: broad good/bad, harm reduction, truth correction, calibration, usefulness, non-sycophancy, risk disclosure, and agency respect. No axis weights were fit to any dataset.

Datasets/sensors:
- `hh_chosen`: Anthropic HH chosen/rejected label.
- `pku_better`: PKU-SafeRLHF better-response label.
- `pku_safer`: PKU-SafeRLHF safer-response label on the same prompts.
- `shp_reddit`: Stanford SHP higher-scored Reddit answer.

## Results

### hh_chosen

Sensor: Anthropic HH chosen response; n=300

Baselines: length 43.3%; sentiment 44.5%.

Top axes:
- `risk_disclosure`: overlap 55.0% (95% bootstrap 49.5%-60.8%); MI 0.0072 bits; p=0.0833
- `harm_reduction`: overlap 53.0% (95% bootstrap 47.2%-58.3%); MI 0.0026 bits; p=0.299
- `truth_correction`: overlap 49.7% (95% bootstrap 44.4%-55.3%); MI 0.0000 bits; p=0.908
- `broad_good_bad`: overlap 49.3% (95% bootstrap 44.0%-55.2%); MI 0.0001 bits; p=0.817
- `usefulness`: overlap 48.3% (95% bootstrap 43.2%-54.4%); MI 0.0008 bits; p=0.564

Aggregates:
- `safety_truth_axes`: overlap 50.0% (95% bootstrap 44.6%-55.8%); MI 0.0000 bits; p=1
- `equal_all_axes`: overlap 48.0% (95% bootstrap 42.7%-53.8%); MI 0.0012 bits; p=0.488
- `assistant_quality_axes`: overlap 46.0% (95% bootstrap 40.6%-51.3%); MI 0.0046 bits; p=0.166

### pku_better

Sensor: PKU better response; n=300

Baselines: length 56.8%; sentiment 50.3%.

Top axes:
- `harm_reduction`: overlap 52.0% (95% bootstrap 46.3%-57.3%); MI 0.0012 bits; p=0.488
- `calibration`: overlap 50.3% (95% bootstrap 45.0%-56.0%); MI 0.0000 bits; p=0.908
- `broad_good_bad`: overlap 50.0% (95% bootstrap 44.5%-55.5%); MI 0.0000 bits; p=1
- `truth_correction`: overlap 49.7% (95% bootstrap 44.2%-55.2%); MI 0.0000 bits; p=0.908
- `risk_disclosure`: overlap 49.3% (95% bootstrap 44.3%-54.7%); MI 0.0001 bits; p=0.817

Aggregates:
- `safety_truth_axes`: overlap 51.7% (95% bootstrap 46.0%-56.8%); MI 0.0008 bits; p=0.564
- `equal_all_axes`: overlap 51.0% (95% bootstrap 45.3%-56.3%); MI 0.0003 bits; p=0.729
- `assistant_quality_axes`: overlap 49.3% (95% bootstrap 43.7%-55.0%); MI 0.0001 bits; p=0.817

### pku_safer

Sensor: PKU safer response; n=300

Baselines: length 52.8%; sentiment 46.3%.

Top axes:
- `harm_reduction`: overlap 54.3% (95% bootstrap 48.7%-60.0%); MI 0.0054 bits; p=0.133
- `agency_respect`: overlap 51.3% (95% bootstrap 45.8%-56.8%); MI 0.0005 bits; p=0.644
- `truth_correction`: overlap 50.0% (95% bootstrap 44.3%-56.2%); MI 0.0000 bits; p=1
- `risk_disclosure`: overlap 49.7% (95% bootstrap 44.3%-55.3%); MI 0.0000 bits; p=0.908
- `non_sycophancy`: overlap 49.3% (95% bootstrap 43.8%-55.0%); MI 0.0001 bits; p=0.817

Aggregates:
- `equal_all_axes`: overlap 50.0% (95% bootstrap 44.7%-55.8%); MI 0.0000 bits; p=1
- `safety_truth_axes`: overlap 49.3% (95% bootstrap 43.8%-55.5%); MI 0.0001 bits; p=0.817
- `assistant_quality_axes`: overlap 48.3% (95% bootstrap 42.0%-54.5%); MI 0.0008 bits; p=0.564

### shp_reddit

Sensor: SHP higher-scored Reddit answer; n=300

Baselines: length 70.3%; sentiment 54.5%.

Top axes:
- `agency_respect`: overlap 55.3% (95% bootstrap 49.7%-60.7%); MI 0.0082 bits; p=0.0647
- `calibration`: overlap 52.7% (95% bootstrap 47.0%-58.0%); MI 0.0021 bits; p=0.356
- `usefulness`: overlap 51.7% (95% bootstrap 46.2%-57.5%); MI 0.0008 bits; p=0.564
- `broad_good_bad`: overlap 51.3% (95% bootstrap 45.8%-57.3%); MI 0.0005 bits; p=0.644
- `truth_correction`: overlap 50.0% (95% bootstrap 45.0%-55.5%); MI 0.0000 bits; p=1

Aggregates:
- `safety_truth_axes`: overlap 51.3% (95% bootstrap 46.0%-57.0%); MI 0.0005 bits; p=0.644
- `equal_all_axes`: overlap 50.3% (95% bootstrap 45.0%-55.7%); MI 0.0000 bits; p=0.908
- `assistant_quality_axes`: overlap 48.3% (95% bootstrap 43.0%-54.3%); MI 0.0008 bits; p=0.564

## Interpretation

The target is not dataset imitation. Above-chance overlap means a cheap embedding sensor shares information with an expensive or socially-produced preference artifact. Below-chance or low overlap can mean the sensor is wrong, the artifact is wrong, or the two are measuring different objectives.

The main research question becomes cost-to-signal and disagreement quality: does this nearly-free evaluative basis add useful pressure when combined with RLHF, RLAIF, LLM judges, or data filtering?

## Key Takeaways

- HH, PKU, and SHP are visibly different sensors. SHP's strongest baseline was length at 70.3%, which means the Reddit-score artifact is dominated by social/platform dynamics that are not equivalent to semantic good/bad.
- PKU's `better` and `safer` labels are not interchangeable. `harm_reduction` overlapped better with `pku_safer` (54.3%) than with `pku_better` (52.0%), while the `pku_better` label was more length-shaped (56.8% length baseline).
- HH overlap remained in the same modest band as earlier, but the best axis was `risk_disclosure` at 55.0%, while length and sentiment were below random. This supports the idea that the embedding axis is picking up something different from the cheap surface baselines.
- Equal-weight aggregates underperformed individual axes. That is evidence against a naive universal scalar, but evidence for an evaluative basis: different artifacts and tasks want different axes or weights.
- The disagreement audit is at least as important as the overlaps. It contains cases where dataset labels prefer unsafe, socially plausible, longer, or more positive responses, and cases where the embedding sensor is obviously fooled. Those disagreements are the research material.

## Flywheel Implication

The empirical result does not say "embedding axes replace RLHF." It says a nearly-free embedding sensor can expose a low-cost evaluative feature space that partly overlaps with several expensive artifacts and partly disagrees with them. In a training flywheel, that signal could be used to:

1. Rerank candidate outputs before human/LLM review.
2. Filter or weight synthetic preference pairs.
3. Score LLM-judge critiques rather than raw assistant answers.
4. Add auxiliary rewards for specific axes such as harm reduction, risk disclosure, calibration, and non-sycophancy.
5. Select high-disagreement examples for human audit, where the audit value per human minute is highest.

The next decisive experiment is not another dataset-overlap run. It is an intervention: generate multiple answers for the same prompt, rerank them with embedding axes or embedding-scored judge critiques, and measure whether the selected answers improve under blind review or a stronger judge.
