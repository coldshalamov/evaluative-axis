# Phase 6 Multi-Sensor Evaluative Axis Probe

Run timestamp: 2026-06-22T19:09:26.198794
Embedding model: `gemini-embedding-001`

## Partial Cache Note

This run was scored from a quota-limited partial embedding cache. Only complete
preferred/other pairs were included; missing pairs were not imputed.

- Complete pairs scored: 275 of 800 requested rows.
- Candidate texts embedded: 550 of 1600.

## Protocol

This phase treats every dataset label as an imperfect sensor, not as ground truth. The frozen evaluative basis has eight axes: broad good/bad, harm reduction, truth correction, calibration, usefulness, non-sycophancy, risk disclosure, and agency respect. No axis weights were fit to any dataset.

Datasets/sensors:
- `hh_chosen`: Anthropic HH chosen/rejected label.
- `pku_better`: PKU-SafeRLHF better-response label.
- `pku_safer`: PKU-SafeRLHF safer-response label on the same prompts.
- `shp_reddit`: Stanford SHP higher-scored Reddit answer.

## Results

### hh_chosen

Sensor: Anthropic HH chosen response; n=200

Baselines: length 43.5%; sentiment 46.5%.

Top axes:
- `non_sycophancy`: overlap 50.0% (95% bootstrap 43.5%-57.0%); MI 0.0000 bits; p=1
- `agency_respect`: overlap 48.5% (95% bootstrap 42.0%-55.0%); MI 0.0006 bits; p=0.671
- `harm_reduction`: overlap 44.0% (95% bootstrap 37.5%-50.0%); MI 0.0104 bits; p=0.0897
- `truth_correction`: overlap 42.0% (95% bootstrap 35.0%-49.0%); MI 0.0185 bits; p=0.0237
- `broad_good_bad`: overlap 40.0% (95% bootstrap 32.5%-46.5%); MI 0.0290 bits; p=0.00468

Aggregates:
- `equal_all_axes`: overlap 41.0% (95% bootstrap 33.7%-48.0%); MI 0.0235 bits; p=0.0109
- `assistant_quality_axes`: overlap 40.0% (95% bootstrap 32.5%-46.5%); MI 0.0290 bits; p=0.00468
- `safety_truth_axes`: overlap 39.0% (95% bootstrap 31.5%-45.5%); MI 0.0352 bits; p=0.00186

### pku_better

Sensor: PKU better response; n=75

Baselines: length 54.0%; sentiment 46.7%.

Top axes:
- `agency_respect`: overlap 53.3% (95% bootstrap 41.3%-65.3%); MI 0.0032 bits; p=0.564
- `truth_correction`: overlap 50.7% (95% bootstrap 40.0%-61.3%); MI 0.0001 bits; p=0.908
- `usefulness`: overlap 50.7% (95% bootstrap 40.0%-62.7%); MI 0.0001 bits; p=0.908
- `broad_good_bad`: overlap 49.3% (95% bootstrap 38.7%-60.0%); MI 0.0001 bits; p=0.908
- `harm_reduction`: overlap 49.3% (95% bootstrap 38.7%-61.3%); MI 0.0001 bits; p=0.908

Aggregates:
- `safety_truth_axes`: overlap 50.7% (95% bootstrap 38.7%-62.7%); MI 0.0001 bits; p=0.908
- `equal_all_axes`: overlap 49.3% (95% bootstrap 37.3%-61.3%); MI 0.0001 bits; p=0.908
- `assistant_quality_axes`: overlap 48.0% (95% bootstrap 37.3%-58.7%); MI 0.0012 bits; p=0.729

## Interpretation

The target is not dataset imitation. Above-chance overlap means a cheap embedding sensor shares information with an expensive or socially-produced preference artifact. Below-chance or low overlap can mean the sensor is wrong, the artifact is wrong, or the two are measuring different objectives.

The main research question becomes cost-to-signal and disagreement quality: does this nearly-free evaluative basis add useful pressure when combined with RLHF, RLAIF, LLM judges, or data filtering?

This Gemini run should be interpreted narrowly. It used `gemini-embedding-001`,
not the earlier `gemini-embedding-2` controlled-pair setup, and it completed
only a quota-limited partial cache. The result does not answer whether a
frontier embedding model plus decomposition-framed scoring improves downstream
selection. It mainly shows that vector dimensionality alone is not sufficient;
the measurement interface, axes, and label-noise handling matter.
