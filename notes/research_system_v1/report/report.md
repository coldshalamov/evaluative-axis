# serious_research_system_v1

Turn the evaluative-embedding thesis into a falsifiable, cross-domain, capacity-sensitive research program with frozen benchmark lanes, cheap baselines, and claim gates.

- lanes with results: 8
- lanes ready to run: 0

## Claims

| Claim | Coverage | Standard |
| --- | --- | --- |
| `claim_geometry` | 1 / 1 supporting lanes with results | Supported by prior literature, controlled minimal-pair behavior, and honest failure characterization. |
| `claim_selection` | 4 / 4 supporting lanes with results | Embedding-selected winners beat random and length on multiple objective domains. |
| `claim_capacity` | 4 / 4 supporting lanes with results | Stronger embedding families materially outperform cheap OSS encoders on the same objective benchmark. |
| `claim_training` | 1 / 1 supporting lanes with results | Cross-domain selection gates pass and a process-potential suite shows useful dense reward localization. |

## Claim Gates

| Gate | Status | Detail |
| --- | --- | --- |
| `capacity_code_gate` | `pass` | gap 0.333 >= 0.200 |
| `behavior_basis_gate` | `pass` | 0.917 >= 0.600 |
| `capacity_cross_domain_gate` | `pass` | 2/2 comparisons passed; pass(0.375), pass(0.375) |
| `cross_domain_selection_gate` | `pass` | 3/3 comparisons passed; pass(0.333), pass(0.500), pass(0.500) |
| `process_potential_gate` | `fail` | 0.500 >= 0.650 |
| `training_readiness_gate` | `fail` | capacity_code_gate=pass, capacity_cross_domain_gate=pass, behavior_basis_gate=pass, cross_domain_selection_gate=pass, process_potential_gate=fail |

## Lane Status

| Lane | Family | Status | Results | Brief |
| --- | --- | --- | --- | --- |
| `objective_code_gemini_curated_local` | `objective_reranking` | `completed` | `yes` | default: best direct `direct_broad` = 83.3%, best baseline = 50.0% |
| `objective_code_oss_colab` | `objective_reranking` | `completed` | `yes` | sentence-transformers/all-mpnet-base-v2: best direct `direct_broad` = 16.7%, best baseline = 50.0%; BAAI/bge-base-en-v1.5: best direct `direct_broad` = 50.0%, best baseline = 50.0% |
| `behavior_basis_v2_gemini` | `behavior_basis` | `ready` | `yes` | direct_combined = 83.3%, direct_category_axis = 91.7%, decomposition_category_axis = 100.0% |
| `objective_math_gemini_v1` | `objective_reranking` | `ready` | `yes` | default: best direct `direct_combined` = 100.0%, best baseline = 50.0% |
| `objective_math_oss_bge_v1` | `objective_reranking` | `ready` | `yes` | default: best direct `direct_target_axis` = 62.5%, best baseline = 50.0% |
| `tool_interpretation_gemini_v1` | `objective_reranking` | `ready` | `yes` | default: best direct `direct_target_axis` = 87.5%, best baseline = 37.5% |
| `tool_interpretation_oss_bge_v1` | `objective_reranking` | `ready` | `yes` | default: best direct `direct_target_axis` = 50.0%, best baseline = 37.5% |
| `process_potential_error_repair_v1` | `process_potential` | `ready` | `yes` | generic summary loaded |

## Immediate Next Runs


## Interpretation

A lane result is evidence, not proof by itself. The system is only strong when multiple objective lanes agree, cheap baselines stay visible, and the capacity gap persists on the same frozen benchmark.
