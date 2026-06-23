# Codex Goal v2 — Gemini Rerun (<2500 chars)

```
Rerun Phases 1-2 of the embedding reward signal experiment using Gemini Embedding 2 instead of the all-mpnet fallback. The first run got 53.2% on all-mpnet — statistically significant but too weak. This run tests whether a stronger model produces a usable signal.

PREREQUISITE: Gemini API key must be available. Check environment variable GOOGLE_API_KEY or GEMINI_API_KEY. If in Colab, check userdata.get('GOOGLE_API_KEY'). If no key is found, STOP and document "no key available" in research_log.md — do not fall back to all-mpnet again, we already have those results.

PHASE 1 RERUN — Axis validation with Gemini.
Use gemini-embedding-exp-03-07 (or current available model). Important: Gemini Embedding 2 does NOT use task_type parameter. Instead, prepend task instructions to text.
Test TWO axis strategies:
1. expanded_words: same anchor words as v1 run
2. multi_anchor_sentences: use 5 positive anchor sentences about quality/honesty/helpfulness and 5 negative sentences, average each set, take difference
For BOTH strategies, rerun the same 61 statement pairs from phase1/test_statements.json.
KEY METRIC: sycophancy category accuracy. If >50% (vs 0% on all-mpnet), the model upgrade fixes the core problem.
Save to phase1_gemini/results_summary.md.

PHASE 2 RERUN — Preference prediction with Gemini.
Score the SAME 5000 HH-RLHF pairs from v1. Test TWO scoring modes:
1. response_only: embed just the final assistant turn (same as v1)
2. prompt_response: embed "User: {prompt}\nAssistant: {response}" as a single string
For prompt_response, also test with Gemini task instruction prepended: "Evaluate whether the following assistant response is helpful, honest, and safe: "
Compute same baselines and sentiment-discordant analysis as v1.
KEY METRICS: overall agreement, sentiment-discordant agreement (must beat 50%), and comparison to v1's 53.2%.
Save to phase2_gemini/results_summary.md.

PHASE 3 — If ANY Gemini Phase 2 variant exceeds 60% agreement AND beats both baselines, proceed to Phase 3 DPO training with rDPO (loss_type="robust" in TRL DPOConfig, label_smoothing calibrated to agreement rate). Use Colab T4 GPU with QLoRA + Unsloth.

PHASE 4 — Always. Update phase4/final_report.md with Gemini results appended.

After EVERY phase, update research_log.md. Read CODEX_GUIDANCE.md and PHASE2_DIAGNOSIS.md for detailed strategy context.

DONE WHEN: research_log.md has Gemini Phase 1 and Phase 2 entries with numerical results comparing to v1, AND phase4/final_report.md is updated.
```
