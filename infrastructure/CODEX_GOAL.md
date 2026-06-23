# Codex Goal (<2500 chars — paste this into the Codex goal field)

```
Execute the research pipeline in RESEARCH_PLAN.md. Complete Phases 0 through 2 minimum, Phase 3 if Phase 2 passes criteria, Phase 4 always.

PHASE 0 — Literature review. Write phase0/literature_review.md covering the 8 papers listed in the plan. Search for any prior work using embedding projection directly as reward signal (no classifier). Confirm the gap exists.

PHASE 1 — Axis validation. Use Gemini Embedding 2 API to embed anchors and test statements. If API fails, fall back to sentence-transformers/all-mpnet-base-v2 in Colab (no API needed). Produce phase1/results_summary.md with: anchor projection accuracy (% on correct side), statement pair accuracy on 50+ pairs, concept convergence cosine scores. Evaluate against decision criteria in the plan.

PHASE 2 — Preference prediction. Score 5000 Anthropic HH-RLHF pairs with embedding axis. Compute baselines (random, length, sentiment). Test at least 2 anchor strategies. Analyze failures. Produce phase2/results_summary.md with agreement rate, baseline comparison, failure breakdown. Evaluate against decision criteria.

PHASE 3 — Only if Phase 2 agreement >60% AND beats length+sentiment baselines. DPO fine-tune smallest available Gemma with QLoRA on Colab T4 using embedding-scored preference pairs. Evaluate with LLM judge and sycophancy checks. If GPU unavailable, skip and note in log.

PHASE 4 — Always. Write phase4/final_report.md compiling all results.

After EVERY phase, update research_log.md with: what was done, key numbers, interpretation, decision.

If any tool/API/dataset fails, use the fallback paths documented in RESEARCH_PLAN.md before stopping. Never stop without documenting why in research_log.md.

DONE WHEN: research_log.md has entries for all completed phases with numerical results and decisions, AND phase4/final_report.md exists summarizing findings.
```
