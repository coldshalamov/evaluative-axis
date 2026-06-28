# AGENTS.md

## Read First — In This Order

1. `methodology/RESEARCH_CONTEXT.md` — Environment, method, pitfalls, template
2. `methodology/RESEARCH_DIRECTIONS.md` — All open questions, prioritized
3. `methodology/EXPERIMENT_SPECS.md` — Detailed experiment designs ready to run
4. `paper/draft.md` — The paper (read intro + results sections)

## Scope

This repo is a research workspace for testing whether evaluative embedding
geometry can act as a cheap answer-selection or training signal for LLM
alignment.

## Critical Rules

1. **Always test on the BALANCED battery.** The original 50-case battery is
   64% firmness-biased. Use original 50 + warmth 20 (in
   `notes/research_cycles/battery_rebalancing/warmth_cases.jsonl`). Report
   results on BOTH splits separately plus combined.

2. **Always test on ALL THREE local models.** Results on one model mean
   nothing. The three models are: snowflake/snowflake-arctic-embed-m,
   BAAI/bge-m3, nomic-ai/nomic-embed-text-v1.5.

3. **Never average axis vectors.** To combine multiple axes, score each
   independently and SUM the scores or use MAJORITY VOTE. Averaging
   recreates the parent direction and kills signal.

4. **Always use User/Assistant framing.** Embed responses as
   `f"User: {prompt}\nAssistant: {response}"`, not raw response text.

5. **Use the project venv.** Python path:
   `C:\Users\93rob\.cache\codex-embedding-venv\Scripts\python.exe`

6. **Save results as JSON.** Every experiment saves structured results to
   `notes/research_cycles/<experiment_name>/`. Print human-readable output
   AND save machine-readable JSON.

7. **Negative results are valuable.** Report them honestly. Do not spin
   a failure as a partial success.

8. **Don't theorize — experiment.** Theory has outpaced evidence. The only
   way to resolve open questions is by running tests.

9. **Budget is zero.** Do not propose experiments requiring paid APIs (except
   Google free tier), GPU rental, or human annotators.

10. **Check `methodology/RESEARCH_DIRECTIONS.md` for status.** If an
    experiment is marked DONE, don't redo it. If it's NOT STARTED, it's
    fair game.

## Repo Map

```
paper/draft.md                    Main paper
methodology/
  RESEARCH_CONTEXT.md             Environment, pitfalls, template (READ FIRST)
  RESEARCH_DIRECTIONS.md          All open questions with priorities
  EXPERIMENT_SPECS.md             Detailed experiment designs
  DECISIVE_EVIDENCE_PLAN.md       What counts as proof
  RIGOR_GUARDRAILS.md             Experimental discipline rules
scripts/
  run_cycle001_intervention.py    AXES dict (5 ML-jargon axes), imported by many scripts
  run_battery_rebalance_test.py   Battery rebalancing test
  run_score_magnitude_analysis.py Score magnitude and margin analysis
  run_osgood_dimensions.py        Osgood EPA test
  run_bootstrap_ci.py             Bootstrap confidence intervals
  run_category_breakdown.py       Per-category accuracy breakdown
  run_baselines_comparison.py     Baselines comparison
  run_axis_correlation.py         Inter-axis correlation
notes/research_cycles/
  cycle_002_potential_shaping/    Battery (50 cases) + early results
  battery_rebalancing/            Warmth cases (20) + rebalance results
  optimal_basis/                  Basis set search results
  osgood_dimensions/              Osgood EPA test results
```

## What NOT To Do

- Do not propose hiring people or paying for services
- Do not suggest arXiv (user was rejected for endorsement)
- Do not suggest cold-emailing researchers (already tried, failed)
- Do not use extended analogies or unexplained jargon
- Do not dump percentages without explaining what they mean
- Do not declare "Hard/Soft" or Osgood Potency as a positive finding
  (it's a battery artifact — see pitfall #1 in RESEARCH_CONTEXT.md)
- Do not run experiments only on the original 50-case battery
- Do not combine axis vectors by averaging
