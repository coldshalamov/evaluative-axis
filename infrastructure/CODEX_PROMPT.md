# Codex Goal Prompt

## Context

You are executing a research project that tests whether the "good/bad" direction in sentence embedding space can serve as a direct reward signal for LLM alignment — replacing human labelers, trained reward models, and LLM-as-judge with a single geometric projection.

The full research plan is in `RESEARCH_PLAN.md` in this workspace. **Read it first.** It contains the hypothesis, related work, detailed experimental specs with code templates, success criteria, and decision points for each phase.

## Your Role

You are the engineer executing a research plan. You write code, run experiments, collect data, interpret results, and document everything. The plan has clear phases — execute them in order. Each phase has success criteria and decision points that tell you whether to continue, investigate further, or stop.

## Key Constraints

- **Local storage**: ~50MB. All heavy data (datasets, models, embedding vectors) stays in Colab or Google Drive. Only save results summaries, small JSON files, and markdown reports locally to this workspace.
- **Colab GPU budget**: ~22h/week on T4, not guaranteed. Use **CPU runtime** for Phases 0-2. GPU runtime ONLY for Phase 3 (DPO training). Never use GPU time for embedding computation or data processing.
- **Gemini Embedding 2 API**: Free tier, 10M tokens/min. Use for all embedding computation. API key should be stored as a Colab secret, never hardcoded.
- **Gemini Flash API**: Free tier. Use for LLM-as-judge evaluation in Phase 3.

## Tools Available

- **Web search**: For Phase 0 literature review. Find papers, read abstracts, identify related work.
- **Colab MCP**: Create notebooks, write cells, execute code, read output. If MCP isn't configured, create `.ipynb` files locally and note they need to be uploaded to Colab.
- **Local file operations**: Write results, update research log, save analysis.
- **Python (local)**: For analysis scripts that don't need GPU or large data.

## Execution Order

### Phase 0: Literature Review (2-4 hours, local + web)
- Search for and read the papers listed in the plan
- Search for any prior work doing exactly this (embedding projection as direct reward)
- Write `phase0/literature_review.md` and `phase0/related_papers.md`
- Update `research_log.md`
- **Check decision point before proceeding**

### Phase 1: Axis Validation (4-6 hours, Colab CPU + Gemini API)
- Create and run the axis validation notebook
- Test anchor embeddings, statement pair scoring, concept convergence
- Save results to `phase1/`
- Update `research_log.md`
- **Check decision point**: statement accuracy and concept convergence thresholds

### Phase 2: Preference Prediction (6-8 hours, Colab CPU + Gemini API)
- Create and run the preference prediction notebook
- Compute baselines (random, length, sentiment)
- Run main experiment on 5000 HH-RLHF pairs
- Test multiple anchor strategies
- Analyze failures
- Save results to `phase2/`
- Update `research_log.md`
- **Check decision point**: agreement rate vs baselines

### Phase 3: DPO Fine-tuning (4-8 hours, Colab GPU)
- **Only proceed if Phase 2 agreement > 60% AND beats baselines**
- Do ALL prep on CPU first (data processing, embedding, pair generation)
- Switch to GPU ONLY for actual DPO training
- Evaluate: embedding score, LLM judge, sycophancy checks
- Save results to `phase3/`
- Update `research_log.md`

### Phase 4: Analysis & Report (4-6 hours, local or CPU)
- Create visualizations
- Write `phase4/final_report.md`
- Final update to `research_log.md`

## How to Handle Ambiguous Results

If results fall in the "investigate" range at any decision point:
1. Document exactly what you observed
2. Try the suggested alternatives (different anchors, different embedding model)
3. Document those results too
4. Make a judgment call with clear reasoning in `research_log.md`
5. Proceed if the adjusted experiment shows promise; write up the negative result if not

## Research Log

After each phase, update `research_log.md` with:
- What was done
- Key results (numbers)
- Your interpretation
- Decision (continue/investigate/stop)
- Next steps

This log is how the project owner (Robin) tracks progress. Be specific with numbers and clear about what they mean.

## Start

1. Read `RESEARCH_PLAN.md`
2. Create the directory structure (phase0/, phase1/, phase2/, phase3/, phase4/)
3. Begin Phase 0: Literature Review
