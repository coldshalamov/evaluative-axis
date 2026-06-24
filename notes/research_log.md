# Research Log: Embedding Geometry as Direct Reward Signal

**Project started**: June 21, 2026  
**Researcher**: Robin Gattis  
**Research plan**: See `RESEARCH_PLAN.md`

---

## Prerequisites Checklist

- [ ] Gemini API key obtained (free from https://aistudio.google.com/apikey)
- [ ] Colab MCP configured (https://github.com/googlecolab/colab-mcp) OR plan to create notebooks manually
- [ ] Google Drive has space for notebooks and small result files

---

## Project Overview

**Hypothesis**: The "good/bad" direction in sentence embedding space can serve as a direct reward signal for LLM preference fine-tuning, without trained reward models or human labels.

**Key insight**: Human evaluative judgment (good/bad) is already compressed into embedding model geometry from pretraining on human text. This is the cheapest possible reward signal — a single dot product.

**What makes this different from prior work**: Sun et al. (2025) showed embeddings work as INPUT to trained reward classifiers. We skip the classifier and use raw geometric projection. No training, no labels.

**Primary embedding model**: Gemini Embedding 2 (3072-dim, top MTEB, free API)

---

## Log Entries

*(Codex: append entries here after each phase)*

## June 21, 2026 — Phase 0: Literature Foundation

### What was done

Verified and rewrote the literature/gap review using primary paper pages where
available. Covered the 8 required sources: InstructGPT/RLHF, DPO,
Constitutional AI/RLAIF, Reusing Embeddings, Value Entanglement, Latent
Affective Representations, The Measurement of Meaning, and Representation
Engineering. Also searched adjacent prior art for direct embedding projection
as a reward signal.

### Key results

Required papers covered: 8/8. Gap-search queries run: 6. Exact method found:
0. Close/adjacent methods found: 4 important items. Legend is the closest prior
because it discovers a safety direction and projects response pairs along that
direction to annotate preference margins. Virtue Semantics uses moral-goodness
embedding projection as an analysis metric. Intrinsic Guardrails uses a
Semantic Valence Vector for activation-space intervention. Reusing Embeddings
uses embeddings as inputs to trained reward models.

### Interpretation

The gap exists, but it is narrower than the original novelty framing. The
project should not claim that projection-based alignment signals are untouched.
The live question is whether a broad external sentence-embedding good/bad axis
is strong enough as a direct, no-classifier preference generator.

### Decision

Continue to Phase 1. The exact method was not found, and no paper was found
showing this exact method fails.

### Next steps

Run Phase 1 axis validation. Gemini API credentials are not present in the local
environment, and Colab MCP connection returned `false`, so use the documented
sentence-transformers fallback and record that fallback in Phase 1.

## June 21, 2026 - Phase 1: Axis Validation

### What was done
Ran anchor projection, 61 statement-pair tests, antonym concept convergence, and statement-level concept convergence using `sentence-transformers:sentence-transformers/all-mpnet-base-v2`. Gemini was not used because credentials were unavailable and Colab MCP returned false.

### Key results
Anchor projection accuracy: 100.0%. Statement pair accuracy: 55.7%. Mean antonym concept cosine: 0.3245. Mean statement-level concept cosine: 0.4211. Incorrect statement pairs: 27.

### Interpretation
The broad evaluative axis is usable for controlled statement pairs if statement accuracy clears the Phase 1 threshold. Any sycophancy or mixed-case misses are carried into Phase 2 as risk signals.

### Decision
investigate_before_phase2.

### Next steps
Proceed to Phase 2 preference prediction on HH-RLHF because this run's goal
requires Phase 2 minimum, but treat Phase 1 as a warning signal rather than a
pass.

## June 21, 2026 - Phase 2: Preference Prediction

### What was done
Scored 5000 Anthropic HH-RLHF train pairs with four anchor strategies, using final assistant turns only. Computed random, length, and VADER sentiment baselines plus sentiment-discordant and confidence-filtered analyses.

### Key results
Best strategy: expanded_words. Agreement: 53.2%. Length baseline: 43.2%. Sentiment baseline: 48.3%. Sentiment-discordant agreement: 43.8% over 2452 pairs.

### Interpretation
The direct projection signal is interesting only if it clears 60% and beats trivial baselines. Otherwise it is too noisy as a standalone preference source, though it may still be useful as a feature or confidence signal.

### Decision
do_not_proceed_to_phase3.

### Next steps
Run Phase 3 only if the gate passed and GPU is available; otherwise document the skip and compile Phase 4.

## June 21, 2026 - Phase 3: DPO Fine-Tuning Gate

### What was done
Evaluated whether Phase 3 should run based on Phase 2 metrics and local GPU availability.

### Key results
Status: skipped_phase2_gate_failed. Phase 2 best agreement: 53.2%. Length baseline: 43.2%. Sentiment baseline: 48.3%. CUDA available: False.

### Interpretation
Phase 2 did not meet the required gate: agreement >60% and better than both length and sentiment baselines.

### Decision
Skip Phase 3 for this run.

### Next steps
Compile Phase 4 final report with Phase 3 marked as skipped.

## June 21, 2026 - Phase 4: Final Report

### What was done
Compiled Phase 0 through Phase 3 findings into a final report and generated Phase 1/2 figures.

### Key results
Final report exists at phase4/final_report.md. Figures generated: 5. Phase 2 best agreement: 53.2%. Phase 3 status: skipped_phase2_gate_failed.

### Interpretation
The final report frames the outcome as an empirical test of a minimal projection baseline with explicit limitations around fallback model use, context truncation, and missing DPO evaluation.

### Decision
Complete for this run.

### Next steps
Rerun with Gemini/Colab if the goal is to compare against the planned primary embedding model.

## June 21, 2026 - Gemini Rerun Prerequisite Check

### What was done
Opened a fresh Google Colab notebook through Chrome and executed a Gemini key
probe. The probe checked local/runtime environment variables
`GOOGLE_API_KEY` and `GEMINI_API_KEY`, then checked Colab Secrets via
`google.colab.userdata` for the same key names. Also checked the local Codex
environment for those variables.

### Key results
Gemini API key found: 0. Colab secret `GOOGLE_API_KEY`: missing. Local
environment `GOOGLE_API_KEY` / `GEMINI_API_KEY`: missing. Gemini model probe
could not run. A handoff note was written to
`phase1_gemini/no_key_available.md`.

### Interpretation
The v2 Gemini rerun is blocked at the explicit prerequisite. Per
`CODEX_GOAL_V2.md`, this run must not fall back to all-mpnet again because the
all-mpnet results already exist.

### Decision
Stop the Gemini rerun until a Gemini API key is available.

### Next steps
Add a Gemini key from https://aistudio.google.com/apikey as a Colab Secret
named `GOOGLE_API_KEY` or set `GOOGLE_API_KEY` / `GEMINI_API_KEY` in the local
environment, then rerun the v2 goal.

## June 21, 2026 - Gemini Rerun Prerequisite Check 2

### What was done
Reconnected to the existing Colab notebook through Chrome and inspected the
current Colab Secrets panel without reading any secret values. Also rechecked
the local environment for `GOOGLE_API_KEY` and `GEMINI_API_KEY`. Added a
ready-to-run script, `scripts/run_gemini_rerun.py`, that implements the v2
Gemini rerun with `gemini-embedding-2`, prompt+response scoring, two anchor
strategies, statistical tests, v1 comparisons, log updates, and final-report
addendum generation.

### Key results
Colab Secrets panel status: "No secrets saved." Local key variables found: 0.
`scripts/run_gemini_rerun.py` syntax check: pass. No-key dry run: exited with
`NO_GEMINI_KEY_FOUND` and did not fall back to all-mpnet.

### Interpretation
The experiment is still blocked at the Gemini key prerequisite, but the next
run is now mechanically prepared. Once a key exists, the runner can execute the
Gemini Phase 1 and Phase 2 rerun without additional setup work.

### Decision
Keep the Gemini rerun goal active; do not mark complete and do not fall back.

### Next steps
Add `GOOGLE_API_KEY` or `GEMINI_API_KEY` in Colab Secrets or the local
environment, then run `python scripts/run_gemini_rerun.py --sample-size 5000`.

## June 21, 2026 - Gemini Rerun Prerequisite Check 3

### What was done
Reconnected to the Colab notebook through Chrome, opened the Colab Secrets
panel, and used the built-in `Gemini API keys` helper. The helper was inspected
without reading or copying any secret value.

### Key results
Local key variables found: 0. Colab Secrets still showed no saved secrets.
The Colab helper reported `No keys found` and stated that no Gemini API key has
been created yet in Google AI Studio for this account.

### Interpretation
The remaining blocker is not Colab connectivity or script readiness. The
connected account lacks a Gemini API key. Creating the first key is a persistent
access-credential action, so Codex should not click through it without
explicit action-time confirmation.

### Decision
Blocked on user confirmation / external credential creation.

### Next steps
If Robin wants Codex to continue in-browser, explicitly confirm: "Create the
first Gemini API key in Google AI Studio and import it into Colab Secrets."
Otherwise Robin can create the key manually and save it as `GOOGLE_API_KEY`,
then rerun `python scripts/run_gemini_rerun.py --sample-size 5000`.

## June 22, 2026 - Gemini API Key Permission Probe

### What was done
Stored the user-provided Google API key in `.env.local`, added `.env` /
`.env.*` to `.gitignore`, patched `scripts/run_gemini_rerun.py` to load local
env files automatically, and ran sanitized Gemini API probes without printing
the key.

### Key results
Key source: local environment via `.env.local`. Key present: yes. API access:
blocked. `models:list`, `generateContent`, and `embedContent` all returned
HTTP 403 `PERMISSION_DENIED` for `generativelanguage.googleapis.com` methods.
The rerun script syntax check passed.

### Interpretation
The blocker changed from "no key exists" to "the provided key is not permitted
to call the Gemini Generative Language API." This is likely an API restriction,
application restriction, or a key/project without the Generative Language API
enabled.

### Decision
Do not run the Gemini rerun yet; the key cannot access the required Gemini
embedding endpoint.

### Next steps
Use a Google AI Studio Gemini API key, or update this key/project so
`generativelanguage.googleapis.com` is enabled and allowed by the key
restrictions. Then rerun `python scripts/run_gemini_rerun.py --sample-size
5000`.

## June 22, 2026 - Gemini Phase 1: Axis Validation

### What was done
Reran the 61 controlled Phase 1 statement pairs with Gemini embeddings using expanded word anchors and multi-anchor sentence anchors.

### Key results
Best strategy: multi_anchor_sentences. Statement accuracy: 70.5%. Sycophancy accuracy: 0.0% (v1: 0.0%).

### Interpretation
Sycophancy accuracy is the key diagnostic for whether Gemini separates substantive quality from positive tone better than all-mpnet.

### Decision
Continue to Gemini Phase 2.

### Next steps
Run HH-RLHF prompt+response scoring and compare against v1 and baselines.

## June 22, 2026 - Gemini Rerun Final State

### What was done
Verified the final Gemini rerun state after Phase 2 quota failure. Confirmed
`scripts/run_gemini_rerun.py` compiles, `phase2_gemini/results_summary.md`
exists, `phase2_gemini/quota_blocked.md` exists, and `phase4/final_report.md`
contains the Gemini rerun addendum.

### Key results
Gemini Phase 1 completed: `gemini-embedding-2`, 3072 dimensions,
`multi_anchor_sentences` best strategy, 70.5% statement-pair accuracy, 0.0%
sycophancy accuracy. Gemini Phase 2 completed result sets: 0. Full 5000-pair
Gemini agreement: not computed. Usable key/project: HTTP 429 quota exhausted.
Second existing key: HTTP 403 leaked-key rejection. Fresh key creation:
blocked by AI Studio as suspicious.

### Interpretation
The stronger embedding model improved controlled axis validation, but the
central Phase 2 preference-prediction hypothesis remains unresolved for Gemini
because of external API quota. This is not a Colab compute problem.

### Decision
Stop the Gemini rerun here without Phase 3. Phase 3 is not gated in because
Gemini Phase 2 did not produce a complete agreement rate above 60% and above
the baselines.

### Next steps
After adding a non-leaked key/project with enough quota, rerun:
`& '.\.tmp\phase-env\Scripts\python.exe' scripts\run_gemini_rerun.py --sample-size 5000 --skip-phase1 --max-workers 1 --batch-size 50`.

## June 22, 2026 - Gemini Phase 2: Preference Prediction Blocked

### What was done
Attempted the Gemini Phase 2 HH-RLHF rerun after obtaining a working Google AI
Studio key from the browser. Patched the runner to use Gemini
`batchEmbedContents`, adaptive batch splitting, local `.env.local` loading,
key-rotation support, `--skip-phase1`, and Phase 2 embedding checkpoints.
Tried the existing second AI Studio key and attempted to create a fresh key in
AI Studio.

### Key results
Gemini Phase 1 completed with `gemini-embedding-2`: best strategy
`multi_anchor_sentences`, 70.5% statement-pair accuracy, 0.0% sycophancy
accuracy, 3072-dimensional embeddings. Gemini Phase 2 complete HH-RLHF result
sets: 0. Agreement rate: not computed. The usable key/project returned HTTP
429 quota exhaustion, including on a later two-text smoke probe. The second
existing key returned HTTP 403 because Google reports it as leaked. AI Studio
fresh-key creation returned: `Failed to generate API key, The request is
suspicious. Please try again.`

### Interpretation
The Gemini Phase 1 result supports the hypothesis that a stronger embedding
model improves the controlled axis validation signal. The Phase 2 question is
still unresolved because the blocker is external Gemini API quota, not the
local code or Colab compute. Colab would help for Phase 3 GPU fine-tuning, but
it does not bypass Gemini API quota for Phase 2 embeddings.

### Decision
Do not proceed to Phase 3 from Gemini results. Treat Gemini Phase 2 as blocked
until a non-leaked key/project with sufficient Gemini embedding quota is
available. Keep the all-mpnet Phase 2 result as the only completed 5000-pair
preference-prediction result.

### Next steps
Enable billing/prepay or manually create a fresh non-leaked Google AI Studio
key, then rerun:
`& '.\.tmp\phase-env\Scripts\python.exe' scripts\run_gemini_rerun.py --sample-size 5000 --skip-phase1 --max-workers 1 --batch-size 50`.

## June 22, 2026 - Gemini Phase 1: Axis Validation

### What was done
Reran the 61 controlled Phase 1 statement pairs with Gemini embeddings using expanded word anchors and multi-anchor sentence anchors.

### Key results
Best strategy: multi_anchor_sentences. Statement accuracy: 70.5%. Sycophancy accuracy: 0.0% (v1: 0.0%).

### Interpretation
Sycophancy accuracy is the key diagnostic for whether Gemini separates substantive quality from positive tone better than all-mpnet.

### Decision
Continue to Gemini Phase 2.

### Next steps
Run HH-RLHF prompt+response scoring and compare against v1 and baselines.

## June 22, 2026 - Gemini Phase 1: Axis Validation

### What was done
Reran the 61 controlled Phase 1 statement pairs with Gemini embeddings using expanded word anchors and multi-anchor sentence anchors.

### Key results
Best strategy: multi_anchor_sentences. Statement accuracy: 70.5%. Sycophancy accuracy: 0.0% (v1: 0.0%).

### Interpretation
Sycophancy accuracy is the key diagnostic for whether Gemini separates substantive quality from positive tone better than all-mpnet.

### Decision
Continue to Gemini Phase 2.

### Next steps
Run HH-RLHF prompt+response scoring and compare against v1 and baselines.

## June 22, 2026 - Gemini Rerun EOF Decision

### What was done
Final verification after the Gemini API quota block.

### Key results
Gemini Phase 1 completed with `gemini-embedding-2`: 70.5% statement-pair
accuracy, 0.0% sycophancy accuracy, and 3072-dimensional embeddings. Gemini
Phase 2 completed result sets: 0. Full 5000-pair Gemini agreement: not
computed. Usable key/project: HTTP 429 quota exhausted. Second existing key:
HTTP 403 leaked-key rejection. Fresh key creation: blocked by AI Studio as
suspicious.

### Interpretation
The stronger embedding model improved controlled axis validation, but Gemini
Phase 2 is externally blocked by API quota/key availability. Colab does not
solve this specific blocker because the bottleneck is API quota, not compute.

### Decision
Do not proceed to Phase 3 from Gemini results. The only completed 5000-pair
Phase 2 result remains the all-mpnet run at 53.2% agreement.

### Next steps
Add a non-leaked key/project with enough quota, then rerun:
`& '.\.tmp\phase-env\Scripts\python.exe' scripts\run_gemini_rerun.py --sample-size 5000 --skip-phase1 --max-workers 1 --batch-size 50`.

## June 22, 2026 - Open-Source Embedding Pilot

### What was done
Ran an open-source embedding pilot as a fallback because Gemini Embedding 2 API
quota remained blocked and the Colab MCP connector returned false.

### Key results
Model: BAAI/bge-large-en-v1.5. Sample size: 50. Best variant: atomic_evaluation__anti_sycophancy_quality. Agreement: 61.0%. Sentiment-discordant agreement: 59.6%. Length baseline: 42.0%. Sentiment baseline: 46.0%.

### Interpretation
This tests whether a stronger open-source embedding model and decomposition-framed text improve the projection signal while Gemini API quota is unavailable.

### Decision
Use the pilot directionally only; run a larger GPU/Colab sample when Colab execution or Gemini quota is available.

### Next steps
If the best variant beats 55% and baselines, rerun at 1000+ pairs or move to Colab GPU for a larger embedding model.

## June 22, 2026 - Colab Connectivity and BGE-Small Pilot

### What was done
Used Chrome browser control to operate the existing Colab scratchpad directly.
Connected the notebook runtime, ran stdout and GPU probes, installed
`sentence-transformers`, `datasets`, and `vaderSentiment`, and tested whether
Colab could be used for the next embedding runs. Also ran a stable local
300-pair BGE-small pilot after the long Colab-cell route proved unreliable.

### Key results
Colab runtime connected: yes. Short-cell execution: yes. GPU available: no
(`torch 2.11.0+cpu`, `torch.cuda.is_available() == False`, no `nvidia-smi`).
Formal `colab_mcp.open_colab_browser_connection`: timed out after the runtime
was connected. Gemini probe after the user's stated reset: still HTTP 429 on
three consecutive minute-spaced retries before one embedding completed.
BGE-small 300-pair pilot best variant:
`atomic_evaluation__anti_sycophancy_quality`, 59.2% agreement, 51.6%
sentiment-discordant agreement over 161 pairs, length baseline 43.3%,
sentiment baseline 44.5%, z vs random 3.18, p = 0.0015.

### Interpretation
Colab itself is usable through browser control for short setup/probe cells, but
the formal MCP websocket bridge and long scratchpad-cell pasting are not
reliable enough for full experiments. The BGE-small result strengthens the
case that decomposition framing helps: response-only variants were below
random, while `atomic_evaluation` with the anti-sycophancy axis nearly reached
the Phase 3 gate.

### Decision
Do not proceed to Phase 3. Treat the BGE-small pilot as promising but not
gated in; it is below 60% and is only 300 pairs. Use Colab only for short cells
until a GPU runtime or real notebook-file workflow is available.

### Next steps
Run a larger confirmation with `atomic_evaluation__anti_sycophancy_quality`
when either Gemini quota works or Colab GPU/notebook-file execution is
available. The highest-value Gemini command remains:
`& '.\.tmp\phase-env\Scripts\python.exe' scripts\run_gemini_rerun.py --sample-size 5000 --skip-phase1 --max-workers 1 --batch-size 50 --scoring-modes prompt_response_instruction,atomic_evaluation`.

## June 22, 2026 - Open-Source Embedding Pilot

### What was done
Ran an open-source embedding pilot as a fallback because Gemini Embedding 2 API
quota remained blocked and the Colab MCP connector returned false.

### Key results
Model: BAAI/bge-large-en-v1.5. Sample size: 200. Best variant: atomic_evaluation__anti_sycophancy_quality. Agreement: 52.2%. Sentiment-discordant agreement: 47.1%. Length baseline: 43.5%. Sentiment baseline: 46.5%.

### Interpretation
This tests whether a stronger open-source embedding model and decomposition-framed text improve the projection signal while Gemini API quota is unavailable.

### Decision
Use the pilot directionally only; run a larger GPU/Colab sample when Colab execution or Gemini quota is available.

### Next steps
If the best variant beats 55% and baselines, rerun at 1000+ pairs or move to Colab GPU for a larger embedding model.

## June 22, 2026 - Open-Source Embedding Pilot

### What was done
Ran an open-source embedding pilot as a fallback because Gemini Embedding 2 API
quota remained blocked and the Colab MCP connector returned false.

### Key results
Model: BAAI/bge-m3. Sample size: 20. Best variant: atomic_evaluation__anti_sycophancy_quality. Agreement: 35.0%. Sentiment-discordant agreement: 25.0%. Length baseline: 35.0%. Sentiment baseline: 55.0%.

### Interpretation
This tests whether a stronger open-source embedding model and decomposition-framed text improve the projection signal while Gemini API quota is unavailable.

### Decision
Use the pilot directionally only; run a larger GPU/Colab sample when Colab execution or Gemini quota is available.

### Next steps
If the best variant beats 55% and baselines, rerun at 1000+ pairs or move to Colab GPU for a larger embedding model.

## June 22, 2026 - Open-Source Embedding Pilot

### What was done
Ran an open-source embedding pilot as a fallback because Gemini Embedding 2 API
quota remained blocked and the Colab MCP connector returned false.

### Key results
Model: BAAI/bge-small-en-v1.5. Sample size: 300. Best variant: atomic_evaluation__anti_sycophancy_quality. Agreement: 59.2%. Sentiment-discordant agreement: 51.6%. Length baseline: 43.3%. Sentiment baseline: 44.5%.

### Interpretation
This tests whether a stronger open-source embedding model and decomposition-framed text improve the projection signal while Gemini API quota is unavailable.

### Decision
Use the pilot directionally only; run a larger GPU/Colab sample when Colab execution or Gemini quota is available.

### Next steps
If the best variant beats 55% and baselines, rerun at 1000+ pairs or move to Colab GPU for a larger embedding model.

## June 22, 2026 - Phase 5 Verification Probe

### What was done
Ran a compact verification suite with `BAAI/bge-small-en-v1.5` to separate
three questions: whether the good/bad axis binds context around locally bad
phrases, whether prompt+response/process framing improves HH-RLHF pair ranking,
and which high-confidence HH disagreements look like surface-feature failures.
After the first pass showed broad axes failing ordinary context, added a
task-specific `contextual_harm_reduction` axis and reran the same 56 context
pairs and 500 HH-RLHF pairs.

### Key results
- Context polarity, best non-oracle: `bare_context__contextual_harm_reduction`
  at 64.3% (z=2.14, p=0.0325).
- Context polarity, oracle decomposition upper bound:
  `oracle_decomposition__multi_anchor_sentences` at 100.0%.
- HH sample 500; best `prompt_response__contextual_harm_reduction` at 55.8%
  (z=2.59, p=0.00949); length baseline 41.3%; sentiment baseline 46.9%.
- Sentiment-discordant HH subset for best variant: 57.0% over 256 pairs.

### Interpretation
The context-polarity test asks a cleaner question than raw HH agreement: can
the axis score "refused lying under oath" above "encouraged lying under oath"
even though both contain the same bad local phrase? The result says generic
good/bad axes are too blunt, but a more specific harm-reduction axis can recover
real signal. The 100% oracle-decomposition score is not an independent
evaluator; it shows the embedding axis can read explicit good/bad decomposition
when the relevant atoms are written into the text.

### Decision
Do not treat one global embedding projection as a production reward. Continue
the research using a small basis of aspect-specific evaluative axes and process
or outcome text, because this is where the signal became cleaner.

### Next steps
Run the same Phase 5 context-polarity set on Gemini/frontier embeddings when
quota is available. If the frontier model raises non-oracle context accuracy
substantially above 64.3%, then rerun HH-RLHF at 1000-5000 pairs with
aspect-specific axes. Also test scoring generated decomposition/outcome reports
instead of scoring final answers directly.

## June 22, 2026 - Phase 6: Multi-Sensor Evaluative Axis Probe

### What was done
Ran a frozen multi-axis embedding evaluator across multiple imperfect preference
artifacts instead of treating HH-RLHF as authoritative. Used
`BAAI/bge-small-en-v1.5` with eight predeclared axes: broad good/bad, harm
reduction, truth correction, calibration, usefulness, non-sycophancy, risk
disclosure, and agency respect. No weights were fit to any dataset. Scored 300
samples each from Anthropic HH, PKU-SafeRLHF, and Stanford SHP; PKU contributes
both `better` and `safer` label views.

### Key results
- `hh_chosen`: best axis `risk_disclosure` at 55.0% overlap; length baseline
  43.3%; sentiment baseline 44.5%.
- `pku_better`: best axis `harm_reduction` at 52.0%; length baseline 56.8%;
  sentiment baseline 50.3%.
- `pku_safer`: best axis `harm_reduction` at 54.3%; length baseline 52.8%;
  sentiment baseline 46.3%.
- `shp_reddit`: best axis `agency_respect` at 55.3%; length baseline 70.3%;
  sentiment baseline 54.5%.
- Equal-weight aggregates underperformed individual axes, supporting an
  evaluative-basis framing rather than one universal scalar.

### Interpretation
The Phase 6 result supports the user's critique: dataset agreement is not
goodness. Different artifacts measure different things. SHP is strongly
length/social-signal shaped, PKU better and safer labels diverge, and HH's
older preference labels overlap most with risk-disclosure under this model.
The useful object is not a single "accuracy" number but a map of cheap
embedding sensors against expensive/social preference artifacts.

### Decision
Continue the research as an embedding-evaluator flywheel, not as an RLHF
replacement claim. The idea is promising because even crude, cheap axes produce
non-random overlap and useful disagreement audits under poor conditions.

### Next steps
Run an intervention: generate multiple candidate answers per prompt, rerank
with direct embedding axes and embedding-scored LLM critiques, then judge
whether selected answers improve against random, length, and standard LLM-judge
baselines. This tests the actual claim: whether the signal improves output
selection or training data per dollar/token/hour.

## June 22, 2026 - Phase 5: Manual HH Disagreement Adjudication

### What was done
Reviewed a manual grading pass over the 30 strongest Phase 5 HH-RLHF
disagreements, where the best embedding variant
`prompt_response__contextual_harm_reduction` preferred the HH-rejected response
with the largest margins.

### Key results
- Raw Phase 5 HH agreement was 279/500 = 55.8%.
- The top-30 disagreement audit judged 14/30 cases (46.7%) as embedding-right /
  HH-likely-mislabeled, 10/30 (33.3%) as HH-right / genuine embedding miss, and
  6/30 (20.0%) as ties or both-bad cases.
- If the 46.7% bad-label rate generalized to all 221 raw disagreements,
  corrected agreement would be 76.4%. With a 50% discount for selection bias it
  would be 66.1%; with a 70% discount it would be 62.0%.
- Representative HH-label problems included fabricated AI persona claims,
  doxxing-adjacent compliance, misinformation, racist-story compliance, slur
  lists, and empty non-answers.

### Interpretation
The low raw HH agreement is materially influenced by HH label noise. The audit
does not prove a final corrected accuracy because the cases were selected from
the strongest disagreements, but it directly refutes the interpretation that
all HH disagreements are embedding failures.

### Decision
Treat HH-RLHF as one noisy preference sensor, not as ground truth. Future
preference-prediction results should report raw agreement, baseline comparison,
and disagreement adjudication/noise analysis.

### Next steps
Run a blind adjudication of a larger stratified disagreement sample, ideally
including high-confidence, medium-confidence, and near-tie disagreements. Use
that to estimate label-noise-adjusted agreement with uncertainty.

## June 22, 2026 - Phase 6: Gemini Partial Multi-Sensor Probe

### What was done
Added Gemini backend support to `scripts/run_phase6_multi_sensor.py`, including
cached anchor embeddings, resumable candidate embeddings, and a
`--score-partial-cache` mode. Attempted a 1000-sample Gemini Phase 6 run and a
200-sample bounded run with `gemini-embedding-001`.

### Key results
- `gemini-embedding-001` probe succeeded with 3072-dimensional normalized
  vectors.
- The 1000-sample run hit repeated HTTP 429 quota throttling after 250/8000
  candidate texts.
- The 200-sample run hit repeated HTTP 429 throttling after 550/1600 candidate
  texts.
- Partial cache scoring produced 275 complete pairs: 200 `hh_chosen` pairs and
  75 `pku_better` pairs.
- On the 200 HH pairs, best Gemini axis was `non_sycophancy` at 50.0%;
  length baseline 43.5%; sentiment baseline 46.5%. Several broad axes were
  anti-correlated with HH on this slice.
- On the 75 PKU-better pairs, best Gemini axis was `agency_respect` at 53.3%;
  length baseline 54.0%; sentiment baseline 46.7%.

### Interpretation
This is a quota-limited partial result, not the decisive Gemini Embedding 2
experiment. It shows that `gemini-embedding-001` with the frozen broad Phase 6
axes does not automatically improve HH overlap, and it reinforces the larger
lesson: dataset overlap is not the target by itself. The manual HH audit makes
clear that low HH overlap can reflect bad labels or mismatched objectives.

### Decision
Do not proceed to Phase 3 from this partial Gemini result. Keep the completed
BGE-small Phase 6 as the broader multi-sensor baseline and treat the Gemini
partial as a quota/protocol probe.

### Next steps
The next decisive experiment should be an intervention, not another raw
dataset-overlap test: generate several candidate answers per prompt, rerank
with embedding axes and embedding-scored LLM critiques, and blind-judge whether
the selected answer improves over random, length, sentiment, and standard
LLM-judge baselines.

## June 22, 2026 - Phase 5: Full HH Disagreement Grading

### What was done
Reviewed a full grading pass over all 231 HH-RLHF disagreement cases from a
500-pair sample where the embedding axis preferred the HH-rejected response.
Grades were `EMBEDDING_RIGHT`, `HH_RIGHT`, or `EXCLUDE` for cases where both
responses were bad, trivial, marginal, or not useful for training.

### Key results
- Raw HH agreement in this grading file: 269/500 = 53.8%.
- All raw disagreements: 231/500 = 46.2%.
- `EMBEDDING_RIGHT`: 65/231 = 28.1% of disagreements.
- `HH_RIGHT`: 44/231 = 19.0% of disagreements.
- `EXCLUDE`: 122/231 = 52.8% of disagreements.
- Among gradeable disagreements, the embedding was judged better in
  65/(65+44) = 59.6%.
- Excluding the 122 no-signal/both-bad disagreements gives corrected gradeable
  agreement of (269+65)/(269+65+44) = 334/378 = 88.4%.
- Sensitivity estimates: 83.3% if 30% of embedding-right calls are wrong;
  79.9% if 50% are wrong.

### Interpretation
This is the strongest evidence so far that raw HH agreement is a misleading
score for this idea. Most apparent embedding failures against HH are either
cases where HH appears mislabeled, or cases where both responses are bad/trivial
and an automatic reward pipeline should filter or regenerate rather than force
a pairwise preference.

This makes the approach look much more valuable for automatic training
pipelines than the raw 53.8% number suggested. The likely role is not "imitate
HH," but score/filter/rerank candidates and detect bad preference pairs.

### Decision
Promote the full disagreement grading to a central result, with caveats. Do not
claim final proof of training improvement yet, because the grading was not blind
human adjudication and the agreement cases were assumed correct. But the
evidence now strongly favors moving to an intervention test.

### Next steps
Run a no-training intervention benchmark comparing embedding-axis selection,
embedding-scored critique selection, random, length, sentiment, and vanilla
LLM-as-judge. The critical question is whether embedding-selected outputs win
under blind review and whether they do so more cheaply or more robustly than
vanilla LLM judges.

## June 23, 2026 - Concept Capture: Dense Supervision And Good/Bad Basis

### What was done
Read and distilled the attached discussion into project artifacts. Added
`RESEARCH_CONCEPT_NOTES.md`, updated `COLLABORATOR_BRIEF.md`, updated
`methodology/RIGOR_GUARDRAILS.md`, added cumulative-context process scoring to
`methodology/experiment_roadmap.md`, and revised `paper/draft.md`.

### Key ideas captured
- Persona honesty is a lead example because the embedding preferred ontological
  honesty over warm fabrication, matching later assistant norms.
- Good/bad may be a self-regularizing primary axis: specific virtues can
  over-optimize into failure modes, but broad "good" contains cross-pressure
  from many senses of good.
- A scalar-plus-basis view is better than scalar-only or many independent
  targets: broad good/bad supplies the primary reward, while secondary axes
  diagnose or nudge.
- Cumulative full-context scoring is the proposed dense-supervision mechanism:
  embed the whole context so far after each reasoning step/turn and use score
  deltas as process signal.
- Token-level scoring is likely too noisy; sentence, paragraph, step, turn, or
  cumulative-context scoring is the plausible granularity.
- Embedding context-window limits are a real bottleneck; a serious version wants
  long-context embedding evaluators closer to generation-model context lengths.
- Reported axis-convergence numbers from the attached discussion are promising
  but are marked as needing local reproduction before becoming central
  evidence.

### Interpretation
The project's strongest framing is now broader than "cheap HH-RLHF agreement."
It is an evaluative-geometry and dense-supervision hypothesis: embedding axes
may score not only final answers, but the trajectory by which a model decomposes
and evaluates a situation.

### Decision
Preserve the human conceptual framing rather than reducing it to generic ML
language. Move cumulative-context process scoring into the roadmap as a no-GPU
mechanism test before any paid training attempt.

### Next steps
Reproduce the reported axis-convergence experiment locally with saved code and
outputs, then run the no-training candidate-selection and process-scoring
benchmarks before considering Colab/GPU fine-tuning.

## June 23, 2026 - Research Process Fix: Alternating Modes

### What was done
Converted the "Idea / Literature / Experiment / Autopsy / Forest" discussion
into concrete project process artifacts:

- `methodology/RESEARCH_LOOP_PROTOCOL.md`
- `methodology/MECHANISM_MAP.md`
- `methodology/templates/idea_mode.md`
- `methodology/templates/literature_mode.md`
- `methodology/templates/experiment_mode.md`
- `methodology/templates/autopsy_mode.md`
- `methodology/templates/forest_mode.md`
- `methodology/templates/decision_mode.md`
- `notes/research_cycles/README.md`
- `notes/research_cycles/cycle_001_next/idea.md`

Updated `README.md`, `COLLABORATOR_BRIEF.md`, and
`methodology/experiment_roadmap.md` so future work starts from the loop rather
than from a single benchmark.

### Key numbers
No new model-evaluation numbers. This is a process correction.

### Interpretation
The project failure mode was premature narrowing: asking whether the embedding
score matched HH-RLHF, then treating HH agreement as the result. The new loop
requires broad mechanism generation, implication-focused literature review,
frozen experiments, example autopsy, and forest-level interpretation before a
decision.

### Decision
Future experiments are incomplete unless they produce an autopsy and forest
memo. Disagreement examples, both-bad cases, benchmark assumptions, and training
mechanisms are now required outputs.

### Next steps
Use `cycle_001_next` to run the no-training intervention benchmark or the
cumulative-context process-scoring simulation. Before either run, fill in the
remaining mode templates for that cycle.

## June 23, 2026 - Cycle 001: Intervention Benchmark Scaffold

### What was done
Completed the first serious research-cycle chunk around the no-training
candidate-selection intervention test.

Created or completed:

- `notes/research_cycles/cycle_001_next/experiment.md`
- `notes/research_cycles/cycle_001_next/autopsy.md`
- `notes/research_cycles/cycle_001_next/forest.md`
- `notes/research_cycles/cycle_001_next/decision.md`
- `notes/research_cycles/cycle_001_next/seed_candidates.json`
- `scripts/run_cycle001_intervention.py`
- `notes/research_cycles/cycle_001_next/smoke_results/*`
- `notes/research_cycles/cycle_001_next/gemini_smoke_results/quota_blocked.md`

Also updated `README.md`, `COLLABORATOR_BRIEF.md`, `.gitignore`, and
`methodology/experiment_roadmap.md`.

### Key numbers
- Smoke fixture size: 5 prompts, 15 candidates.
- Lexical backend: completed successfully with direct and decomposition
  interfaces.
- Fixture expected-hit rates:
  - `decomposition_combined`: 4/5 = 80.0%.
  - `decomposition_general_evaluative`: 4/5 = 80.0%.
  - `direct_combined`: 3/5 = 60.0%.
  - `length`: 3/5 = 60.0%.
  - `random`: 2/5 = 40.0%.
  - `sentiment`: 1/5 = 20.0%.
- Generated smoke artifacts: `scores.csv`, `scores.json`, `selections.csv`,
  `selections.json`, `blind_review_packet.jsonl`, and `summary.md`.
- Gemini smoke attempt: blocked by HTTP 429 quota exceeded during the embedding
  probe before any Gemini scores were produced.

### Interpretation
The smoke run is not evidence for the research thesis because it is tiny,
hand-authored, and not blinded. It does verify the benchmark interface and shows
why decomposition scoring deserves a serious test: even a crude lexical backend
uses explicit good/bad decomposition more effectively than direct scoring on
the seed fixture.

The Gemini failure is not a Colab or browser-control problem. It is an API quota
or billing-access problem for the embedding model on the available key.

### Decision
The next research action is a blinded 50-prompt intervention pilot using this
scaffold. Treat Gemini as the preferred backend once quota exists; otherwise use
the scaffold to prepare candidate sets and blind review packets without making
model-quality claims.

### Next steps
1. Build a 50-prompt pilot candidate set from mixed sources.
2. Run Gemini embedding mode when quota is available.
3. Blind-review method winners.
4. Fill the autopsy table with actual counts.
5. Promote or demote the mechanism based on blind win rate plus example
   autopsy, not raw dataset agreement.

## June 23, 2026 - Cycle 001: Quota-Free Pilot Build And Local Embeddings

### What was done
Built and ran the quota-free pieces of the next research pipeline:

- Created a 50-prompt intervention pilot with 4 candidates per prompt.
- Used 25 audited HH disagreement prompts and 25 constructed
  adversarial/context prompts.
- Generated pilot answer key and blind review packets.
- Generated a blind HH disagreement adjudication packet from the actual grading
  table.
- Added refusal heuristic and category-axis baselines to
  `scripts/run_cycle001_intervention.py`.
- Ran lexical cheap baselines.
- Installed FastEmbed outside the repo and ran local `BAAI/bge-small-en-v1.5`
  embedding baselines.

### Key numbers
- Pilot size: 50 prompts, 200 candidates.
- Pilot source split: 25 HH disagreement prompts, 25 constructed adversarial
  prompts.
- Category counts: 10 anti-sycophancy, 10 persona honesty, 10 general
  helpfulness, 8 harmful request, 5 factuality, 5 false premise, 1 context
  negation, 1 privacy safety.
- HH blind adjudication packet: 108 gradeable cases from the table
  (`EMBEDDING_RIGHT` 63, `HH_RIGHT` 45, `EXCLUDE` 123). This conflicts with the
  prose summary's 109 gradeable count, and the discrepancy is documented.
- 50-prompt lexical proxy hits:
  - Length: 33/50 = 66.0%.
  - Direct anti-sycophancy: 29/50 = 58.0%.
  - Direct harm reduction: 28/50 = 56.0%.
  - Refusal heuristic: 25/50 = 50.0%.
  - Random: 16/50 = 32.0%.
  - Sentiment: 15/50 = 30.0%.
- 50-prompt FastEmbed/BGE-small proxy hits:
  - Length: 33/50 = 66.0%.
  - Refusal heuristic: 25/50 = 50.0%.
  - Direct anti-sycophancy: 23/50 = 46.0%.
  - Direct category axis: 19/50 = 38.0%.
  - Random: 16/50 = 32.0%.
- Adversarial subset FastEmbed/BGE-small:
  - Length: 24/25 = 96.0%.
  - Refusal heuristic: 21/25 = 84.0%.
  - Direct anti-sycophancy: 18/25 = 72.0%.
- Local install footprint outside repo:
  - `C:\Users\93rob\.cache\codex-embedding-venv`: about 169 MB.
  - `C:\Users\93rob\.cache\codex-fastembed`: about 64 MB.

### Interpretation
The quota-free run produced useful infrastructure and a clear diagnosis, not a
positive result to oversell. The current pilot is length-biased: proxy-best
answers are often longer, so length is too strong. That makes this packet useful
for blind review and stress testing, but the constructed adversarial cases need
length-balanced variants before their proxy-hit rates can be used as evidence.

The local BGE-small run did not beat length or refusal heuristic on the
50-prompt proxy key. This is a real result for the current pilot design, not a
decisive negative result for evaluative geometry. The benchmark interface and
candidate construction are now the next bottleneck.

### Decision
Keep the 50-prompt packet and HH blind adjudication packet. Do not claim the
current proxy-hit rates support the thesis. The next no-quota improvement is to
length-balance the constructed adversarial candidates and then run blind review
on method-selected winners.

### Next steps
1. Build a length-balanced v2 adversarial set.
2. Rerun lexical and FastEmbed baselines.
3. Generate method-winner review packets for blind adjudication.
4. Use Gemini only after quota is available, as a stronger-model comparison.

## June 23, 2026 - Cycle 002: Potential-Shaping Reframe And Controlled Battery

### What was done
Read and integrated the attached external-style research review. Verified the
new prior-work citations for Valence-Assent Axis, PGSRM, TRACE interaction
dynamics, potential-based reward shaping, Turney semantic orientation, Value
Entanglement, and Reusing Embeddings.

Created Cycle 002 artifacts:

- `notes/research_cycles/cycle_002_potential_shaping/idea.md`
- `notes/research_cycles/cycle_002_potential_shaping/literature.md`
- `notes/research_cycles/cycle_002_potential_shaping/experiment.md`
- `notes/research_cycles/cycle_002_potential_shaping/decision.md`
- `notes/research_cycles/cycle_002_potential_shaping/results.md`
- `notes/research_cycles/cycle_002_potential_shaping/controlled_evaluative_axis_battery.jsonl`
- `notes/research_cycles/cycle_002_potential_shaping/controlled_evaluative_axis_battery_v2_length_balanced.jsonl`
- `scripts/run_evaluative_axis_battery.py`

Updated `paper/draft.md` and `methodology/experiment_roadmap.md` so the
project now leads with controlled minimal-pair testing and cumulative-context
potential shaping rather than raw HH agreement.

### Key numbers
- HH table-backed disagreement audit:
  - `EMBEDDING_RIGHT`: 63.
  - `HH_RIGHT`: 45.
  - `EXCLUDE`: 123.
  - Gradeable embedding win rate: 63/108 = 58.3%.
  - Exact two-sided binomial p-value vs 50%: 0.101.
- Controlled battery v0:
  - 23 cases.
  - Better answer longer in 20/23 cases.
  - Mean absolute length gap: 3.57 words.
  - Length baseline: 89.1%.
  - FastEmbed/BGE-small direct combined: 26.1%.
  - FastEmbed/BGE-small decomposition category axis: 52.2%.
- Controlled battery v2:
  - 12 cases.
  - Exact word-count ties in all 12 pairs.
  - Length baseline: 50.0%.
  - Sentiment baseline: 50.0%.
  - FastEmbed/BGE-small direct combined: 8.3%.
  - FastEmbed/BGE-small direct broad evaluative: 0.0%.
  - FastEmbed/BGE-small direct anti-sycophancy: 66.7%.
  - FastEmbed/BGE-small decomposition category axis: 58.3%.

### Interpretation
The attachment's critique is correct: the earlier "corrected accuracy" framing
was too strong because agreement cases were not adjudicated. The HH audit is
qualitative and suggestive until blind review samples both agreements and
disagreements.

The new controlled battery gives a sharper diagnostic. Once length is exactly
controlled, local BGE-small broad good/bad scoring fails badly on the v2
minimal pairs. Narrow axes, especially anti-sycophancy, do better but are not
yet strong enough to claim a general evaluator. This weakens the naive direct
scalar version and strengthens the scalar-plus-basis and potential-shaping
version.

### Decision
Promote cumulative full-context potential shaping as the strongest research
formulation. Keep BGE-small results as a local baseline and negative diagnostic.
Do not spend Gemini quota on broad claims until the length-balanced battery is
expanded and the trajectory-delta test exists.

### Next steps
1. Expand the exact length-balanced v2 battery to 50 cases.
2. Build trajectory cases with injected errors and repairs.
3. Test cumulative potential deltas against final-only and isolated-step
   scoring.
4. Run Gemini on the same battery when quota is available.

## June 23, 2026 - Cycle 002: Local Model Sweep And Trajectory Probe

### What was done

Continued the no-quota research path after BGE-small failed broad direct
good/bad scoring. Added `scripts/sweep_fastembed_battery.py`, fixed a
category-axis mapping bug in `scripts/run_cycle001_intervention.py`, reran the
exact word-count-matched v2 battery across 8 local FastEmbed models, ran the
best local model on the existing 50-prompt proxy pilot, and added
`scripts/run_trajectory_potential_test.py` for a first cumulative process-state
probe.

### Key numbers

- Controlled battery v2 remains 12 exact word-count-tied pairs; length and
  sentiment baselines are both 50.0%.
- The fixed map changed category-axis scoring because the battery categories
  `truthfulness`, `harm_reduction`, `reasoning_rigor`, `context_binding`, and
  `helpfulness` had previously fallen back too often to `general_evaluative`.
- Corrected 8-model sweep:
  - `jinaai/jina-embeddings-v2-small-en`:
    oracle `decomposition_category_axis` = 11/12 = 91.7%.
  - This number is contaminated by the hand-authored decomposition fields,
    which explicitly used "Good parts" and "Bad parts" language. It should not
    be treated as evidence that the embedding model independently inferred
    answer quality.
  - Other strong narrow-axis results: `gte-base` direct harm reduction 75.0%,
    Snowflake direct persona honesty 75.0%, Nomic decomposition harm reduction
    75.0%.
  - Broad direct combined scores stayed poor across models: 8.3% to 33.3%.
- Jina on the old 50-prompt proxy pilot did not validate the controlled result:
  length baseline 33/50 = 66.0%, best embedding method
  `decomposition_persona_honesty` 23/50 = 46.0%,
  `decomposition_category_axis` 15/50 = 30.0%.
- Naive trajectory-potential probe failed:
  - BGE-small category-axis integral 25.0%, combined integral 0.0%.
  - Jina-small category-axis integral 33.3%, combined integral 0.0%.
- Model cache footprint is outside the repo:
  `C:\Users\93rob\.cache\codex-fastembed` about 3.1 GB. Sweep artifacts inside
  the repo are about 0.35 MB.

### Interpretation

BGE-small is not the only available local model, but the apparent 91.7% Jina
result was an oracle-label leakage result, not a valid evaluator result. It
shows only that embeddings can read explicit evaluative labels inserted into
the input. It should remain as a plumbing sanity check and should be excluded
from claims about raw or blind evaluation.

The old 50-prompt proxy pilot remains a stress-test artifact rather than a
validation set, because its labels and candidates are length-biased and not
blind-adjudicated. The failed trajectory probe is also useful: it prevents the
project from assuming that any cumulative reasoning text automatically creates a
usable dense reward signal.

### Decision

Continue the project, but demote the Jina 91.7% result. The next gate must be a
leakage-controlled test: raw answer scoring and blind LLM-generated
decomposition scoring, with no answer labels or good/bad annotations supplied
by the experimenter in the scored text. Do not claim that naive broad good/bad
scoring works, and do not claim oracle-decomposition results as evidence.

### Next steps

1. Build a no-leakage held-out battery from at least 50 cases.
2. Compare raw answer scoring, blind LLM-generated decomposition scoring,
   direct LLM judging, length, sentiment, and refusal heuristics.
3. Add ablations that remove evaluative words such as "good", "bad", "right",
   and "wrong" from decomposition text.
4. Build a better trajectory test using natural model traces or injected
   error/repair traces, not generic strategy templates.
5. Rerun Gemini on the same leakage-controlled battery when API quota is
   available.

## June 23, 2026 - Concept Correction: Emergent Decomposition, Not Oracle Labels

### What was done

Integrated the user's correction that the useful hypothesis is not
experimenter-written decomposition. The actual training hypothesis is that a
model optimized against a clean evaluative embedding reward may learn to
decompose its own scratchpad or planning traces into separable good-making and
bad-making factors, then choose compromises that preserve good and minimize bad.

Added:

- `methodology/EMERGENT_DECOMPOSITION_REWARD_PROTOCOL.md`
- roadmap section: "Emergent Decomposition Reward Simulation"

### Key numbers

No new model numbers. This is a research-design correction.

### Interpretation

Hand-authored decompositions such as `Good parts:` / `Bad parts:` are mostly
tests of whether embeddings read explicit valence words. They are marginally
useful as plumbing checks but not evidence for the central idea. The central
idea requires observing or inducing decomposition under reward pressure without
directly instructing decomposition or leaking labels.

The next useful no-GPU test is an evolutionary feedback simulation: generate
multiple scratchpad-style traces, score cumulative contexts with an embedding
reward, keep high-scoring traces, ask for variants using only scalar/rank
feedback, and test whether decomposition, tradeoff handling, and contextual
negation handling emerge.

### Decision

Prioritize emergent-decomposition and context-sensitivity tests over more
oracle decomposition scoring. The reward must handle protective negative-token
contexts such as `I should avoid introducing this bug` as good, and
positive-sounding unsafe contexts such as `You're absolutely right` as bad when
they support false or harmful premises.

### Next steps

1. Build a context-sensitivity battery around protective negative terms,
   positive-surface bad reasoning, tradeoffs, and repair.
2. Implement the no-training evolutionary feedback loop.
3. Report whether high-scoring traces show spontaneous decomposition under
   blind review.
4. Only after that, consider DPO/GRPO/QLoRA training with embedding-ranked
   traces.
