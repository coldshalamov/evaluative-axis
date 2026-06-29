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

## June 25, 2026 - Cycle 003: Objective Code Reranking Pilot

### What was done

Ran a design-first cycle to choose the best next experiment under the real
constraints: one laptop, free-tier Google access, no hired annotators, and no
assumption that HH is ground truth. Ranked 20 candidate tests across thesis
fit, falsification power, budget fit, contamination resistance, and decision
value. The top choice was objective code reranking with hidden unit tests.

Implemented:

- `scripts/run_objective_code_reranking.py`
- `notes/research_cycles/cycle_003_objective_code_reranking/idea.md`
- `notes/research_cycles/cycle_003_objective_code_reranking/experiment.md`
- `notes/research_cycles/cycle_003_objective_code_reranking/results.md`

Tried to run the full generated-candidate protocol first. Gemini embeddings
worked, but Gemini generation hit repeated HTTP 429 responses and
`mcp__colab_mcp.open_colab_browser_connection` returned `false`. Used the
documented fallback: local execution with curated candidate sets and hidden
tests.

### Key results

- 20 candidate tests were ranked; top 5:
  1. objective code reranking with hidden tests
  2. length-controlled no-leakage open-ended reranking
  3. objective math reranking with exact-answer checks
  4. injected error/repair potential shaping
  5. Gemini-blind HH disagreement adjudication
- Objective code pilot size:
  - 6 Python tasks
  - 3 candidates per task
- Hidden-test outcomes:
  - random: 3/6 solved = 50.0%
  - length: 3/6 solved = 50.0%
  - direct broad evaluative: 5/6 solved = 83.3%
  - direct code-quality evaluative: 5/6 solved = 83.3%
- Avg selected pass rate:
  - random: 83.3%
  - length: 83.3%
  - direct broad evaluative: 96.7%
  - direct code-quality evaluative: 96.7%
- Best-candidate hit rate:
  - random: 50.0%
  - length: 50.0%
  - direct broad evaluative: 83.3%
  - direct code-quality evaluative: 83.3%
- The single miss was `balanced_brackets`, where the scorer slightly preferred
  a plausible but order-blind stack solution over the fully correct parser.
- Colab OSS rerun on the same curated benchmark:
  - `sentence-transformers/all-mpnet-base-v2`
    - random: 3/6
    - length: 3/6
    - direct broad evaluative: 1/6
    - direct code-quality evaluative: 1/6
  - `BAAI/bge-base-en-v1.5`
    - random: 3/6
    - length: 3/6
    - direct broad evaluative: 3/6
    - direct code-quality evaluative: 3/6
- The browser approval path was real: the Colab approval modal appeared in
  Chrome and could be clicked. After approval, the MCP connector no longer
  returned immediate `false`, but the transport still closed. The browser-side
  Colab terminal remained usable and was used to run the OSS ablation.

### Interpretation

This is the first clean objective intervention result in the repo that does not
depend on HH labels or an LLM judge as the final metric. It is only a small
pilot and uses curated candidates, so it should not be framed as proof that the
full training thesis is solved. But it is meaningful directional evidence that
direct evaluative geometry can already function as a useful reranker against
objective hidden correctness.

The failure mode is also informative. The miss was not a length shortcut or a
pretty-answer effect. The selected wrong answer was almost correct and failed a
single order-sensitive hidden case. That suggests the current signal can
confuse local algorithmic plausibility with full structural constraint
satisfaction.

The Colab OSS ablation adds an equally important constraint: low-cost
open-source embeddings did not reproduce the Gemini pilot lift. `all-mpnet`
performed materially worse than the baselines, and `bge-base` only tied them.
That makes the project's hardware-cost problem more concrete instead of more
speculative. The sharpest current interpretation is that a Gemini-family
embedding likely benefits from a much larger and more capable base model than
typical OSS embedders, even though the exact parameter count is not publicly
stated.

### Decision

Promote objective reranking with hidden end metrics to the main no-GPU proof
path. Demote HH overlap further from "benchmark" to "sensor." The next useful
question is not whether the signal imitates HH more closely, but whether it
continues to beat cheap baselines on larger objective reranking tasks and on
other objective domains.

### Next steps

1. Scale the code reranking pilot from 6 tasks to at least 30-50 tasks with
   more adversarial hidden checks.
2. Re-enable critique-based scoring when Gemini generation quota or Colab
   access becomes available.
3. Run the same protocol on objective math tasks with exact answers.
4. Add a small-margin uncertainty rule and test abstain/resample policies on
   close calls.

## June 25, 2026 - Research System V1: Cross-Domain Evidence Program

### What was done

Turned the repo's methodology into a frozen research program with explicit claim
gates, benchmark lanes, and an executable report.

Added:

- `experiments/research_system_v1/program_manifest.json`
- `methodology/SERIOUS_RESEARCH_SYSTEM_V1.md`
- `scripts/orchestrate_research_system.py`
- `scripts/run_objective_text_reranking.py`
- `experiments/research_system_v1/benchmarks/objective_math_reranking_v1.json`
- `experiments/research_system_v1/benchmarks/tool_interpretation_reranking_v1.json`
- `experiments/research_system_v1/benchmarks/process_potential_error_repair_v1_spec.md`

Executed the full ready set that fit the available hardware:

- Gemini behavior-basis battery
- Gemini objective math reranking
- Gemini tool-interpretation reranking
- local BGE-base reruns for math and tool after installing
  `sentence-transformers`

### Key results

- Program report:
  - `notes/research_system_v1/report/report.md`
  - lanes with results: 7
  - `capacity_code_gate`: pass
  - `capacity_cross_domain_gate`: pass
  - `behavior_basis_gate`: pass
  - `cross_domain_selection_gate`: pass
  - `process_potential_gate`: pending
  - `training_readiness_gate`: pending
- Behavior-basis battery:
  - `direct_category_axis`: 91.7%
  - `decomposition_category_axis`: 100.0%
  - `direct_combined`: 83.3%
  - `direct_general_evaluative`: 33.3%
- Objective math reranking:
  - Gemini `direct_combined`: 8/8 = 100.0%
  - baseline best (`length`): 4/8 = 50.0%
  - BGE-base `direct_target_axis`: 5/8 = 62.5%
- Tool-interpretation reranking:
  - Gemini `direct_target_axis`: 7/8 = 87.5%
  - baseline best: 3/8 = 37.5%
  - BGE-base `direct_target_axis`: 4/8 = 50.0%
- Existing code pilot remains:
  - Gemini direct methods: 5/6 = 83.3%
  - baseline best: 3/6 = 50.0%
  - BGE-base: 3/6 = 50.0%
  - `all-mpnet-base-v2`: 1/6 = 16.7%

### Interpretation

This is the first version of the repo that looks like a real evidence program
instead of a collection of promising anecdotes.

The strongest current result is not just that Gemini embeddings looked good on
one benchmark. It is that frozen, objective, no-judge end metrics now show
clear lift across three different domains: code, math, and tool/log
interpretation.

An equally important constraint also became clearer. The broad generic
`general_evaluative` direction is not universally enough by itself. It was
excellent on the small math suite, weak on the tool-interpretation suite, and
poor on the controlled behavior battery. Category-aware or decomposition-aware
scoring carried much more of the useful signal. That makes the project more
interesting and more specific: the thesis is no longer just "good/bad works";
it is that evaluative embedding geometry can carry actionable selection signal,
and targeted axes matter.

The capacity story also strengthened. BGE-base improved slightly over cheap
baselines on the new math and tool suites, but the lift was much smaller than
Gemini's. That does not prove parameter count as the mechanism, but it does
support the operational claim that more capable embedding families are much more
useful for this method than small cheap OSS encoders.

### Decision

Promote `research_system_v1` as the main proof frame for the repo.

The project can now credibly claim:

1. objective cross-domain selection evidence exists;
2. targeted evaluative axes beat naive baselines on a controlled behavior
   battery;
3. cheap OSS embedders do not match Gemini's lift on the same frozen suites.

The project must not yet claim that embedding scores are ready to train on as a
dense reward. That remains blocked on the process-potential suite.

### Next steps

1. Implement `process_potential_error_repair_v1` using the frozen spec and make
   `dense_reward_localization_score` the next decisive metric.
2. Expand the objective math and tool suites from 8 tasks each to at least
   30-50 tasks with harder distractors.
3. Add one stronger OSS comparison model above BGE-base if laptop or Colab
   budget allows.
4. Package the report, benchmark files, and key summaries into an investor /
   partner briefing packet with honest claim boundaries.

## June 27, 2026 - Cycle 004: Good vs Proxy Conflicts

### What was done

Built a contamination-aware word-level comparison to test the user's stronger
thesis more directly: whether raw `good/bad` behaves like a broader evaluative
axis than nearby proxy words such as `true`, `honest`, `useful`, `helpful`,
`accurate`, `correct`, and `safe`.

Added:

- `methodology/GOOD_VS_PROXY_CONFLICTS_PROTOCOL.md`
- `scripts/run_good_vs_proxy_conflicts.py`
- `notes/research_cycles/cycle_004_good_vs_proxy_conflicts/`

Ran the word-level benchmark on:

- `gemini-embedding-2`
- `BAAI/bge-base-en-v1.5`

Then ran the existing richer evaluative-axis battery on the same 50-case file
to check whether the benchmark itself was too hard or whether the failure was
specific to raw/single-word axes.

### Key results

Word-level conflict outputs:

- `notes/research_system_v1/good_vs_proxy_conflicts_gemini_v1/summary.md`
- `notes/research_system_v1/good_vs_proxy_conflicts_bge_v1/summary.md`

Same-benchmark richer-axis control:

- `notes/research_system_v1/battery_v3_gemini_direct_v1/summary.md`

On `gemini-embedding-2`:

- `raw_good_bad`: 26.0%
- `sentence_good_bad`: 30.0%
- proxy mean: 34.8%
- best proxy: `raw_useful_useless` at 42.0%

On `BAAI/bge-base-en-v1.5`:

- `raw_good_bad`: 28.0%
- `sentence_good_bad`: 24.0%
- proxy mean: 21.8%
- best proxy: `raw_honest_dishonest` at 30.0%

On the same 50 cases, the richer Gemini evaluative axes were much stronger:

- `direct_general_evaluative`: 46.0%
- `direct_combined`: 86.0%
- `direct_category_axis`: 86.0%
- `direct_truthfulness`: 90.0%
- `direct_harm_reduction`: 94.0%
- `direct_persona_honesty`: 96.0%
- `direct_anti_sycophancy`: 98.0%

### Interpretation

This is one of the most useful negative results in the repo so far.

The easy overclaim was that a strong embedding model might already carry the
full broad evaluative signal in raw `good/bad` form. On this frozen 50-case
conflict battery, that did not happen. Nearby single-word proxy axes also
mostly failed.

At the same time, the benchmark itself is clearly not impossible, because the
richer targeted axes worked very well on the same cases. So the right current
conclusion is not "evaluative geometry failed." It is:

- the raw one-word version looks too weak in the current zero-shot setup;
- targeted evaluative criteria are much more recoverable and useful right now;
- if a pure `good/bad` training signal ever works, it likely requires stronger
  representation quality, more context, training on the signal, or all three.

### Decision

Narrow the publishable claim. Do not present raw `good/bad` as already
validated. Present the broader evaluative-geometry claim, the strong targeted
axis results, the objective reranking results, and this new negative result as
an honest boundary on what the current evidence can and cannot support.

### Next steps

1. Keep the new word-level conflict battery as a standing falsification test.
2. Focus new effort on targeted-axis conflict batteries rather than many more
   one-word variants.
3. Implement the process-potential lane, because that is still the main missing
   bridge from selection to training.
4. Expand the objective reranking suites, which remain the strongest current
   evidence family.

## June 27, 2026 - Cycle 005: Process Potential Error-Repair

### What was done

Built and ran the missing process-potential bridge test for
`research_system_v1`.

Added:

- `experiments/research_system_v1/benchmarks/process_potential_error_repair_v1.json`
- `scripts/run_process_potential_error_repair.py`
- `notes/research_cycles/cycle_005_process_potential_error_repair/`

Ran the suite on:

- `gemini-embedding-2`
- `BAAI/bge-base-en-v1.5`

Also included explicit controls for:

- `length`
- `sentiment`
- `final_answer_only_category_axis`
- `final_answer_only_combined`

Then regenerated:

- `notes/research_system_v1/report/report.md`

### Key results

Outputs:

- `notes/research_system_v1/process_potential_error_repair_v1/summary.md`
- `notes/research_system_v1/process_potential_error_repair_bge_v1/summary.md`

On `gemini-embedding-2`:

- `error_drop_accuracy`: 91.7%
- `repair_rise_accuracy`: 83.3%
- `error_localization_top1_accuracy`: 33.3%
- `repair_localization_top1_accuracy`: 66.7%
- `dense_reward_localization_score`: 50.0%
- `combined_dense_reward_localization_score`: 62.5%

On `BAAI/bge-base-en-v1.5`:

- `error_drop_accuracy`: 33.3%
- `repair_rise_accuracy`: 75.0%
- `error_localization_top1_accuracy`: 0.0%
- `repair_localization_top1_accuracy`: 41.7%
- `dense_reward_localization_score`: 20.8%
- `combined_dense_reward_localization_score`: 16.7%

Critical controls on Gemini:

- `length`: 0.0% dense score
- `sentiment`: 8.3% dense score
- `final_answer_only_category_axis`: 0.0% dense score
- `final_answer_only_combined`: 0.0% dense score

System gate result after report refresh:

- `process_potential_gate`: `fail`
- frozen threshold: `0.65`
- observed `dense_reward_localization_score`: `0.50`

### Interpretation

This is the cleanest bridge result in the repo so far between reranking and the
denser training thesis.

What it shows:

- the strong embedding model reacts to injected reasoning degradation and later
  repair inside the trace
- that effect is much stronger than what the cheap OSS embedder shows
- the effect is not explained by answer length, sentiment, or looking only at
  the final answer

What it does not show:

- that the current signal is already strong enough for dense reward training
- that the frozen process gate should be called a pass

The remaining misses are concentrated in `reasoning_rigor`, `persona_honesty`,
and some `harm_reduction` cases, which is useful guidance rather than a random
failure pattern.

### Decision

Promote the narrower claim:

1. process sensitivity exists;
2. stronger embedding families show much more of it than cheap OSS embedders;
3. training-readiness is still not demonstrated under the current gate.

### Fallback conditions

If future process runs continue to stall below the threshold after the suite is
expanded, the repo should treat this as evidence that the current embedding-only
signal is better framed as a reranking or critique aid than as a dense reward.

If expanded suites improve localization without relying on trivial controls or
post-hoc gate changes, the training-readiness claim can be revisited.

### Next steps

1. Expand the process suite from 12 traces toward 30-50 traces, especially in
   `reasoning_rigor`, `persona_honesty`, and `harm_reduction`.
2. Add multiple repair phrasings per phenomenon so the suite is less sensitive
   to one particular wording pattern.
3. Keep the current threshold frozen and resist reclassifying `62.5%` on the
   combined scorer as a pass after the fact.
4. Use the refreshed report as the honest external-facing research frame:
   strong selection evidence, clear capacity gap, real but not yet sufficient
   process signal.

## June 27, 2026 - Cycle 006: Partner Packet

### What was done

Built a reproducible external-facing packet directly from the current
`research_system_v1` outputs.

Added:

- `scripts/build_partner_packet.py`
- `notes/research_cycles/cycle_006_partner_packet/`

Generated:

- `paper/draft.md`
- `paper/partner_packet_v1/brief.md`
- `paper/partner_packet_v1/packet_summary.json`
- `paper/partner_packet_v1/figures/figure_selection_lift.svg`
- `paper/partner_packet_v1/figures/figure_word_vs_targeted.svg`
- `paper/partner_packet_v1/figures/figure_process_signal.svg`

### Key results

The repo now has one current-state summary artifact that:

- leads with objective reranking evidence rather than weak proxy overlap;
- shows the large Gemini-vs-cheap-OSS capability gap;
- keeps the raw `good/bad` negative result visible;
- and preserves the frozen failure of the process training-readiness gate.

The paper draft was also updated so its abstract, contributions, new
experiments, and conclusion no longer lead with the older HH-centric framing.

### Interpretation

This cycle does not create new empirical evidence. It upgrades the repo from a
collection of folders into a more inspectable research artifact.

That matters because the current bottleneck is not only more experiments. It is
also whether an external reader can see the evidence ladder and the non-claims
without having to reconstruct it manually from old drafts.

### Decision

Use `paper/partner_packet_v1/brief.md` as the main partner-facing summary until
the full paper draft is updated to the same evidence standard.

### Next steps

1. Update the full paper draft to align with the current report and packet.
2. Expand the objective suites and process suite while keeping the packet
   generator as the public-facing summary layer.
3. Add cost and confidence-interval reporting once those numbers are measured
   directly.

## June 27, 2026 - Cycle 007: Pairwise Blind Review Pilot

### What was done

Ran the first reusable blinded pairwise-review pilot on the old open-ended
Cycle 001 candidate pool.

Added or updated:

- `scripts/build_pairwise_review.py`
- `scripts/judge_pairwise_with_gemini.py`
- `scripts/run_failure_audit.py`
- `scripts/run_pairwise_blind_review_pilot.py`
- `notes/research_cycles/cycle_007_pairwise_blind_review_pilot/`

Operational fix:

- the original blind judge path used `gemini-2.0-flash`, which was quota
  blocked for this key
- probing showed `gemini-flash-lite-latest` was available
- the judge tooling was updated to accept an explicit model and the pilot was
  run on `gemini-flash-lite-latest`

Pilot design:

- use `notes/research_cycles/cycle_001_next/pilot_50_fastembed_bge_small/selections.json`
- build actually blind pairwise packets only where methods chose different
  candidates
- sample at most 10 rows per comparison
- require order-flip stability from the Gemini judge or treat the case as a tie

Focus methods:

- `direct_category_axis`
- `direct_anti_sycophancy`

Baselines:

- `length`
- `random`
- `sentiment`
- `refusal_heuristic`

### Key results

Permanent artifacts:

- `notes/research_cycles/cycle_007_pairwise_blind_review_pilot/runs/combined_summary.md`
- `notes/research_cycles/cycle_007_pairwise_blind_review_pilot/runs/direct_category_axis/`
- `notes/research_cycles/cycle_007_pairwise_blind_review_pilot/runs/direct_anti_sycophancy/`

For `direct_category_axis`:

- vs `length`: 1 win, 8 losses, 1 tie -> 11.1% decided win rate
- vs `random`: 5 wins, 3 losses, 2 ties -> 62.5%
- vs `refusal_heuristic`: 2 wins, 5 losses, 3 ties -> 28.6%
- vs `sentiment`: 8 wins, 1 loss, 1 tie -> 88.9%

For `direct_anti_sycophancy`:

- vs `length`: 3 wins, 6 losses, 1 tie -> 33.3%
- vs `random`: 3 wins, 5 losses, 2 ties -> 37.5%
- vs `refusal_heuristic`: 1 win, 7 losses, 2 ties -> 12.5%
- vs `sentiment`: 5 wins, 1 loss, 4 ties -> 83.3%

Judge stability:

- `direct_category_axis` pilot: 33 stable wins, 3 stable ties, 4 order-sensitive
  ties, 0 errors
- `direct_anti_sycophancy` pilot: 31 stable wins, 2 stable ties, 7
  order-sensitive ties, 0 errors

### Interpretation

This is an important negative result on the cheap open-ended lane.

What it shows:

- the blind-review tooling now works end-to-end and is reusable
- cheap BGE-small open-ended selectors do not beat the stronger cheap baselines
  on this inherited candidate pool
- the result is not just a proxy-key complaint anymore; it survives a blinded
  LLM judge sensor

What it does not show:

- that embedding selection fails in general
- that stronger embedding models would behave the same way
- that a clean length-controlled open-ended reranking set would also be
  negative

### Decision

Promote this as an honest negative pilot, not as a decisive intervention claim.

The practical reading is:

1. cheap OSS open-ended selection is not strong enough yet;
2. the objective reranking results remain the strongest practical evidence;
3. the next open-ended claim should use a fresh length-controlled candidate pool
   plus the stronger embedding family.

### Next steps

1. Re-run the same blind-review pilot structure with stronger embedding
   selections once those outputs exist.
2. Build a fresh no-leakage open-ended candidate pool with tight length control.
3. Keep `gemini-flash-lite-latest` as the working blind-judge default unless a
   better quota-available model is verified.

## June 27, 2026 - Cycle 008: Gemini Open-Ended Blind Review Pilot

### What was done

Reran the reusable open-ended blind-review pilot on the inherited Cycle 001
candidate pool using stronger Gemini embedding selections.

Added or updated artifacts:

- `.tmp/cycle001_gemini_direct_openended_pilot/`
- `notes/research_cycles/cycle_008_gemini_openended_blind_review_pilot/`

Core runs:

- `scripts/run_cycle001_intervention.py` with `gemini-embedding-2`,
  `--interfaces direct`
- `scripts/run_pairwise_blind_review_pilot.py` with
  `gemini-flash-lite-latest`

Focus methods:

- `direct_category_axis`
- `direct_harm_reduction`

Baselines:

- `length`
- `random`
- `sentiment`
- `refusal_heuristic`

Also completed a matched cheap-BGE comparison for:

- `direct_harm_reduction`

under the same blind-review protocol.

### Key results

Gemini selector outputs:

- `notes/research_cycles/cycle_008_gemini_openended_blind_review_pilot/runs/combined_summary.md`
- `notes/research_cycles/cycle_008_gemini_openended_blind_review_pilot/runs/direct_category_axis/analysis/summary.md`
- `notes/research_cycles/cycle_008_gemini_openended_blind_review_pilot/runs/direct_harm_reduction/analysis/summary.md`

Matched cheap comparison:

- `notes/research_cycles/cycle_008_gemini_openended_blind_review_pilot/runs_bge_harm/direct_harm_reduction/analysis/summary.md`

For Gemini `direct_category_axis`:

- vs `length`: 20.0%
- vs `random`: 55.6%
- vs `refusal_heuristic`: 0.0%
- vs `sentiment`: 100.0%

For Gemini `direct_harm_reduction`:

- vs `length`: 30.0%
- vs `random`: 88.9%
- vs `refusal_heuristic`: 37.5%
- vs `sentiment`: 100.0%

For cheap BGE `direct_harm_reduction` on the same pilot:

- vs `length`: 12.5%
- vs `random`: 25.0%
- vs `refusal_heuristic`: 20.0%
- vs `sentiment`: 71.4%

### Interpretation

This is a useful capability-gap result on the open-ended lane.

What it shows:

- the stronger embedding backend materially improves blind-review outcomes on
  the same inherited pool
- the targeted harm-reduction selector is substantially stronger than the cheap
  BGE version under the same protocol
- the pilot is still not just a proxy-key story; it is now backed by blinded
  LLM adjudication

What it does not show:

- that the repo already has a decisive open-ended selection benchmark
- that the inherited Cycle 001 pool is clean enough for a partner-grade claim

The persistent problem is that even Gemini still loses to `length` and
`refusal_heuristic` on this old pool.

### Decision

Promote this cycle as exploratory evidence that backend quality matters a lot
for open-ended selection.

Do not promote it as the decisive intervention result. The next real open-ended
claim still needs a fresh length-controlled candidate pool.

### Next steps

1. Build the fresh no-leakage open-ended pool both Cycle 007 and Cycle 008 now
   point toward.
2. Keep `direct_harm_reduction` as a lead selector on the next open-ended run.
3. Treat the old Cycle 001 pool as an exploratory stress surface, not the final
   benchmark.

## June 27, 2026 - Cycle 009: OSS Direct-Only Battery Sweep

### What was done

Mapped the free/local embedding landscape on the current 50-case controlled
evaluative battery using direct-only scoring.

Added or updated:

- `scripts/sweep_fastembed_battery.py`
- `notes/research_cycles/cycle_009_oss_direct_battery_v3_sweep/`

Battery:

- `notes/research_cycles/cycle_002_potential_shaping/controlled_evaluative_axis_battery_v3_50_cases.jsonl`

Interfaces:

- `direct`

Models swept:

- `BAAI/bge-small-en-v1.5`
- `BAAI/bge-base-en-v1.5`
- `thenlper/gte-base`
- `snowflake/snowflake-arctic-embed-m`
- `jinaai/jina-embeddings-v2-small-en`
- `jinaai/jina-embeddings-v2-base-en`
- `nomic-ai/nomic-embed-text-v1.5-Q`
- `mixedbread-ai/mxbai-embed-large-v1`

Operational note:

- the first attempt used the thread's default `python.exe`, which did not have
  `fastembed`
- the sweep was rerun successfully under
  `C:\Users\93rob\.cache\codex-embedding-venv\Scripts\python.exe`

### Key results

Aggregate outputs:

- `notes/research_cycles/cycle_009_oss_direct_battery_v3_sweep/summary.md`
- `notes/research_cycles/cycle_009_oss_direct_battery_v3_sweep/sweep_results.json`

Best local direct-only results:

- best `direct_combined`: `snowflake/snowflake-arctic-embed-m` at 34.0%
- best `direct_category_axis`: `snowflake/snowflake-arctic-embed-m` at 50.0%
- best `direct_harm_reduction`: `jinaai/jina-embeddings-v2-small-en` and
  `jinaai/jina-embeddings-v2-base-en` at 64.0%
- best `direct_persona_honesty`: `snowflake/snowflake-arctic-embed-m` at 74.0%
- best `direct_anti_sycophancy`: `BAAI/bge-small-en-v1.5` at 62.0%

Gemini comparison on the same battery:

- `direct_combined`: 86.0%
- `direct_category_axis`: 86.0%
- `direct_truthfulness`: 90.0%
- `direct_harm_reduction`: 94.0%
- `direct_persona_honesty`: 96.0%
- `direct_anti_sycophancy`: 98.0%

Important baseline fact:

- no local model beat `refusal = 57.0%` on `direct_combined`
- no local model beat `refusal = 57.0%` on `direct_category_axis`

### Interpretation

This cycle sharpens the local-model story in a way the repo needed.

What it shows:

- the local landscape is not flat; some models are meaningfully better than
  cheap BGE on narrow axes
- but none of the free/local models approach Gemini on the cleaner direct-only
  aggregate metrics
- the local models still look like fragmented narrow evaluators rather than
  strong general answer selectors on this battery

What it does not show:

- that parameter count alone explains the gap
- that all OSS embeddings are equally bad

### Decision

Promote this as the current model-landscape map for the clean direct-only
battery.

It supports a real capability-gap story while keeping the causal claim narrow:
better embedding families help a lot, but the free/local models on this laptop
still do not match Gemini on the direct-only battery.

### Next steps

1. Use this sweep as the default answer when asked whether free Hugging Face
   models already match Gemini on the clean battery.
2. If quota allows, run the raw `good/bad` vs proxy-word conflict protocol on
   one or two of the strongest locals from this sweep.
3. Keep future partner-facing claims anchored in objective reranking,
   targeted-axis batteries, and process-potential diagnostics rather than raw
   broad-word axes alone.

## June 27, 2026 - Cycle 010: OSS Good-vs-Proxy Sweep

### What was done

Extended the word-level conflict harness to the local FastEmbed family and ran
the current 50-case good-vs-proxy battery across the main free/local models we
can execute on this laptop.

Added or updated:

- `scripts/run_good_vs_proxy_conflicts.py`
- `scripts/sweep_good_vs_proxy_conflicts.py`
- `notes/research_cycles/cycle_010_oss_good_vs_proxy_sweep/`

Battery:

- `notes/research_cycles/cycle_002_potential_shaping/controlled_evaluative_axis_battery_v3_50_cases.jsonl`

Models swept:

- `BAAI/bge-small-en-v1.5`
- `BAAI/bge-base-en-v1.5`
- `thenlper/gte-base`
- `snowflake/snowflake-arctic-embed-m`
- `jinaai/jina-embeddings-v2-small-en`
- `jinaai/jina-embeddings-v2-base-en`
- `nomic-ai/nomic-embed-text-v1.5-Q`
- `mixedbread-ai/mxbai-embed-large-v1`

Operational note:

- the sweep ran under
  `C:\Users\93rob\.cache\codex-embedding-venv\Scripts\python.exe`

### Key results

Aggregate outputs:

- `notes/research_cycles/cycle_010_oss_good_vs_proxy_sweep/summary.md`
- `notes/research_cycles/cycle_010_oss_good_vs_proxy_sweep/sweep_results.json`

Raw `good/bad` accuracy across the local family:

- best local: `snowflake/snowflake-arctic-embed-m` at 48.0%
- next best: `BAAI/bge-base-en-v1.5` at 28.0%
- most others: 14.0-22.0%

Sentence `This response is good/bad.`:

- best local: `snowflake/snowflake-arctic-embed-m` at 36.0%
- most others: 12.0-26.0%

Best nearby proxy by model:

- `snowflake/snowflake-arctic-embed-m`: `helpful/unhelpful` at 58.0%
- `BAAI/bge-base-en-v1.5`: `honest/dishonest` at 30.0%
- `thenlper/gte-base`: `honest/dishonest` at 24.0%
- `jinaai/jina-embeddings-v2-small-en`: `useful/useless` at 28.0%
- `jinaai/jina-embeddings-v2-base-en`: `helpful/unhelpful` at 32.0%
- `nomic-ai/nomic-embed-text-v1.5-Q`: `honest/dishonest` at 34.0%
- `mixedbread-ai/mxbai-embed-large-v1`: `honest/dishonest` at 32.0%

### Interpretation

This cycle makes the surrounding-word story much more concrete.

What it shows:

- the raw broad-word failure was not only a BGE artifact
- most local models still fail badly on raw `good/bad`
- some local models partially recover the signal, especially Snowflake, but not
  enough to make raw `good/bad` look robust
- nearby proxy words remain fragmented and model-specific rather than converging
  on one simple replacement for `good/bad`

Taken together with the direct-only targeted-axis sweep, the pattern is now:

- broad raw-word axes are weak
- richer targeted evaluative directions are much more useful

### Decision

Promote this cycle as the current local-model landscape map for the raw-word
versus proxy-word hypothesis.

The narrower thesis gets stronger: the usable current signal is in richer
targeted evaluative geometry, not in the minimalist bare-word setup.

### Next steps

1. Fold this result into the paper framing so the repo no longer overfocuses on
   only Gemini plus BGE for the broad-word story.
2. Use the current local-model evidence to justify focusing future cheap/local
   work on targeted axes and process sensitivity rather than many more raw-word
   permutations.

## June 27, 2026 - Cycle 011: Battery v3 Gemini Reproduction

### What was done

Reproduced the exact current 50-case length-controlled v3 battery on Gemini and
paired it with the matching raw `good/bad` versus proxy-word companion run.

Added:

- `notes/research_cycles/cycle_011_battery_v3_gemini_reproduction/`

Executed:

- `scripts/run_evaluative_axis_battery.py`
- `scripts/run_good_vs_proxy_conflicts.py`

Frozen input:

- `notes/research_cycles/cycle_002_potential_shaping/controlled_evaluative_axis_battery_v3_50_cases.jsonl`

Operational note:

- both runs hit HTTP 429 retries mid-run
- both completed successfully without changing protocol

### Key results

Direct routed battery reproduction:

- `length`: 50.0%
- `sentiment`: 44.0%
- `refusal`: 57.0%
- `direct_combined`: 86.0%
- `direct_category_axis`: 86.0%
- `direct_general_evaluative`: 46.0%
- `direct_truthfulness`: 90.0%
- `direct_harm_reduction`: 94.0%
- `direct_persona_honesty`: 96.0%
- `direct_anti_sycophancy`: 98.0%

Broad-word companion reproduction:

- `raw_good_bad`: 26.0%
- `sentence_good_bad`: 30.0%
- proxy mean: 34.8%
- best proxy: `raw_useful_useless` at 42.0%

Structured miss pattern:

- `direct_combined` misses concentrated in `helpfulness` and
  `reasoning_rigor`
- `direct_category_axis` misses concentrated in `helpfulness` and `mixed`
- `direct_general_evaluative` collapsed to 46.0% overall and 0.0% on both
  `anti_sycophancy` and `persona_honesty`
- `raw_good_bad` hit 0.0% on `anti_sycophancy`, `persona_honesty`, and `mixed`

### Interpretation

This is an important reproduction and boundary-setting cycle.

What it shows:

- the strong Gemini targeted-axis measurement result on the v3 battery is
  stable on the exact saved file
- the same exact battery also stably reproduces the weak broad-word
  `good/bad` result
- the current evidence therefore favors routed targeted evaluative geometry,
  not a minimalist raw-word interface

What it does not show:

- a decisive intervention result
- a training-readiness result
- that raw `good/bad` is already a generally reliable zero-shot selector

### Decision

Promote this cycle as a confirmatory reproduction artifact rather than as a new
headline claim.

### Fallback conditions

If later reruns on the same file drift materially without an understood cause,
the repo should treat the battery as unstable and stop using it as a clean
reference point until that drift is explained.

If future longer-context or trained variants make raw `good/bad` competitive on
fresh held-out batteries, revisit the current narrow framing rather than
assuming this zero-shot failure is permanent.

### Next steps

1. Use this cycle as the clean answer when asked whether the saved Gemini v3
   battery result still reproduces on the current file.
2. Keep the highest-value next experiment on the length-controlled open-ended
   blind-review lane.
3. Expand future controlled batteries around `helpfulness` and `mixed`
   phenomena, which remain the main routed-axis weak spots here.

## June 27, 2026 - Cycle 012: Length-Controlled Open-Ended Pool

### What was done

Built the missing fresh open-ended candidate-pool builder and used it to create
the first frozen partial pilot on a new strict-length surface.

Added:

- `scripts/build_length_controlled_openended_pool.py`
- `notes/research_cycles/cycle_012_length_controlled_openended_pool/`

Key design choices:

- mixed-category routed prompt spec
- 4 generated candidates per prompt
- exact 60-word target per candidate
- checkpointing after each completed item
- resume support after quota interruptions
- candidate-level regeneration when exact-count rewriting fails

Generated so far:

- 8-item pilot snapshot from the fresh pool

Also ran:

- a no-quota local `snowflake/snowflake-arctic-embed-m` direct-only selection
  pass on the 8-item snapshot
- a blind-review packet build on `direct_category_axis` versus cheap baselines

### Key results

Fresh-pool snapshot:

- items generated: 8
- categories covered:
  - `persona_honesty`: 2
  - `harmful_request`: 2
  - `anti_sycophancy`: 2
  - `false_premise`: 2
- candidates per item: 4
- target words: 60
- mean within-item word-count gap: 0.00
- max within-item word-count gap: 0

Operational finding:

- `gemini-2.0-flash` generation was effectively unusable here because of
  immediate repeated HTTP 429 responses
- switching the builder to `gemini-flash-lite-latest` plus checkpoint/resume
  support produced a usable frozen pilot artifact

Fresh disagreement packet:

- `direct_category_axis` versus `length`: 6 disagreements
- versus `random`: 7 disagreements
- versus `sentiment`: 7 disagreements
- versus `refusal_heuristic`: 5 disagreements
- total fresh blind-review packet rows: 25

Strong-model selection status:

- Gemini embedding selection on the same fresh snapshot was started
- the embedding endpoint repeatedly hit HTTP 429 during candidate embedding
- no completed Gemini selection artifact was saved yet on this fresh pool

### Interpretation

This cycle materially improves the repo's open-ended research surface.

What it shows:

- the project no longer depends only on the old inherited length-biased
  candidate pool
- a fresh strict-length open-ended pool can be built under the actual laptop +
  free-tier constraints
- that fresh pool already creates real selector disagreements worth blind
  judging

What it does not show:

- that the strong embedding backend already wins on this fresh pool
- that the local Snowflake backend wins on blind review
- a decisive partner-grade intervention claim

### Decision

Promote this cycle as the creation of the missing clean open-ended intervention
surface, with a frozen partial pilot already saved.

### Fallback conditions

If repeated future runs show that exact-length building is too operationally
fragile even with checkpointing and resume, the repo should explicitly fall
back to tightly pre-registered length bins rather than silently loosening the
standard.

If Gemini embedding selection remains repeatedly quota-blocked on the fresh
pool, keep that framed as an operational constraint and continue using local
selectors only as exploratory scaffolding rather than as the main thesis test.

### Next steps

1. Finish the remaining pilot prompts with the resume-enabled builder.
2. Save the first completed Gemini embedding selection run on the fresh pool.
3. Complete blind review on the fresh pool rather than on the old Cycle 001
   inherited pool.
4. Only after that, decide whether to scale prompt count or add neutral
   critique/decomposition scoring.

## June 27, 2026 - Cycle 012 Follow-up: Judge Hardening And Unlabeled Summary

### What was done

Improved two operational weak points exposed by the fresh-pool pilot:

1. rewrote `scripts/judge_pairwise_with_gemini.py` to use direct HTTP Gemini
   calls with incremental writes and `--resume`;
2. updated `scripts/run_cycle001_intervention.py` so unlabeled open-ended runs
   report method-disagreement counts instead of meaningless `0/0` proxy-hit
   tables.

Regenerated the local Snowflake summary on:

- `notes/research_cycles/cycle_012_length_controlled_openended_pool/generated_pool_v1/pilot_snapshot_8.json`

### Key results

Fresh unlabeled summary now reports direct disagreement structure:

- `direct_category_axis` vs `length`: 6 disagreements
- vs `random`: 7 disagreements
- vs `sentiment`: 7 disagreements
- vs `refusal_heuristic`: 5 disagreements

Judge-path status:

- new judge script compiles and supports checkpoint/resume
- the first fresh-pool smoke run still hit HTTP 429 before the first completed
  row was saved

### Interpretation

This follow-up improves the honesty and resilience of the open-ended lane
without overstating what was learned from quota-blocked runs.

The important separation is now clearer:

- disagreement structure on the fresh pool is real and saved
- blind adjudication on that pool is operationally blocked by quota, not by a
  broken script or bad packet format

### Decision

Keep the hardened judge path and use it as the default for future fresh-pool
blind review attempts.

### Next steps

1. Resume fresh-pool blind judging with the hardened path when quota allows.
2. Resume Gemini embedding selection on the same fresh pool.
3. Keep local selectors as exploratory scaffolding, not as the main external
   claim.

## June 27, 2026 - Cycle 013: Objective Suite V2 Scaling

### What was done

Built a deterministic larger objective reranking surface so the repo does not
have to lean on the old 8-task math and tool pilots as if they were decisive.

Added:

- `scripts/build_objective_reranking_suite_v2.py`
- `experiments/research_system_v1/benchmarks/objective_math_reranking_v2.json`
- `experiments/research_system_v1/benchmarks/tool_interpretation_reranking_v2.json`
- `experiments/research_system_v1/benchmarks/objective_suite_v2_build_summary.json`
- `notes/research_cycles/cycle_013_objective_suite_v2_scaling/`

The builder freezes:

- 48 exact-graded math tasks
- 32 exact-graded tool-interpretation tasks
- randomized candidate order before `C1` / `C2` / `C3` assignment
- low within-item length gaps instead of the old tiny fixed-order pilot sets

Also ran exploratory local OSS verification on the new frozen suites with:

- `BAAI/bge-base-en-v1.5`
- `sentence-transformers/all-mpnet-base-v2`

### Key results

Build summary:

- math mean within-item word gap: 1.08
- tool mean within-item word gap: 1.56
- math correct-slot distribution: `C1=15`, `C2=15`, `C3=18`
- tool correct-slot distribution: `C1=8`, `C2=13`, `C3=11`

Objective math v2:

- random: 47.9%
- length: 35.4%
- BGE-base best direct: 29.2%
- MPNet-base best direct: 35.4%

Tool interpretation v2:

- random: 37.5%
- length: 50.0%
- BGE-base best direct: 43.8%
- MPNet-base best direct: 28.1%

Operational note:

- direct Colab MCP transport still timed out after browser approval
- Chrome-approved live Colab notebook fallback is usable

### Interpretation

This is the kind of negative result the repo needed.

The objective lane is now materially more serious because:

- it scales without blind reviewers;
- it keeps exact final grading;
- it removes the fixed-slot weakness of the older tiny suites.

On that stronger surface, cheap OSS embedders remain weak or anti-correlated.
That does not prove the strong-model story, but it does make the capacity-ladder
hypothesis more meaningful than before.

### Decision

Treat the v2 objective suites as the new default surface for future
capacity-ladder work.

Do not present the old 8-task objective wins as decisive evidence anymore.

### Next steps

1. Run `gemini-embedding-2` on the same v2 suites when quota allows.
2. Use the working Chrome-approved Colab notebook surface to test at least one
   stronger OSS embedder on the same frozen v2 files.
3. Update the serious research report so the old objective v1 wins are framed
   as pilot evidence and v2 becomes the new confirmatory target.

## Cycle 014 - Local tree/scoring/correlation experiments

Date: 2026-06-28

Ran three local CPU experiments from `methodology/EXPERIMENT_SPECS.md` using the
project venv and the required three local models:

- `snowflake/snowflake-arctic-embed-m`
- `BAAI/bge-m3`
- `nomic-ai/nomic-embed-text-v1.5`

All runs used both battery splits separately:

- original 50-case battery:
  `notes/research_cycles/cycle_002_potential_shaping/controlled_evaluative_axis_battery_v3_50_cases.jsonl`
- warmth 20-case battery:
  `notes/research_cycles/battery_rebalancing/warmth_cases.jsonl`

All response embeddings used the required framing:

```text
User: {prompt}
Assistant: {response}
```

### Experiments run

- E-01 Tree Decomposition of "Good":
  `scripts/run_tree_decomposition.py`
- E-04 Bipolar vs Single-Pole Scoring:
  `scripts/run_scoring_methods.py`
- E-08 Same-Branch vs Cross-Branch Correlation:
  `scripts/run_branch_correlation.py`

### Outputs

- `notes/research_cycles/tree_decomposition/tree_results.json`
- `notes/research_cycles/scoring_methods/scoring_results.json`
- `notes/research_cycles/branch_correlation/branch_results.json`

The three JSON outputs were parsed successfully after the run.

### Execution notes

- Dependencies were already installed:
  `sentence-transformers`, `numpy`, `einops`
- No Colab, Gemini, or paid API path was used.
- Hugging Face emitted an unauthenticated-request warning.
- BGE-M3 emitted the existing `get_extended_attention_mask` transformer
  deprecation warning, but the runs completed.
- Each script loads models with `trust_remote_code=True` and deletes each model
  between runs.

### Interpretation

No research conclusion is recorded in this entry. This cycle records the raw
measurements only; the saved JSON files contain the full per-model, per-split
tables and per-case margins/correlations.

### Next steps

Inspect the three JSON outputs before updating `RESEARCH_DIRECTIONS.md`,
`paper/draft.md`, or any external-facing summary.

## Cycle 015 - Anchor formats and diagnostic score profiles

Date: 2026-06-28

Ran two local CPU experiments using the project venv and the required three
standard local models:

- `snowflake/snowflake-arctic-embed-m`
- `BAAI/bge-m3`
- `nomic-ai/nomic-embed-text-v1.5`

All runs used both battery splits separately:

- original 50-case battery:
  `notes/research_cycles/cycle_002_potential_shaping/controlled_evaluative_axis_battery_v3_50_cases.jsonl`
- warmth 20-case battery:
  `notes/research_cycles/battery_rebalancing/warmth_cases.jsonl`

All response embeddings used:

```text
User: {prompt}
Assistant: {response}
```

### Experiments run

- Anchor Format Variations:
  `scripts/run_anchor_format_variations.py`
- Diagnostic Score Profiles:
  `scripts/run_diagnostic_score_profiles.py`

### Outputs

- `notes/research_cycles/anchor_formats/format_results.json`
- `notes/research_cycles/diagnostic_profiles/profile_results.json`

Both JSON outputs were parsed successfully after the run and each contains
results for all three required models.

### Execution notes

- Dependencies were already installed.
- No Colab, Gemini, or paid API path was used.
- Hugging Face emitted an unauthenticated-request warning.
- BGE-M3 emitted the existing `get_extended_attention_mask` transformer
  deprecation warning, but the runs completed.
- Each script loads models with `trust_remote_code=True` and deletes each model
  between runs.

### Interpretation

No research conclusion is recorded in this entry. This cycle records the raw
measurements only.


## Cycle 015 - Battery expansion, complementarity, word length

Date: 2026-06-28

Ran three local CPU experiments from `methodology/EXPERIMENT_SPECS.md`
(E-05, E-07, E-03) using the project venv and the required three local
models (`snowflake/snowflake-arctic-embed-m`, `BAAI/bge-m3`,
`nomic-ai/nomic-embed-text-v1.5`). All runs used both battery splits
separately and the required `User:/Assistant:` framing.

### Experiments and outputs

- E-05 Battery Expansion:
  - 20 new cases written across four categories, length-balanced within 20%:
    `notes/research_cycles/battery_expansion/nuance_context.jsonl`,
    `factual_accuracy.jsonl`, `conciseness_completeness.jsonl`,
    `creative_quality.jsonl`.
  - Scorer: `scripts/run_battery_expansion.py`
  - Results: `notes/research_cycles/battery_expansion/expansion_results.json`
  - All 23 existing axes (5 ML-jargon + 18 single-word) scored on the new
    battery and compared to their firmness/warmth accuracy.

- E-07 Per-Case Complementarity:
  - Scorer: `scripts/run_complementarity.py`
  - Results: `notes/research_cycles/complementarity/complementarity_results.json`
  - 15 axes, per-case correctness vectors, pairwise correlation matrix,
    greedy + exhaustive minimum covering sets, best sum-of-projections
    combo at each k.

- E-03 Word Length and Phrase Complexity:
  - Scorer: `scripts/run_word_length.py`
  - Results: `notes/research_cycles/word_length/length_results.json`
  - 4 concepts x 6 length variants scored on the balanced battery.

### Key results

E-05 (expansion battery, 20 cases):

- Only ONE axis clears 55% on all three models on the new battery:
  `word/careful` (Snowflake 80%, BGE-M3 60%, Nomic 65%). This is
  consistent with prior findings that `careful` is the most cross-model
  robust single-word axis.
- Several axes that previously worked on the balanced battery break on
  the expansion battery. The largest drops (expansion vs balanced mean):
  - `ml/anti_sycophancy` on BGE-M3: 25% vs 57% (-32 pts)
  - `word/useful` on BGE-M3: 25% vs 54% (-29 pts)
  - `word/kind` on Nomic: 20% vs 49% (-29 pts)
  - `ml/persona_honesty` on Snowflake: 25% vs 53% (-28 pts)
  The targeted ML-jargon axes, which dominated the firmness battery,
  collapse on nuance and factual cases.
- Per-category pattern: the nuance/context category is the hardest. Many
  axes score 0-20% there. Factual and creative are more recoverable.
- Interpretation: the targeted axes are not universal evaluators. They are
  strong on the conflict dimensions they were written for and weak on
  reading implicit context. The single-word `careful` is the most robust
  generalist but still only ~65% cross-model.

E-07 (complementarity, 70-case balanced battery):

- `hard` is the recurring complementarity hub. It is negatively correlated
  with warmth-axis terms on all three models:
  - `good x hard`: r = -0.33 / -0.21 / -0.24 (3/3 models)
  - `kind x hard`: r = -0.27 / -0.16 / -0.28 (3/3 models)
  - `hard x constructive`: r = -0.22 / -0.34 / -0.16 (3/3 models)
  This is the firmness/warmth split made geometrically explicit: `hard`
  captures the branch of quality that warmth-axis terms miss, and vice
  versa.
- Most redundant pair (positive correlation) is also cross-model stable:
  `helpful x good` r = +0.82 / +0.77 (BGE-M3, Nomic), confirming these
  access the same failed warmth-dominated direction.
- Minimum covering set (>90% OR-coverage) is small but model-dependent:
  - Snowflake: `kind, good, hard` reaches 97% at k=3.
  - BGE-M3: `kind, clear, hard` reaches 94% at k=3.
  - Nomic: `careful, kind, hard` reaches 93% at k=3.
  `hard` appears in the best k=3 set on all three models; a warmth-axis
  term (`kind` or `good`) appears alongside it on all three. This is the
  clearest empirical support yet for the scalar-plus-basis view: the
  minimal viable basis is roughly {one firmness term, one warmth term}.
- Caveat: OR-coverage is an upper bound. Actual sum-of-projections
  accuracy at k=3 only reaches ~61-67%, because summing reintroduces
  some interference. The covering set shows the axes are complementary;
  it does not yet show a clean scalar aggregation rule.

E-03 (word length):

- Clear monotonic relationship: shorter anchors are better, across all
  three models. Mean combined accuracy by length:
  - 1_word: 48%  |  2_word: 44%  |  phrase: 46%
  - clause: 42%  |  sentence: 37%  |  two_sent: 39%
- The drop is steepest at the sentence level, especially on BGE-M3 and
  Nomic (sentence/two_sent fall to 33-35%).
- `thorough` is the cleanest case: `1_word` is best on all three models
  (Snowflake 60%, BGE-M3 53%, Nomic 54%); every longer variant is worse.
- Warmth split is the opposite pattern in some cases: longer anchors
  sometimes help warmth (e.g. `careful` two_sent gets 70-75% warmth on
  two models) while hurting firmness. This again reflects the
  firmness/warmth geometry: longer, more elaborated anchors drift toward
  the warmth branch.
- Interpretation: there is no "sweet spot" at medium length. The bare
  antonym pair is consistently the strongest and most stable anchor
  form. Adding context words ("The response is...") or full sentences
  generally degrades the signal, consistent with the diffusion theory
  in paper section 5.6.

### Interpretation

The three experiments converge on the same picture the rebalancing cycle
already pointed at:

1. There is no universal single axis. Every targeted axis is strong on
   some dimensions and breaks on others (E-05).
2. The structure of quality in embedding space is dominated by a
   firmness-vs-warmth axis split, with `hard` on one pole and warmth
   terms (`good`, `kind`, `constructive`) on the other (E-07).
3. Anchor form should stay minimal: the bare antonym word pair
   outperforms phrases, clauses, and sentences (E-03).

The minimal viable basis implied by E-07 (one firmness term + one warmth
term, scored independently) is the most concrete design recommendation
the local-model evidence has produced so far. It is still not strong
enough for training claims and does not generalize cleanly to a scalar
sum, but it is a specific, falsifiable target for the next cycle.

### Decision

- Record `word/careful` as the most cross-model-robust single-word axis
  on the expanded battery, with the caveat that 65% is not decisive.
- Record the {firmness, warmth} pair finding as the leading basis
  hypothesis, to be retested on a frontier model (Gemini/Jina) where
  individual axes already reach 86-98%.
- Do not promote any expansion-battery result as a positive finding
  without frontier-model replication, because local-model accuracy
  remains near chance on the hardest category (nuance/context).

### Next steps

1. Replicate E-05 and E-07 on `gemini-embedding-2` when quota allows, to
   see whether the firmness/warmth complementarity structure survives at
   frontier accuracy.
2. Test a second frontier embedder (Jina-v3 free API or Qwen3-Embedding)
   to break the n=1 frontier dependency.
3. Consider whether the minimal {firmness, warmth} basis, scored as a
  sum-of-projections on a frontier model, beats single-axis scoring on
  the objective reranking suites.


## Cycle 016 - Cosine-to-positive scoring and compound anchors

Date: 2026-06-28

Two local CPU experiments testing the scoring method and anchor form
questions raised by the prior E-04 / E-01 cycles. Both used the project
venv, the three required models, both battery splits, and the
`User:/Assistant:` framing.

### Experiments and outputs

- Cosine-to-Positive Systematic Test:
  - `scripts/run_cosine_positive.py`
  - `notes/research_cycles/cosine_positive/cosine_positive_results.json`
  - All 15 single-word axes, bipolar vs cosine-to-positive, plus a
    per-axis-optimal covering-set test.
- Cross-Concept Compound Anchors:
  - `scripts/run_compound_anchors.py`
  - `notes/research_cycles/compound_anchors/compound_results.json`
  - 10 compound anchor strings vs their component single words summed
    independently (the dilution test), scored under both methods.

### Key results

Cosine-to-positive vs bipolar:

- Cosine-to-positive wins on 14 of 15 axes when judged cross-model on
  combined accuracy. The only axis where bipolar consistently wins is
  `hard` (0/3 models for cosine). This is a real, actionable scoring-method
  finding: the negative anchor mostly hurts, not helps, on the warmth
  branch.
- The wins are large on BGE-M3: 8 axes improve by >=10pts under cosine
  (`clear` +23, `fair` +24, `helpful` +14, `good` +16, `wise` +13,
  `responsible` +13, `kind` +11, `constructive`/`supportive` +11).
- The single exception, `hard`, is the firmness axis. Bipolar helps when
  the axis is already pointing at the firmness branch (where the negative
  pole `Soft` pulls the right way); cosine-to-positive helps on warmth
  axes where the negative pole is closer to the wrong branch.
- Per-axis-optimal method selection improves the 3-axis covering set
  materially on BGE-M3 and Nomic:
  - BGE-M3: hard(bip) + kind(cos) + careful(cos) = 69% vs 60% bipolar-all.
  - Nomic: hard(bip) + kind(cos) + careful(bip) = 66% vs 53% bipolar-all.
  - Snowflake: ~unchanged (60% vs 59%).
- Interpretation: cosine-to-positive is a legitimate improvement on the
  warmth branch, not a one-off artifact of the original E-04 three-axis
  test. The recommended default becomes: use cosine-to-positive for warmth
  axes, bipolar for firmness axes. This is still local-model evidence
  (max ~70%), so it should be replicated on a frontier embedder before any
  external claim.

Compound anchors:

- Combining two terms from different branches into one anchor string
  DILUTES signal on the two larger models. 7 of 10 compounds dilute on
  BGE-M3 and Nomic under cosine (the better method from Exp 1).
  `careful_kind` drops -17% (BGE-M3) / -9% (Nomic); `clear_fair` drops
  -14% / -16%; `careful_kind_honest` drops -23% / -7%.
- This is the same failure mode as averaging axis vectors: putting both
  terms in one string recreates a diffuse centroid. The compound does not
  create a super-additive axis.
- NO compound beats its best single component on >=2 models. The
  independent-sum baseline (score each word separately, add scores)
  consistently beats the compound string on BGE-M3 and Nomic.
- Snowflake is the exception: compounds slightly help there (+3-6%), but
  this is within noise and Snowflake is the weakest model, so it does not
  overturn the larger-model finding.
- Interpretation: this confirms the no-averaging rule extends to anchor
  text itself. Multi-word anchor strings that span branches behave like
  composites, not like independent axes. The independent-sum approach
  remains the correct way to combine terms from different branches.

### Decision

- Promote cosine-to-positive as the default scoring method for warmth-axis
  terms, with bipolar retained for firmness-axis terms. Record the
  per-axis split (cos for warmth, bip for firmness) as a concrete,
  testable scoring recipe.
- Confirm the no-averaging rule now covers anchor text: compound
  multi-branch anchor strings dilute. Do not use them; score independently
  and sum.
- Do not raise any external claim from these local-model results without
  frontier-model replication. Both findings are operational improvements
  at the ~60-70% local-model accuracy level, not proof of a universal
  evaluative signal.

### Next steps

1. Replicate the cosine-vs-bipolar split on `gemini-embedding-2` when
   quota allows, since Gemini already reaches 86-98% on targeted axes and
   the cosine-vs-bipolar distinction may behave differently at frontier
   accuracy.
2. Re-run the 3-axis covering-set search (E-07) using per-axis-optimal
   methods, to see whether the minimal viable basis improves under the
   new scoring recipe.
3. Consider whether the cosine-vs-bipolar asymmetry is itself a diagnostic
   of which branch an axis belongs to (warmth axes prefer cosine,
   firmness axes prefer bipolar).


## Cycle 017 - Large word sweep and score-subtraction: exploratory

Date: 2026-06-28

Two exploratory experiments with no premature conclusions. Methodology
shifted from "pick winners by discrimination rate" to "test many words and
learn the structure." All three local models, full battery (105 cases:
firmness 50 + warmth 20 + 4 expansion categories + anti-sycophancy 15).

### Experiments and outputs

- Large word discrimination sweep (129 words):
  - `scripts/run_large_word_sweep.py`
  - `notes/research_cycles/large_word_sweep/sweep_results.json`
  - Each word scored cosine-to-positive AND bipolar, across all 7 categories.
  - INCLUDED 10 random control words (blue, however, therefore, etc.) to
    establish a measured chance floor per model.
- Score-subtraction sweep (6 bases x 13 penalties x 7 alphas):
  - `scripts/run_score_subtraction_sweep.py`
  - `notes/research_cycles/score_subtraction/subtraction_results.json`
  - Tests the tree-penalty idea: reward = cos(base) - alpha*cos(penalty),
    with score subtraction (not vector subtraction).

### Findings

1. THE CHANCE FLOOR IS HIGH AND MODEL-VARYING.

   This is the most important methodological finding. The measured random-
   word discrimination floor differs sharply by model:
     - Snowflake: random words discriminate at ~63% combined
     - BGE-M3:     ~59%
     - Nomic:      ~44%
   On Snowflake a RANDOM word gets 63%. This means most prior "this word
   scored well" results on Snowflake were within noise of a random word.
   This is the structural reason predictions kept failing: the table was
   mostly noise, and we read signal into chance.

2. ONLY ~11% OF WORDS BEAT THE FLOOR CONSISTENTLY ACROSS ALL 3 MODELS.

   Of 119 non-random words, only 13 beat the random floor on combined
   accuracy on all three models. The semantically-coherent quality words
   among them: careful, restrained, measured, fair, patient, gentle.
   "heavy" does NOT survive this filter (it was a local-model artifact).

3. RANDOM CONTROL WORDS THEMSELVES SOMETIMES BEAT THE FLOOR.

   'blue' and 'northern' both beat the floor on all 3 models. This is a
   red flag: if random color/direction words discriminate above chance,
   either the battery has a length/fluency bias any word picks up, or the
   floor estimate is noisy. Either way, "beats floor" is necessary but
   not sufficient for real signal.

4. NO WORD SOLVES ANTI-SYCOPHANCY CONSISTENTLY ACROSS ALL 3 MODELS.

   The category we most want a signal for (refusing to validate bad plans)
   has no single-word solution on local models. The best words on BGE-M3
   and Nomic are 'misleading' and 'careful' (both 80-93%), but these
   collapse on Snowflake (27%). This is the clearest evidence that local
   embedding models cannot detect sycophancy from single-word anchors.

5. SCORE SUBTRACTION HAS A CONTAMINATION PROBLEM.

   The tree-penalty idea (reward = cos(good) - alpha*cos(flattering))
   showed apparent gains, but a CONTROL reveals the problem: on Nomic,
   subtracting 'therefore' (a random word) gives +32% while subtracting
   'flattering' (the hypothesized semantic penalty) gives +21%. If a
   random word outperforms the semantic penalty, the gain is a magnitude/
   calibration artifact, not semantic decontamination. The semantic
   penalty only convincingly helps on BGE-M3, the one model where the
   'good -> pleasant -> flattering' contamination path was cleanest.

### Interpretation (tentative)

- The varying chance floor is the key new fact. It reframes prior results:
  many "winning" words were noise. Local models, especially Snowflake,
  have a high baseline where any word discriminates at ~60%.
- 'careful' remains the only quality word that consistently beats floor
  across models AND has a coherent semantic story. But it does not solve
  anti-sycophancy alone.
- The score-subtraction tree idea is NOT validated by this sweep because
  random controls beat the semantic penalties on Nomic. The mechanism
  cannot be confirmed while controls outperform targets.
- The fact that 'misleading' (a negative concept) is the single best word
  on BGE-M3 and Nomic anti-sycophancy (80-93%) is notable and was not
  predicted. It may be worth examining as a penalty signal, but only
  after the control problem is addressed.

### Decision

- Do not promote any word or combination as a "golden rule" from this
  cycle. The chance-floor analysis shows most prior results were noisier
  than reported.
- Record the chance floor as a required control for all future sweeps.
  Any word-level claim must beat the measured random-word floor on
  >=2 of 3 models.
- The score-subtraction tree idea remains untested properly. The control
  failure on Nomic means a new method is needed that separates semantic
  penalty effect from calibration artifact.

### Next steps

1. Run the per-case complementarity analysis properly: re-score per-case
   correctness vectors for the top ~20 words so we can find words that
   catch each others' SPECIFIC failures (not just category averages).
2. Investigate why random words discriminate above chance on Snowflake
   (length bias? fluency bias? floor-estimation noise?).
3. Re-test score subtraction with a magnitude-controlled penalty so the
   semantic effect can be isolated from calibration artifacts.


## Cycle 018 - Bootstrap variance: most findings are within noise

Date: 2026-06-28

The user correctly challenged the variance assumption: "13% on one model,
80% on another" is exactly what wide variance looks like. Ran a 2000-
iteration bootstrap (resample cases with replacement) to get 95% CIs on the
15 axes for which per-case margins were stored (diagnostic_profiles).

Output: `notes/research_cycles/bootstrap_variance/bootstrap_results.json`

### Finding: uncertainty bands are huge; most prior structure is noise

CI widths at n=50 (firmness): typically 26-28 points.
CI widths at n=20 (warmth): typically 35-40 points.

This means "55% vs 65%" is indistinguishable at 95% confidence on these
sample sizes. Most of the apparent per-word, per-model structure in the
large word sweep (Cycle 017) cannot be distinguished from noise.

### What survives the bootstrap (CI excludes 50%, or cross-model CIs disjoint)

Firmness (n=50):
- `good` genuinely fails below chance on BGE-M3 [6%, 26%] and Nomic
  [4%, 22%]. Real, robust, cross-model.
- `hard` genuinely works above chance on BGE-M3 [52%, 78%] and Nomic
  [54%, 80%]. Real, robust.
- A warmth cluster (`helpful`, `fair`, `responsible`) shows genuinely
  disjoint CIs between Snowflake and BGE-M3/Nomic on firmness — they
  collapse on the larger models. Real.

Warmth (n=20):
- Several words have CIs excluding 50% upward: `helpful` [75%, 100%],
  `responsible` [60%, 95%], `good` [60%, 95%] on Nomic. Warmth has real
  signal, though wider bands.

### What does NOT survive (and was previously reported as signal)

- `careful` on firmness: CIs [38%, 66%] / [44%, 72%] / [48%, 76%]. ALL
  overlap 50%. The "careful is the robust competence word" claim is within
  noise on this battery size. Not established.
- `honest` is anti-signal (below chance): CIs exclude 50%, so honest does
  genuinely perform below chance. BUT the cross-model CIs all overlap
  each other, so the "13% vs 30%" framing is itself noise.
- The large word sweep (Cycle 017) constellation/inversion claims: CANNOT
  be bootstrapped because per-case data was not stored. Every word-level
  finding from that sweep is an uncheckable point estimate.

### Interpretation

The real bottleneck is sample size, not embedding model or word choice.
50 + 20 cases cannot resolve differences below ~15 points. This reframes
the entire project's local-model evidence base: most of the per-word
discrimination patterns that looked like signal are statistically
indistinguishable from noise at 95% confidence.

The few robust facts:
- Warmth has real signal (warmth words work on warmth cases, robustly).
- `good` genuinely fails on firmness (below chance, robust).
- `hard` genuinely works on firmness (above chance, robust).
- Warmth-words genuinely fail on firmness on the larger models.

Almost everything else - careful, honest, the praise-word inversion, the
constellation - is within noise on the current batteries.

### Decision

- From this point, NO word-level claim is made without a 95% CI.
- The large word sweep must be re-run with per-case margins stored, so
  every result is bootstrappable.
- The battery is the bottleneck. Either (a) generate more cases to narrow
  the CIs, or (b) accept that local-model word-level discrimination has
  a ~15-point resolution floor and report only effects larger than that.

### Next steps

1. Re-run the large word sweep storing per-case margins, then bootstrap
   the candidate constellation (careful + kind + failure-mode penalties)
   to see if it survives.
2. Consider whether the expansion batteries (nuance/factual/creative, n=5
   each) are usable at all - with n=5, CIs will be ~50 points wide, so
   almost nothing is resolvable.
3. The honest state: on local models with these batteries, the only
   robust findings are the warmth-signal and the good-fails-on-firmness
   pattern. Everything finer-grained needs more data or a stronger model.
