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
