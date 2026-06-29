# Codex Goal: Centroid Direction Test Battery

## Your Role

You are the researcher/implementer. Claude (the project manager) has designed
the test regimen. Your job is to run the tests, analyze results, and report
findings honestly. Do not editorialize or hype — report what the numbers say.

## Master Spec

Read `methodology/CENTROID_METHOD_SPEC.md` first. It defines the method,
documents all existing evidence, and specifies every test you need to run.

## Environment

- Python venv: `C:\Users\93rob\.cache\codex-embedding-venv\Scripts\python.exe`
- Windows 11, PowerShell
- 32GB RAM, no GPU, CPU-only embedding
- Only Google API key available (in `.env.local`), frequently quota-limited
- `datasets` library (5.0) is installed for HuggingFace dataset access
- `sentence_transformers` is installed for local embedding models

## CRITICAL RULES

1. **DO NOT delete any files.** No scripts, no result JSONs, nothing. Everything
   is evidence. You may create new files only.
2. **DO NOT reorganize the repo.** No renaming, no moving files.
3. **All three models** on every test: snowflake-arctic-embed-m, bge-m3, nomic-embed-text-v1.5
4. **Save all results** as JSON to `notes/research_cycles/centroid_deep/`
5. **Save all scripts** to `scripts/`
6. **Free RAM** between models: `del model; gc.collect()`
7. **User/Assistant framing**: Always embed responses as "User: {prompt}\nAssistant: {response}"
8. **UTF-8 encoding**: Use `sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')`

## Tests Already Completed (DO NOT re-run)

Check `notes/research_cycles/centroid_deep/` for all existing results.

- **TEST 1A**: External validation → NEGATIVE (chance on SHP/UltraFeedback)
- **TEST 2A**: Vocabulary projection → CLAIM HOLDS (no word > 0.30)
- **TEST 3A**: PCA dimensionality → 51 PCs for 80%, centroid >> PCA
- **TEST 4A**: Margin-length → NOT A CONFOUND (inconsistent across models)
- **TEST 6A-B**: Gameability → NOT GAMEABLE (1.9% flip rate)
- **TEST 1C**: Additional models → MIXED (gte 65.6%, e5 72.1%, mxbai 70.5%, all fail permutation)
- **META-VALIDITY**: Five tests complete → METHOD IS SOUND
  - Response-only BEATS full-format on 2/3 models (+11.4pp Snowflake, +8.2pp Nomic)
  - Label flip perfectly symmetric, LOO very stable
  - Absolute scoring meaningless — pairwise only
  - Warmth influence asymmetry is Snowflake-specific (p=0.0002)

## More Completed Tests (DO NOT re-run)

- **TEST 6C**: Score diversity → TOLERATES DIVERSITY (between > within variance, ratio 1.8-2.7x, formal/technical penalized)
- **TEST 2B**: Phrase projection → CLAIM HOLDS (no phrase > 0.30, marginal improvement over single words)

## Remaining Tests (Priority Order)

**1. Cross-Author Validation (accumulate over multiple runs)**
- Script exists: `scripts/run_cross_author_validation.py`
- Generates pairs using Gemini 2.5 Flash (20 req/day free limit)
- Accumulates to `notes/research_cycles/centroid_deep/gemini_generated_pairs.jsonl`
- Needs >= 20 pairs to analyze. Currently has 9 partial pairs.
- Run once per day to accumulate, then analyze when >= 20 pairs available

## After Running

For each test, write a short analysis at the end of the script output:
- What the numbers mean
- Whether they support or threaten the centroid method's claims
- Any surprises or concerns

Save a summary of all results to `notes/research_cycles/centroid_deep/test_summary.md`.

## Script Template

Use the template from `methodology/CENTROID_METHOD_SPEC.md` Part 5.
