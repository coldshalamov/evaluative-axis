# Cycle 003 Results

Date: June 25, 2026

## What This Cycle Tried To Do

This cycle asked a practical question:

> Can evaluative embedding scoring pick a better answer from several candidates
> when the final metric is objective hidden-task success rather than HH overlap
> or an LLM judge?

The landscape scan in `idea.md` ranked 20 possible next experiments under the
real project constraints: one laptop, free-tier Google access, no hired
annotators, and no assumption that HH is ground truth.

The five highest-value targets were:

1. objective code reranking with hidden tests;
2. length-controlled no-leakage open-ended reranking;
3. objective math reranking with exact-answer checks;
4. injected error/repair potential shaping;
5. Gemini-blind HH disagreement adjudication.

The top target won because it gives an objective end metric, avoids oracle
decomposition, avoids HH as truth, and can be run cheaply.

## Actual Execution

Planned top test:

- score multiple code answers per task with evaluative embeddings;
- compare direct answer scoring against critique-based scoring;
- evaluate the selected answer with hidden unit tests.

Actual execution on June 25:

- `mcp__colab_mcp.open_colab_browser_connection` returned `false`;
- a generated-candidate smoke attempt hit repeated Gemini HTTP 429 responses;
- the cycle therefore used a documented fallback:
  - local execution;
  - curated candidate sets with one correct and two plausible-wrong answers per
    task;
  - Gemini embeddings only;
  - objective hidden tests as the final metric.

This means the cycle completed the highest-value test class in a reduced form:
the direct-answer reranking slice. The critique slice remains untested because
generation quota blocked it.

## Colab OSS Ablation

After the local Gemini-embedding pilot, the same curated objective code setup
was pushed into a live Colab runtime through Chrome and rerun with open-source
embedding models using `scripts/run_objective_code_reranking_oss.py`.

Operationally, this matters:

- the Colab approval modal did exist in the browser as expected;
- approving it changed the failure mode from immediate `false` to transport
  failure on the MCP side;
- the browser-side Colab terminal still worked and was used to clone the repo,
  transfer the cycle-003 runner and candidate JSON, install
  `sentence-transformers`, and execute the OSS reranking sweep.

Models tested in Colab:

- `sentence-transformers/all-mpnet-base-v2`
- `BAAI/bge-base-en-v1.5`

### OSS Colab Results

| Model | Method | Tasks solved | Solve rate | Best-candidate hit rate | Avg selected pass rate |
| --- | --- | ---: | ---: | ---: | ---: |
| `sentence-transformers/all-mpnet-base-v2` | `random` | 3 / 6 | 50.0% | 50.0% | 83.3% |
| `sentence-transformers/all-mpnet-base-v2` | `length` | 3 / 6 | 50.0% | 50.0% | 83.3% |
| `sentence-transformers/all-mpnet-base-v2` | `direct_broad` | 1 / 6 | 16.7% | 16.7% | 83.3% |
| `sentence-transformers/all-mpnet-base-v2` | `direct_code` | 1 / 6 | 16.7% | 16.7% | 80.0% |
| `BAAI/bge-base-en-v1.5` | `random` | 3 / 6 | 50.0% | 50.0% | 83.3% |
| `BAAI/bge-base-en-v1.5` | `length` | 3 / 6 | 50.0% | 50.0% | 83.3% |
| `BAAI/bge-base-en-v1.5` | `direct_broad` | 3 / 6 | 50.0% | 50.0% | 83.3% |
| `BAAI/bge-base-en-v1.5` | `direct_code` | 3 / 6 | 50.0% | 50.0% | 83.3% |

### OSS Interpretation

This is useful because it sharpens the model-quality story:

- the local Gemini embedding pilot beat random and length on the same task set;
- a respectable open-source sentence-transformer model (`all-mpnet-base-v2`)
  failed badly on the same reranking task;
- `bge-base-en-v1.5` only matched the cheap baselines instead of beating them.

So the current evidence does not support a vague claim that "any embedding
model can do this." The result is more specific and more valuable:

> On this objective reranking pilot, the Gemini-family embedding behaved very
> differently from the cheap OSS encoders. A likely explanation is that the
> Gemini embedding model is backed by a much larger and more capable foundation
> model than typical open embedding models, even though the exact parameter
> count is not publicly stated. That supports the user's practical complaint
> that low-cost open-source embeddings often do not carry enough evaluative
> geometry for this use case.

## Pilot Setup

Files:

- `code_tasks_v1.json`
- `code_candidates_v1.json`
- `pilot_results/summary.md`
- `pilot_results/summary.json`

Task count:

- 6 pure-Python tasks

Candidates per task:

- 3

Selection methods:

- random
- longer-code baseline
- direct broad evaluative score
- direct code-quality evaluative score

Final metric:

- hidden unit-test pass rate

## Objective Pilot Results

| Method | Solve rate | Tasks solved | Avg selected pass rate | Best-candidate hit rate | Avg regret vs best |
| --- | ---: | ---: | ---: | ---: | ---: |
| `random` | 50.0% | 3 / 6 | 83.3% | 50.0% | 0.167 |
| `length` | 50.0% | 3 / 6 | 83.3% | 50.0% | 0.167 |
| `direct_broad` | 83.3% | 5 / 6 | 96.7% | 83.3% | 0.033 |
| `direct_code` | 83.3% | 5 / 6 | 96.7% | 83.3% | 0.033 |

## Failure Analysis

The direct methods missed exactly one task: `balanced_brackets`.

Why the miss matters:

- the wrong candidate was not nonsense;
- it looked algorithmic and passed 4/5 hidden checks;
- it failed the order-sensitive case `([)]`.

This is the interesting kind of failure. The embedding signal did not simply
prefer longer or nicer-looking code. It mostly picked the correct program, but
on one narrow constraint-heavy task it slightly preferred a plausible local
pattern over full structural correctness.

Score margins on the failure were tiny:

- `direct_broad`: wrong-top minus correct-second = `0.002774`
- `direct_code`: wrong-top minus correct-second = `0.004337`

That suggests a useful future policy:

- treat very small top-two margins as uncertainty;
- abstain, resample, or route those cases to a stronger interface such as blind
  critique scoring.

## Interpretation

This cycle does not prove the full thesis. It does show something important:

1. The project now has one objective intervention result where HH labels are
   not the scoring truth.
2. Broad evaluative geometry and a code-quality-flavored evaluative geometry
   both beat random and length baselines on hidden correctness in this pilot.
3. The win is directional evidence that the signal can be practically useful as
   a reranker before any weight training.
4. The miss pattern argues against overselling: the current signal can confuse
   "looks like the right kind of reasoning" with "fully respects constraint
   structure."
5. The Colab OSS ablation makes the model-quality constraint sharper: this is
   not a generic property of cheap embeddings, and weak OSS models can fail or
   wash out entirely on the same objective reranking benchmark.
6. The most plausible current story is not "embeddings never carry the signal,"
   but "cheap OSS embedders often do not, while a Gemini-family embedder likely
   does because it inherits a far more capable base model and training stack."

The strongest valid claim from this cycle is:

> On a small objective code-reranking pilot with hidden unit tests, direct
> evaluative embedding scoring selected better candidates than random or
> length-based baselines. This is directional evidence for practical reranking
> utility, not proof that the full training thesis is solved.

## What To Do Next

The cycle changes the project landscape in a useful way.

Do next:

1. Scale Target 1 from 6 tasks to at least 30-50 tasks with more adversarial
   constraint cases.
2. Re-enable the critique lane when Gemini generation quota or Colab becomes
   available.
3. Run Target 3 next: objective math reranking with exact answers.
4. Build a margin-based uncertainty rule and measure whether abstention helps.
5. Use HH only as a sensor again, not as the deciding benchmark.
6. Keep OSS reruns as a cost-quality ablation, not as the main evidence lane.

Do not do next:

- do not frame HH mismatch as a decisive negative result by itself;
- do not claim this pilot proves training will work end to end;
- do not use curated-candidate success as if it were a final benchmark.
