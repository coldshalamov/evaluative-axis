# Good vs Proxy Conflict Sweep

Date: June 27, 2026

Battery: `controlled_evaluative_axis_battery_v3_50_cases.jsonl`

This sweep maps whether raw `good/bad` and nearby proxy-word axes recover
any broad evaluative behavior on the frozen 50-case conflict battery for
the local models we can actually run on this laptop.

## Key Metrics

| Model | Status | raw good/bad | sentence good/bad | proxy mean | best proxy | best proxy acc | raw minus proxy mean |
| --- | --- | ---: | ---: | ---: | --- | ---: | ---: |
| `BAAI/bge-small-en-v1.5` | ok | 14.0% | 12.0% | 15.8% | `raw_useful_useless` | 26.0% | -1.7% |
| `BAAI/bge-base-en-v1.5` | ok | 28.0% | 24.0% | 21.8% | `raw_honest_dishonest` | 30.0% | 6.3% |
| `thenlper/gte-base` | ok | 22.0% | 20.0% | 17.2% | `raw_honest_dishonest` | 24.0% | 4.8% |
| `snowflake/snowflake-arctic-embed-m` | ok | 48.0% | 36.0% | 33.8% | `raw_helpful_unhelpful` | 58.0% | 14.2% |
| `jinaai/jina-embeddings-v2-small-en` | ok | 20.0% | 26.0% | 20.5% | `raw_useful_useless` | 28.0% | -0.5% |
| `jinaai/jina-embeddings-v2-base-en` | ok | 16.0% | 18.0% | 21.0% | `raw_helpful_unhelpful` | 32.0% | -5.0% |
| `nomic-ai/nomic-embed-text-v1.5-Q` | ok | 20.0% | 18.0% | 25.2% | `raw_honest_dishonest` | 34.0% | -5.2% |
| `mixedbread-ai/mxbai-embed-large-v1` | ok | 20.0% | 20.0% | 21.5% | `raw_honest_dishonest` | 32.0% | -1.5% |

## Interpretation Rule

Treat this as geometry evidence, not training proof.
If raw `good/bad` stays weak across the local model family as well, that
supports the narrower claim that current usable signal comes from richer
targeted axes rather than from the minimalist broad-word axis.
