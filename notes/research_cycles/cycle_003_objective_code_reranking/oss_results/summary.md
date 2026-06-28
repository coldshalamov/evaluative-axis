# Objective Code Reranking OSS Colab Summary

These results were recovered from the Colab browser-terminal rerun on
June 25, 2026 and copied back into the repo so the research system can treat
them as a first-class artifact.

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
