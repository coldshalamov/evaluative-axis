# Research Context for New Sessions

Read this before doing anything. It contains the hard-won lessons from
dozens of experiments. Ignoring these will waste time rediscovering things
we already know.

---

## What this project is

We're testing whether projecting text onto evaluative directions in embedding
space can serve as a cheap alignment signal for LLM training. Instead of
expensive human preferences or LLM judges, define a "good/bad" direction from
anchor words, embed a response, take the dot product. That's the quality score.

The paper draft is at `paper/draft.md`. The open research questions are at
`methodology/RESEARCH_DIRECTIONS.md`.

## Core method

```python
# 1. Embed positive and negative anchor terms
pos_embs = model.encode(["Careful"])   # or multiple terms
neg_embs = model.encode(["Reckless"])

# 2. Compute normalized axis vector
axis = pos_embs.mean(axis=0) - neg_embs.mean(axis=0)
axis = axis / np.linalg.norm(axis)

# 3. Score a response by projecting onto the axis
response_text = f"User: {prompt}\nAssistant: {response}"
response_emb = model.encode([response_text])
score = np.dot(response_emb[0], axis)

# 4. Pairwise comparison: higher score = "better" response
accuracy = (score_better > score_worse)
```

## Environment

- **OS**: Windows 11. Use PowerShell for running Python, not Bash.
- **Python venv**: `C:\Users\93rob\.cache\codex-embedding-venv\Scripts\python.exe`
- **API keys**: Only Google API key available (in `.env.local`). No Jina,
  Mistral, Cohere, or Voyage keys. Do not propose experiments requiring
  paid APIs we don't have.
- **Hardware**: 32GB RAM, Intel integrated graphics (2GB), NO CUDA GPU.
  All local embedding models must run on CPU.
- **Gemini quota**: Free tier, frequently exhausted (HTTP 429). Check
  before planning experiments that depend on it.
- **Budget**: Zero. The user cannot buy API credits, hire annotators, or
  access paid services.

## Local models available

These three local models are the standard test set. Always test on all three
for cross-model consistency:

```python
MODELS = [
    "snowflake/snowflake-arctic-embed-m",   # 33M params, 768d
    "BAAI/bge-m3",                           # 567M params, 1024d
    "nomic-ai/nomic-embed-text-v1.5",        # 137M params, 768d
]
```

Load via `sentence_transformers.SentenceTransformer(name, trust_remote_code=True)`.
Delete model between runs to free RAM: `del model`.

### API models

- **Gemini Embedding 2**: Best performer (86-98% on targeted axes). Google API,
  free tier, frequently quota-limited (HTTP 429). Key in `.env.local`.
- **Jina Embeddings v5**: API-based, free tier. Performs like local models,
  NOT like Gemini. Key in `.env.local`. Confirms Gemini's advantage is
  model-specific, not a property of being API-based.
  See `scripts/run_jina_v5_battery.py` for the embedder class.

## Test batteries

### Original 50-case battery (BIASED — use with caution)
`notes/research_cycles/cycle_002_potential_shaping/controlled_evaluative_axis_battery_v3_50_cases.jsonl`

WARNING: This battery is 64% firmness-biased (32/50 cases reward pushback/
firmness). Any axis that detects firmness will score well on this battery
regardless of whether it detects actual quality. ALWAYS test on the warmth
split too.

### Warmth cases (20 cases)
`notes/research_cycles/battery_rebalancing/warmth_cases.jsonl`

Cases where the better response is warm, patient, encouraging, or validating.
The worse response is cold, clinical, dismissive, or pedantic.

### Combined balanced battery (70 cases)
Concatenate both files. This is the minimum valid test. An axis must score
>50% on BOTH the original 50 AND the warmth 20 to be considered viable.

### JSONL format
```json
{"id":"...", "category":"...", "phenomenon":"...",
 "prompt":"user's message",
 "better":"the response we label as better",
 "worse":"the response we label as worse"}
```

### Embedding format for scoring
ALWAYS wrap responses as:
```
User: {case['prompt']}
Assistant: {case['better' or 'worse']}
```
This "User:/Assistant:" framing gives the embedding model context about
what it's evaluating. Omitting it degrades results.

## ML-jargon axes (multi-sentence)

Defined in `scripts/run_cycle001_intervention.py` as the `AXES` dict.
Five axes: general_evaluative, harm_reduction, truthfulness, persona_honesty,
anti_sycophancy. Each has 3 positive and 3 negative anchor sentences.

---

# PITFALLS — Read This Carefully

These are mistakes we've already made. Do not repeat them.

## 1. Battery firmness bias

The original 50-case battery rewards pushback in 64% of cases. Categories
like anti_sycophancy, harm_reduction, reasoning_rigor, and truthfulness
all have "better" responses that push back, correct, refuse, or disagree.

**Consequence**: ANY axis that correlates with firmness will score well on
this battery. "Hard/Soft" scored 58-68% and we initially reported it as
a finding. It was a battery artifact — "Hard" is just a firmness detector,
and the battery rewards firmness.

**Rule**: Never report a result from the original battery alone. Always
test on the warmth split too. If an axis scores well on firmness cases
but fails on warmth cases, it's detecting firmness, not quality.

## 2. Compositing kills signal

Averaging multiple axis vectors into one composite axis ALWAYS degrades
performance. This recreates the "diffuse cloud" problem — averaging children
reconstructs the parent's unfocused direction.

**What works instead**: Score each axis independently, then SUM the scores
or use MAJORITY VOTE. This preserves the specificity of each axis.

**Rule**: Never average axis vectors. Always score independently and combine
the scores.

## 3. Frequency does not predict signal strength

"Good" is the most frequent evaluative word in English and the worst
performer. "Careful" is far less frequent and the best single-word
performer on firmness cases.

**Theory**: What matters is evaluative SPECIFICITY — whether the word
almost always appears in evaluative contexts. "Good" fails because it
appears in thousands of non-evaluative contexts (good morning, good faith,
good enough). "Careful" almost always appears in evaluative contexts.

**Rule**: Do not assume common words will work better. Test empirically.

## 4. Cross-model consistency is essential

A result that works on one model but not others is not robust. The three
local models often disagree — an axis might score 80% on one model and
40% on another.

**Rule**: Always report results on all three models. Flag results that
depend on a specific model.

## 5. Pairwise accuracy is the metric

We measure "what fraction of the time does the better response score
higher than the worse response on this axis?" 50% is chance. Below 50%
means the axis is actively pointing the wrong way.

**Important**: This is not the same as absolute score magnitude. An axis
can have small absolute scores but correct pairwise ordering, or large
absolute scores but wrong ordering.

## 6. "Good/Bad" is bipolar, not broken

"Good/Bad" scores 16% on firmness cases (inverted) but 85% on warmth
cases (correct). It's not noise — it's a strong signal pointing at the
warmth branch of quality. This is a real finding about HOW "good" works
in embedding space, not evidence that it's useless.

## 7. Discrimination vs training signal

Our battery tests whether an axis can rank two pre-written responses.
In actual training, the model generates responses and could learn to
satisfy ALL quality dimensions simultaneously. A broad term like "good"
might be a mediocre discriminator but a good training signal. Our
methodology can't distinguish these.

## 8. Don't over-theorize

Theory has outpaced evidence. There are many plausible theories about
why certain things work. The only way to resolve them is experiments.
Don't spend time writing up theories — run the test.

## 9. Paper style

When explaining results: use concrete examples from actual data, state
what the test does, what the number means, whether it's good or bad.
No jargon, no extended analogies, no unexplained percentage dumps.

## 10. The "good" tree

"Good" has thousands of child meanings (careful, honest, kind, wise...).
Each child has its own children (careful -> deliberate, attentive, precise).
A response satisfies a fraction of "good's" children but might satisfy
ALL of "careful's" children. This explains why specific terms give
cleaner signal than broad ones, AND why no single specific term covers
everything. The research program is about finding the right SET of terms
at the right level of the tree.

---

# EXPERIMENT DESIGN TEMPLATE

When creating a new experiment, follow this structure:

```python
#!/usr/bin/env python3
"""One-line description of what this tests and why.

Hypothesis: [what we expect to see and why]
Depends on: [files, prior results]
Output: [where results are saved]
"""

import json, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

# Standard batteries
BATTERY_ORIGINAL = ROOT / "notes/research_cycles/cycle_002_potential_shaping/controlled_evaluative_axis_battery_v3_50_cases.jsonl"
BATTERY_WARMTH = ROOT / "notes/research_cycles/battery_rebalancing/warmth_cases.jsonl"

# Standard models — always test all three
MODELS = [
    "snowflake/snowflake-arctic-embed-m",
    "BAAI/bge-m3",
    "nomic-ai/nomic-embed-text-v1.5",
]

def read_jsonl(path):
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows

def compute_axis(embed_fn, positive, negative):
    pos_embs = embed_fn(positive)
    neg_embs = embed_fn(negative)
    axis = pos_embs.mean(axis=0) - neg_embs.mean(axis=0)
    return axis / (np.linalg.norm(axis) + 1e-12)

def pairwise_accuracy(better_embs, worse_embs, axis):
    correct = 0
    for i in range(len(better_embs)):
        sb = float(np.dot(better_embs[i], axis))
        sw = float(np.dot(worse_embs[i], axis))
        if sb > sw:
            correct += 1
        elif sb == sw:
            correct += 0.5
    return correct / len(better_embs)

def main():
    from sentence_transformers import SentenceTransformer

    original = read_jsonl(BATTERY_ORIGINAL)
    warmth = read_jsonl(BATTERY_WARMTH)

    for model_name in MODELS:
        model = SentenceTransformer(model_name, trust_remote_code=True)
        embed_fn = lambda texts: model.encode(texts, show_progress_bar=False,
                                               convert_to_numpy=True)

        # Embed battery responses WITH User/Assistant framing
        orig_better = embed_fn([f"User: {c['prompt']}\nAssistant: {c['better']}"
                                for c in original])
        orig_worse = embed_fn([f"User: {c['prompt']}\nAssistant: {c['worse']}"
                               for c in original])
        warm_better = embed_fn([f"User: {c['prompt']}\nAssistant: {c['better']}"
                                for c in warmth])
        warm_worse = embed_fn([f"User: {c['prompt']}\nAssistant: {c['worse']}"
                               for c in warmth])

        # ... test your hypothesis here ...
        # ALWAYS report: original accuracy, warmth accuracy, combined accuracy
        # ALWAYS report: per-model results, not just averages

        del model  # free RAM before loading next model

    # Save results as JSON to notes/research_cycles/your_experiment/

if __name__ == "__main__":
    main()
```

### Checklist before declaring a result:
- [ ] Tested on all 3 models
- [ ] Tested on both firmness and warmth splits
- [ ] Used User/Assistant framing for response embedding
- [ ] Saved raw results as JSON (not just printed)
- [ ] Noted which results are above/below chance (50%)
- [ ] Checked for battery composition confounds
- [ ] Did NOT average axis vectors (scored independently if combining)
