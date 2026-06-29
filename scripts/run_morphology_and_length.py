#!/usr/bin/env python3
"""Two questions in one run:

1. MORPHOLOGY: does word form (warm vs warmth vs warmly) change discrimination?
   Tests adjective/noun/adverb/comparative forms of 6 key concepts.

2. LENGTH THEORY: does response length explain the high random-word floor?
   If random words discriminate at 60% because of length, then:
   - better responses should be systematically longer/shorter than worse
   - a pure length signal should match the random-word floor
   - controlling for length should drop random words toward 50%

The length theory is the one generalization that could explain the variance.
"""

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
ORIG = ROOT / "notes" / "research_cycles" / "cycle_002_potential_shaping" / "controlled_evaluative_axis_battery_v3_50_cases.jsonl"
WARM = ROOT / "notes" / "research_cycles" / "battery_rebalancing" / "warmth_cases.jsonl"
MODELS = ["snowflake/snowflake-arctic-embed-m", "BAAI/bge-m3", "nomic-ai/nomic-embed-text-v1.5"]

# Morphology variants
MORPH = {
    "warm":   ["warm", "warmth", "warmer", "warmly"],
    "good":   ["good", "goodness", "better", "well"],
    "careful":["careful", "carefully", "carefulness"],
    "honest": ["honest", "honestly", "honesty"],
    "kind":   ["kind", "kindly", "kindness"],
    "helpful":["helpful", "helpfully", "helpfulness"],
}
# Random words for floor reference
RAND = ["blue", "however", "therefore", "purple", "table", "running"]


def read_jsonl(p):
    return [json.loads(l) for l in Path(p).read_text(encoding="utf-8").splitlines() if l.strip()]


def framed(c, k):
    return f"User: {c['prompt']}\nAssistant: {c[k]}"


def cosine_to(embs, anchor):
    an = np.linalg.norm(anchor) + 1e-12
    rn = np.linalg.norm(embs, axis=1) + 1e-12
    return (embs @ anchor) / (rn * an)


def acc(b, w):
    m = np.asarray(b) - np.asarray(w)
    return (np.sum(m > 0) + 0.5 * np.sum(m == 0)) / len(m)


def main():
    from sentence_transformers import SentenceTransformer

    orig = read_jsonl(ORIG)
    warm = read_jsonl(WARM)
    print(f"Firmness {len(orig)}, Warmth {len(warm)}")

    all_forms = sorted({f for forms in MORPH.values() for f in forms} | set(RAND))

    for model_name in MODELS:
        ms = model_name.split("/")[-1]
        print(f"\n{'='*78}")
        print(f"MODEL: {ms}")
        print(f"{'='*78}")
        model = SentenceTransformer(model_name, trust_remote_code=True)
        def emb(texts):
            return model.encode(texts, show_progress_bar=False, convert_to_numpy=True)

        word_embs = {w: emb([w])[0] for w in all_forms}

        for split_name, cases in [("FIRMNESS", orig), ("WARMTH", warm)]:
            be = emb([framed(c, "better") for c in cases])
            we = emb([framed(c, "worse") for c in cases])
            # length signal: word count of responses
            len_better = np.array([len(c["better"].split()) for c in cases])
            len_worse = np.array([len(c["worse"].split()) for c in cases])
            len_acc = acc(len_better, len_worse)
            # embedding-norm signal (proxy for "semantic density")
            norm_b = np.linalg.norm(be, axis=1)
            norm_w = np.linalg.norm(we, axis=1)
            norm_acc = acc(norm_b, norm_w)

            print(f"\n  {split_name} (n={len(cases)})")
            print(f"  LENGTH signal (word count): better-longer wins {len_acc:.0%}")
            print(f"  EMBED-NORM signal: higher-norm wins {norm_acc:.0%}")
            print(f"  {'word':<16s} {'disc':>6s}")
            rand_accs = []
            for w in RAND:
                a = acc(cosine_to(be, word_embs[w]), cosine_to(we, word_embs[w]))
                rand_accs.append(a)
            print(f"  {'RANDOM floor':<16s} {np.mean(rand_accs):.0%}  (range {min(rand_accs):.0%}-{max(rand_accs):.0%})")
            print(f"  --- morphology variants ---")
            for concept, forms in MORPH.items():
                accs = {}
                for f in forms:
                    a = acc(cosine_to(be, word_embs[f]), cosine_to(we, word_embs[f]))
                    accs[f] = a
                spread = max(accs.values()) - min(accs.values())
                s = "  ".join(f"{f}={a:.0%}" for f, a in accs.items())
                flag = " <-- spread>15pp" if spread > 0.15 else ""
                print(f"  {concept:<16s} {s}{flag}")

        del model

    # Now the length-correlation analysis across the full picture
    print(f"\n{'='*78}")
    print("LENGTH THEORY CHECK: does length explain the random floor?")
    print(f"{'='*78}")
    all_cases = orig + warm
    len_b = np.array([len(c["better"].split()) for c in all_cases])
    len_w = np.array([len(c["worse"].split()) for c in all_cases])
    better_longer = np.sum(len_b > len_w)
    worse_longer = np.sum(len_w > len_b)
    equal = np.sum(len_b == len_w)
    print(f"  Across {len(all_cases)} cases:")
    print(f"    better response is LONGER: {better_longer} ({better_longer/len(all_cases):.0%})")
    print(f"    worse response is LONGER:  {worse_longer} ({worse_longer/len(all_cases):.0%})")
    print(f"    equal length:              {equal}")
    print(f"\n  If better responses are systematically longer, that ALONE")
    print(f"  would give any word ~{max(better_longer,worse_longer)/len(all_cases):.0%} discrimination via length-halo.")


if __name__ == "__main__":
    main()
