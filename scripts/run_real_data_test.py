#!/usr/bin/env python3
"""Test anchors on a REAL preference dataset (Anthropic HH-RLHF).

This is the proper experiment:
- Real human-labeled preferences (not hand-crafted)
- n=500 pairs (not 70)
- Words, phrases, sentences, proverbs as anchors
- All three models
- Find what's STABLE across models
"""

import json, gc, random, re
from pathlib import Path
import numpy as np
from numpy.linalg import norm

CACHE_DIR = Path(__file__).resolve().parents[1] / "notes" / "research_cycles" / "real_data_test"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

MODELS = [
    "snowflake/snowflake-arctic-embed-m",
    "BAAI/bge-m3",
    "nomic-ai/nomic-embed-text-v1.5",
]

# =========================================================
# ANCHORS: words, phrases, sentences, proverbs
# =========================================================
ANCHORS = {
    # --- Single words ---
    "Good": "Good",
    "Helpful": "Helpful",
    "Honest": "Honest",
    "Careful": "Careful",
    "Thorough": "Thorough",
    "Wise": "Wise",
    "Fair": "Fair",
    "Clear": "Clear",
    "Thoughtful": "Thoughtful",
    "Respectful": "Respectful",
    "Kind": "Kind",
    "Accurate": "Accurate",
    "Responsible": "Responsible",
    "Genuine": "Genuine",
    "Insightful": "Insightful",
    "Competent": "Competent",
    "Skeptical": "Skeptical",

    # --- Negative single words (higher cos = worse, so we invert) ---
    "~Bad": "Bad",
    "~Harmful": "Harmful",
    "~Dishonest": "Dishonest",
    "~Sycophantic": "Sycophantic",
    "~Flattering": "Flattering",
    "~Misleading": "Misleading",
    "~Reckless": "Reckless",
    "~Nice": "Nice",
    "~Enthusiastic": "Enthusiastic",
    "~Pleasant": "Pleasant",

    # --- Short phrases ---
    "good response": "A good response",
    "helpful answer": "A helpful answer",
    "honest+helpful": "Honest and helpful",
    "genuinely useful": "Genuinely useful",
    "careful+thorough": "Careful and thorough",
    "bad response": "A bad response",
    "harmful answer": "A harmful and misleading answer",
    "sycophantic response": "A sycophantic response that tells the user what they want to hear",

    # --- Full sentences ---
    "helps user decide": "This response helps the user make an informed decision",
    "genuinely serves": "This is a genuinely helpful and honest response that serves the user well",
    "tells what need": "This response tells the user what they need to hear, not what they want to hear",
    "accurate+balanced": "This response is accurate, balanced, and genuinely useful",
    "misleading+flattering": "This response is misleading and flattering, telling the user what they want to hear",
    "thorough+honest": "A thorough, honest response that addresses the actual question",

    # --- Proverbs and cultural anchors ---
    "golden rule": "Do unto others as you would have them do unto you",
    "truth sets free": "The truth shall set you free",
    "open rebuke": "Better is open rebuke than hidden love",
    "honest kiss": "An honest answer is like a kiss on the lips",
    "iron sharpens": "Iron sharpens iron, so one person sharpens another",
    "soft answer": "A soft answer turns away wrath",
    "no guidance falls": "Where there is no guidance, a people falls",
    "faithful wounds": "Faithful are the wounds of a friend",
    "first do no harm": "First, do no harm",
    "know thyself": "Know thyself",
    "measure twice": "Measure twice, cut once",
}


def extract_last_turn(conversation):
    """Extract the last Human question and Assistant response from HH-RLHF format."""
    # Format: \n\nHuman: ...\n\nAssistant: ...\n\nHuman: ...\n\nAssistant: ...
    turns = re.split(r'\n\nHuman: |\n\nAssistant: ', conversation)
    turns = [t.strip() for t in turns if t.strip()]

    if len(turns) >= 2:
        # Last human turn and last assistant turn
        # In multi-turn, find the last Human and last Assistant
        human_turns = []
        assistant_turns = []
        parts = conversation.split('\n\n')
        last_human = ""
        last_assistant = ""
        for part in parts:
            part = part.strip()
            if part.startswith("Human: "):
                last_human = part[7:]
            elif part.startswith("Assistant: "):
                last_assistant = part[11:]
        return last_human, last_assistant

    return "", conversation


def load_data():
    """Load and cache a sample from Anthropic HH-RLHF."""
    cache_file = CACHE_DIR / "hh_rlhf_sample.json"

    if cache_file.exists():
        print("Loading cached dataset sample...")
        with open(cache_file, "r", encoding="utf-8") as f:
            return json.load(f)

    print("Downloading Anthropic HH-RLHF dataset...")
    from datasets import load_dataset

    ds = load_dataset("Anthropic/hh-rlhf", split="test")
    print(f"  Full test set: {len(ds)} examples")

    # Sample 500
    random.seed(42)
    indices = random.sample(range(len(ds)), min(500, len(ds)))

    pairs = []
    for idx in indices:
        example = ds[idx]
        chosen_human, chosen_assistant = extract_last_turn(example["chosen"])
        rejected_human, rejected_assistant = extract_last_turn(example["rejected"])

        # Skip if extraction failed or responses too short
        if len(chosen_assistant) < 10 or len(rejected_assistant) < 10:
            continue

        # Use the chosen prompt (should be same for both)
        prompt = chosen_human if chosen_human else rejected_human

        pairs.append({
            "prompt": prompt[:500],  # truncate long prompts
            "chosen": chosen_assistant[:1000],  # truncate long responses
            "rejected": rejected_assistant[:1000],
        })

    print(f"  Extracted {len(pairs)} valid pairs")

    with open(cache_file, "w", encoding="utf-8") as f:
        json.dump(pairs, f, indent=2)

    return pairs


def bootstrap_ci(correct_array, n_boot=5000, ci=0.95):
    n = len(correct_array)
    arr = np.array(correct_array, dtype=float)
    boot_means = np.array([np.mean(np.random.choice(arr, size=n, replace=True))
                           for _ in range(n_boot)])
    alpha = (1 - ci) / 2
    return np.percentile(boot_means, 100*alpha), np.percentile(boot_means, 100*(1-alpha))


def main():
    from sentence_transformers import SentenceTransformer

    pairs = load_data()
    n = len(pairs)
    print(f"\nDataset: {n} preference pairs from Anthropic HH-RLHF")

    np.random.seed(42)

    # Also test 10 random nonsense anchors as baseline
    RANDOM_ANCHORS = {
        "~rnd:Banana": "Banana",
        "~rnd:Purple": "Purple",
        "~rnd:Tuesday": "Tuesday",
        "~rnd:Chair": "Chair",
        "~rnd:Mountain": "Mountain",
        "~rnd:Penguin": "Penguin",
        "~rnd:Guitar": "Guitar",
        "~rnd:Sandwich": "Sandwich",
        "~rnd:Volcano": "Volcano",
        "~rnd:Elevator": "Elevator",
    }

    all_anchors = {**ANCHORS, **RANDOM_ANCHORS}
    results_all_models = {}

    for model_name in MODELS:
        short = model_name.split("/")[-1]
        print(f"\n{'='*80}")
        print(f"MODEL: {short}")
        print(f"{'='*80}")

        model = SentenceTransformer(model_name, trust_remote_code=True)
        embed = lambda texts: model.encode(texts, show_progress_bar=True, convert_to_numpy=True,
                                           batch_size=32)

        # Embed all responses
        print("  Embedding chosen responses...")
        chosen_embs = embed([f"User: {p['prompt']}\nAssistant: {p['chosen']}" for p in pairs])
        print("  Embedding rejected responses...")
        rejected_embs = embed([f"User: {p['prompt']}\nAssistant: {p['rejected']}" for p in pairs])

        # Normalize
        for i in range(n):
            chosen_embs[i] = chosen_embs[i] / (norm(chosen_embs[i]) + 1e-12)
            rejected_embs[i] = rejected_embs[i] / (norm(rejected_embs[i]) + 1e-12)

        # Embed all anchors
        anchor_embs = {}
        for name, text in all_anchors.items():
            e = embed([text])[0]
            anchor_embs[name] = e / (norm(e) + 1e-12)

        # Test each anchor
        results = []
        for name, text in all_anchors.items():
            a_emb = anchor_embs[name]

            # Compute cosine similarities
            chosen_cos = [float(np.dot(chosen_embs[i], a_emb)) for i in range(n)]
            rejected_cos = [float(np.dot(rejected_embs[i], a_emb)) for i in range(n)]

            if name.startswith("~") and not name.startswith("~rnd:"):
                # Negative anchor: LOWER similarity to bad thing = better
                correct = [rejected_cos[i] > chosen_cos[i] for i in range(n)]
            else:
                # Positive anchor or random: HIGHER similarity = better
                correct = [chosen_cos[i] > rejected_cos[i] for i in range(n)]

            acc = sum(correct) / n
            lo, hi = bootstrap_ci(correct)

            # Margin analysis
            if name.startswith("~") and not name.startswith("~rnd:"):
                margins = [rejected_cos[i] - chosen_cos[i] for i in range(n)]
            else:
                margins = [chosen_cos[i] - rejected_cos[i] for i in range(n)]

            results.append({
                "name": name,
                "text": text,
                "acc": acc,
                "ci_lo": lo,
                "ci_hi": hi,
                "mean_margin": np.mean(margins),
                "median_margin": np.median(margins),
            })

        results.sort(key=lambda x: -x["acc"])
        results_all_models[short] = results

        # Print results by category
        print(f"\n  ALL ANCHORS RANKED BY ACCURACY (n={n}):")
        print(f"  {'Name':30s} {'Acc':>5s} {'CI':>13s} {'Margin':>8s} Text")
        print(f"  {'-'*100}")
        for r in results:
            ci_str = f"[{r['ci_lo']:.0%},{r['ci_hi']:.0%}]"
            print(f"  {r['name']:30s} {r['acc']:4.1%} {ci_str:>13s} {r['mean_margin']:+.5f}  {r['text'][:50]}")

        # Summary stats
        random_accs = [r["acc"] for r in results if r["name"].startswith("~rnd:")]
        positive_accs = [r["acc"] for r in results if not r["name"].startswith("~")]
        negative_accs = [r["acc"] for r in results if r["name"].startswith("~") and not r["name"].startswith("~rnd:")]

        print(f"\n  SUMMARY:")
        print(f"    Random words:  mean={np.mean(random_accs):.1%}  range=[{np.min(random_accs):.1%}, {np.max(random_accs):.1%}]")
        print(f"    Positive anchors: mean={np.mean(positive_accs):.1%}  range=[{np.min(positive_accs):.1%}, {np.max(positive_accs):.1%}]")
        if negative_accs:
            print(f"    Negative anchors (inverted): mean={np.mean(negative_accs):.1%}  range=[{np.min(negative_accs):.1%}, {np.max(negative_accs):.1%}]")

        # Test COMBINATIONS: cos(positive) - cos(negative)
        print(f"\n  COMBINATION TESTS: cos(positive) - alpha * cos(negative)")
        combos = [
            ("Good", "Flattering"),
            ("Good", "Enthusiastic"),
            ("Good", "Sycophantic"),
            ("Good", "Nice"),
            ("Good", "Bad"),
            ("Helpful", "Flattering"),
            ("Honest", "Flattering"),
            ("genuinely serves", "misleading+flattering"),
            ("helps user decide", "sycophantic response"),
            ("thorough+honest", "misleading+flattering"),
            ("golden rule", "~Bad"),
            ("honest kiss", "~Flattering"),
        ]
        print(f"  {'Combo':50s} {'a=0.5':>6s} {'a=1.0':>6s} {'a=1.5':>6s}")
        print(f"  {'-'*75}")
        for pos_name, neg_name in combos:
            # Look up the actual anchor name
            pos_key = pos_name
            neg_key = neg_name if neg_name in anchor_embs else f"~{neg_name}"
            if pos_key not in anchor_embs or neg_key not in anchor_embs:
                continue

            pos_emb = anchor_embs[pos_key]
            neg_emb = anchor_embs[neg_key]
            accs = []
            for alpha in [0.5, 1.0, 1.5]:
                chosen_score = [float(np.dot(chosen_embs[i], pos_emb)) - alpha * float(np.dot(chosen_embs[i], neg_emb))
                                for i in range(n)]
                rejected_score = [float(np.dot(rejected_embs[i], pos_emb)) - alpha * float(np.dot(rejected_embs[i], neg_emb))
                                  for i in range(n)]
                correct = [chosen_score[i] > rejected_score[i] for i in range(n)]
                acc = sum(correct) / n
                accs.append(acc)
            label = f"cos({pos_name}) - a*cos({neg_name})"
            print(f"  {label:50s} {accs[0]:5.1%}  {accs[1]:5.1%}  {accs[2]:5.1%}")

        del model
        gc.collect()

    # =========================================================
    # CROSS-MODEL STABILITY
    # =========================================================
    print(f"\n{'='*80}")
    print(f"CROSS-MODEL STABILITY")
    print(f"{'='*80}")

    # For each anchor, show accuracy across all models
    anchor_names = [r["name"] for r in results_all_models[list(results_all_models.keys())[0]]]
    print(f"\n  {'Anchor':30s}", end="")
    for short in results_all_models:
        print(f" {short:>12s}", end="")
    print(f" {'Range':>7s} {'Mean':>6s}")
    print(f"  {'-'*90}")

    stability_data = []
    for name in anchor_names:
        accs = []
        for short, results in results_all_models.items():
            r = next(r for r in results if r["name"] == name)
            accs.append(r["acc"])
        spread = max(accs) - min(accs)
        mean_acc = np.mean(accs)
        stability_data.append({"name": name, "accs": accs, "spread": spread, "mean": mean_acc})

    # Sort by mean accuracy
    stability_data.sort(key=lambda x: -x["mean"])
    for sd in stability_data:
        print(f"  {sd['name']:30s}", end="")
        for a in sd["accs"]:
            print(f" {a:11.1%}", end="")
        print(f" {sd['spread']:6.1%}  {sd['mean']:5.1%}")

    # Most stable (small spread) among those above 55%
    good_stable = [sd for sd in stability_data if sd["mean"] > 0.55 and sd["spread"] < 0.10]
    if good_stable:
        print(f"\n  STABLE AND ABOVE 55% (spread < 10%):")
        for sd in sorted(good_stable, key=lambda x: -x["mean"]):
            print(f"    {sd['name']:30s} mean={sd['mean']:.1%}  spread={sd['spread']:.1%}")

    # Save full results
    out = {}
    for short, results in results_all_models.items():
        out[short] = results
    with open(CACHE_DIR / "results.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"\n  Full results saved to {CACHE_DIR / 'results.json'}")


if __name__ == "__main__":
    main()
