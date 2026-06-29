#!/usr/bin/env python3
"""Multi-direction quality scoring.

The single centroid direction is a compromise between opposing quality
dimensions (firmness vs warmth). This tests whether separate directions
for different quality types, combined intelligently, beat the single centroid.

Approaches:
1. Two directions (firmness + warmth), max-of-two scoring
2. Two directions, prompt-based routing (embed the prompt, decide which direction)
3. Category-specific directions with leave-one-out
4. PCA on battery pair differences to find the principal quality dimensions
5. Balanced sub-sampling: what if firmness and warmth had equal weight?
"""

import json, gc, sys, io
from pathlib import Path
import numpy as np
from numpy.linalg import norm
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

ROOT = Path(__file__).resolve().parents[1]
BATTERY_50 = ROOT / "notes" / "research_cycles" / "cycle_002_potential_shaping" / "controlled_evaluative_axis_battery_v3_50_cases.jsonl"
WARMTH_20 = ROOT / "notes" / "research_cycles" / "battery_rebalancing" / "warmth_cases.jsonl"
EXPANSION_DIR = ROOT / "notes" / "research_cycles" / "battery_expansion"
OUT_DIR = ROOT / "notes" / "research_cycles" / "centroid_deep"

MODELS = [
    "snowflake/snowflake-arctic-embed-m",
    "BAAI/bge-m3",
    "nomic-ai/nomic-embed-text-v1.5",
]


def load_cases(path):
    cases = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                cases.append(json.loads(line))
    return cases


def embed_and_norm(model, texts):
    embs = model.encode(texts, convert_to_numpy=True, batch_size=32)
    for i in range(len(embs)):
        embs[i] /= norm(embs[i]) + 1e-12
    return embs


def accuracy(better_embs, worse_embs, direction):
    n = len(better_embs)
    correct = sum(
        1 for i in range(n)
        if np.dot(better_embs[i], direction) > np.dot(worse_embs[i], direction)
    )
    return correct / n


def make_dir(better_embs, worse_embs):
    d = better_embs.mean(axis=0) - worse_embs.mean(axis=0)
    n = norm(d)
    if n < 1e-12:
        return d
    return d / n


def score_margin(emb, direction):
    return float(np.dot(emb, direction))


def main():
    from sentence_transformers import SentenceTransformer

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    np.random.seed(42)

    orig_cases = load_cases(BATTERY_50)
    warmth_cases = load_cases(WARMTH_20)
    battery = orig_cases + warmth_cases

    expansion = []
    for f in sorted(EXPANSION_DIR.glob("*.jsonl")):
        expansion.extend(load_cases(f))

    # Identify warmth vs firmness in battery
    warmth_cats = {"emotional_support", "warmth_appropriate", "patience_needed",
                   "gentle_teaching", "appropriate_agreement"}
    firm_idx = [i for i, c in enumerate(battery)
                if c.get("category", "") not in warmth_cats]
    warm_idx = [i for i, c in enumerate(battery)
                if c.get("category", "") in warmth_cats]

    print(f"Battery: {len(battery)} (firmness-like: {len(firm_idx)}, warmth-like: {len(warm_idx)})")
    print(f"Expansion: {len(expansion)}")

    all_results = {}

    for model_name in MODELS:
        short = model_name.split("/")[-1]
        print(f"\n{'='*80}")
        print(f"MODEL: {short}")
        print(f"{'='*80}")

        model = SentenceTransformer(model_name, trust_remote_code=True)

        bat_better = embed_and_norm(model, [c["better"] for c in battery])
        bat_worse = embed_and_norm(model, [c["worse"] for c in battery])
        exp_better = embed_and_norm(model, [c["better"] for c in expansion])
        exp_worse = embed_and_norm(model, [c["worse"] for c in expansion])

        # Also embed prompts for routing
        bat_prompts = embed_and_norm(model, [c["prompt"] for c in battery])
        exp_prompts = embed_and_norm(model, [c["prompt"] for c in expansion])

        fi = np.array(firm_idx)
        wi = np.array(warm_idx)

        dir_full = make_dir(bat_better, bat_worse)
        dir_firm = make_dir(bat_better[fi], bat_worse[fi])
        dir_warm = make_dir(bat_better[wi], bat_worse[wi])

        mr = {}

        # =================================================================
        # BASELINE: Single centroid
        # =================================================================
        base_bat = accuracy(bat_better, bat_worse, dir_full)
        base_exp = accuracy(exp_better, exp_worse, dir_full)
        print(f"\n  Baseline (single centroid):")
        print(f"    Battery: {base_bat:.0%}  Expansion: {base_exp:.0%}")
        mr["baseline"] = {"battery": float(base_bat), "expansion": float(base_exp)}

        # =================================================================
        # 1. MAX-OF-TWO: score on both directions, use whichever gives bigger margin
        # =================================================================
        print(f"\n--- 1. Max-of-Two Directions ---")

        def max_of_two_accuracy(b_embs, w_embs):
            n = len(b_embs)
            correct = 0
            for i in range(n):
                firm_margin = score_margin(b_embs[i], dir_firm) - score_margin(w_embs[i], dir_firm)
                warm_margin = score_margin(b_embs[i], dir_warm) - score_margin(w_embs[i], dir_warm)
                # Pick whichever direction gives bigger absolute margin
                if abs(firm_margin) > abs(warm_margin):
                    if firm_margin > 0:
                        correct += 1
                else:
                    if warm_margin > 0:
                        correct += 1
            return correct / n

        mot_bat = max_of_two_accuracy(bat_better, bat_worse)
        mot_exp = max_of_two_accuracy(exp_better, exp_worse)
        print(f"    Battery: {mot_bat:.0%}  Expansion: {mot_exp:.0%}")
        mr["max_of_two"] = {"battery": float(mot_bat), "expansion": float(mot_exp)}

        # =================================================================
        # 2. PROMPT-BASED ROUTING
        # =================================================================
        print(f"\n--- 2. Prompt-Based Routing ---")

        # Learn to classify prompts as "needs firm response" vs "needs warm response"
        # by looking at which direction scores the training pair correctly
        prompt_labels = []
        for i in range(len(battery)):
            firm_correct = np.dot(bat_better[i], dir_firm) > np.dot(bat_worse[i], dir_firm)
            warm_correct = np.dot(bat_better[i], dir_warm) > np.dot(bat_worse[i], dir_warm)
            if firm_correct and not warm_correct:
                prompt_labels.append(0)  # firmness
            elif warm_correct and not firm_correct:
                prompt_labels.append(1)  # warmth
            elif firm_correct and warm_correct:
                prompt_labels.append(2)  # both work
            else:
                prompt_labels.append(3)  # neither works

        from collections import Counter
        label_counts = Counter(prompt_labels)
        print(f"    Prompt routing labels: firm_only={label_counts[0]}, warm_only={label_counts[1]}, both={label_counts[2]}, neither={label_counts[3]}")

        # Simple approach: use prompt embedding cosine to firmness/warmth prompt centroids
        firm_prompt_center = bat_prompts[fi].mean(axis=0)
        firm_prompt_center /= norm(firm_prompt_center) + 1e-12
        warm_prompt_center = bat_prompts[wi].mean(axis=0)
        warm_prompt_center /= norm(warm_prompt_center) + 1e-12

        def routed_accuracy(b_embs, w_embs, prompts):
            n = len(b_embs)
            correct = 0
            for i in range(n):
                cos_firm = np.dot(prompts[i], firm_prompt_center)
                cos_warm = np.dot(prompts[i], warm_prompt_center)
                if cos_firm > cos_warm:
                    direction = dir_firm
                else:
                    direction = dir_warm
                if np.dot(b_embs[i], direction) > np.dot(w_embs[i], direction):
                    correct += 1
            return correct / n

        route_bat = routed_accuracy(bat_better, bat_worse, bat_prompts)
        route_exp = routed_accuracy(exp_better, exp_worse, exp_prompts)
        print(f"    Battery: {route_bat:.0%}  Expansion: {route_exp:.0%}")
        mr["prompt_routing"] = {"battery": float(route_bat), "expansion": float(route_exp)}

        # =================================================================
        # 3. PCA ON PAIR DIFFERENCES
        # =================================================================
        print(f"\n--- 3. PCA on Pair Differences ---")

        diffs = bat_better - bat_worse
        pca = PCA(n_components=10)
        pca.fit(diffs)

        print(f"    Explained variance by top 10 PCs:")
        cumvar = 0
        for i, v in enumerate(pca.explained_variance_ratio_[:10]):
            cumvar += v
            print(f"      PC{i+1}: {v:.4f} (cumul: {cumvar:.4f})")

        # Test each PC as a quality direction
        print(f"\n    Accuracy using each PC as direction:")
        print(f"    {'PC':>4s} {'Battery':>8s} {'Expansion':>10s} {'cos(centroid)':>14s}")
        for i in range(min(5, pca.n_components_)):
            pc_dir = pca.components_[i]
            pc_dir /= norm(pc_dir) + 1e-12
            # Check sign: PC might point wrong way
            bat_acc_pos = accuracy(bat_better, bat_worse, pc_dir)
            bat_acc_neg = accuracy(bat_better, bat_worse, -pc_dir)
            if bat_acc_neg > bat_acc_pos:
                pc_dir = -pc_dir
                bat_acc = bat_acc_neg
            else:
                bat_acc = bat_acc_pos
            exp_acc = accuracy(exp_better, exp_worse, pc_dir)
            cos_c = float(np.dot(pc_dir, dir_full))
            print(f"    PC{i+1:>2d} {bat_acc:7.0%}  {exp_acc:9.0%}  {cos_c:+13.4f}")

        # Multi-PC: use top-k PCs as features for logistic regression
        print(f"\n    Logistic regression on top-k PCs:")
        for k in [2, 3, 5]:
            pc_train = pca.transform(diffs)[:, :k]
            X_tr = np.vstack([pc_train, -pc_train])
            y_tr = np.concatenate([np.ones(len(battery)), np.zeros(len(battery))])

            exp_diffs = exp_better - exp_worse
            pc_test = pca.transform(exp_diffs)[:, :k]
            X_te = np.vstack([pc_test, -pc_test])
            y_te = np.concatenate([np.ones(len(expansion)), np.zeros(len(expansion))])

            clf = LogisticRegression(max_iter=1000, C=1.0)
            clf.fit(X_tr, y_tr)
            lr_acc = clf.score(X_te, y_te)
            print(f"      Top-{k} PCs: expansion OOS = {lr_acc:.0%}")

        mr["pca_explained_var"] = [float(v) for v in pca.explained_variance_ratio_[:10]]

        # =================================================================
        # 4. BALANCED SUBSAMPLING
        # =================================================================
        print(f"\n--- 4. Balanced Subsampling ---")

        n_warm = len(warm_idx)
        n_repeats = 100
        bal_accs = []
        for _ in range(n_repeats):
            sub_firm = np.random.choice(fi, size=n_warm, replace=False)
            all_idx = np.concatenate([sub_firm, wi])
            direction = make_dir(bat_better[all_idx], bat_worse[all_idx])
            bal_accs.append(accuracy(exp_better, exp_worse, direction))

        print(f"    Balanced {n_warm}+{n_warm} (mean over {n_repeats} resamples):")
        print(f"    Expansion: {np.mean(bal_accs):.1%} [{np.percentile(bal_accs, 2.5):.0%}, {np.percentile(bal_accs, 97.5):.0%}]")

        # Also test: warmth-weighted direction (upweight warmth cases)
        weight = np.ones(len(battery))
        weight[wi] = len(fi) / len(wi)  # upweight warmth to match firmness count
        weighted_better = (bat_better * weight[:, None]).sum(axis=0) / weight.sum()
        weighted_worse = (bat_worse * weight[:, None]).sum(axis=0) / weight.sum()
        dir_weighted = weighted_better - weighted_worse
        dir_weighted /= norm(dir_weighted) + 1e-12

        wt_bat = accuracy(bat_better, bat_worse, dir_weighted)
        wt_exp = accuracy(exp_better, exp_worse, dir_weighted)
        print(f"    Warmth-upweighted centroid: Battery={wt_bat:.0%} Expansion={wt_exp:.0%}")

        mr["balanced"] = {
            "mean_exp": float(np.mean(bal_accs)),
            "weighted_battery": float(wt_bat),
            "weighted_expansion": float(wt_exp),
        }

        # =================================================================
        # 5. EXPANSION PER-CASE DETAIL WITH ALL APPROACHES
        # =================================================================
        print(f"\n--- 5. Per-Case Comparison (expansion) ---")
        print(f"    {'ID':>4s} {'Category':20s} {'Centroid':>9s} {'MaxTwo':>7s} {'Route':>7s}")

        methods_correct = {"centroid": 0, "max_two": 0, "routed": 0}
        for i in range(len(expansion)):
            c = expansion[i]
            cid = c.get("id", i)
            cat = c.get("category", "?")[:20]

            cent_ok = np.dot(exp_better[i], dir_full) > np.dot(exp_worse[i], dir_full)
            fm = score_margin(exp_better[i], dir_firm) - score_margin(exp_worse[i], dir_firm)
            wm = score_margin(exp_better[i], dir_warm) - score_margin(exp_worse[i], dir_warm)
            mot_ok = (fm > 0) if abs(fm) > abs(wm) else (wm > 0)
            cf = np.dot(exp_prompts[i], firm_prompt_center)
            cw = np.dot(exp_prompts[i], warm_prompt_center)
            route_dir = dir_firm if cf > cw else dir_warm
            route_ok = np.dot(exp_better[i], route_dir) > np.dot(exp_worse[i], route_dir)

            if cent_ok: methods_correct["centroid"] += 1
            if mot_ok: methods_correct["max_two"] += 1
            if route_ok: methods_correct["routed"] += 1

            if not cent_ok or not mot_ok or not route_ok:
                marks = f"{'OK' if cent_ok else 'XX':>9s} {'OK' if mot_ok else 'XX':>7s} {'OK' if route_ok else 'XX':>7s}"
                print(f"    {cid:>4s} {cat:20s} {marks}")

        print(f"\n    Totals: centroid={methods_correct['centroid']}/35  max_two={methods_correct['max_two']}/35  routed={methods_correct['routed']}/35")

        all_results[short] = mr

        del model
        gc.collect()

    # Final summary
    print(f"\n{'='*80}")
    print(f"FINAL SUMMARY")
    print(f"{'='*80}")

    print(f"\n  {'Method':25s}", end="")
    for short in all_results:
        print(f"  {short:>20s}", end="")
    print()

    for method_key in ["baseline", "max_of_two", "prompt_routing"]:
        label = method_key.replace("_", " ")
        print(f"  {label:25s}", end="")
        for short, r in all_results.items():
            val = r[method_key]["expansion"]
            print(f"  {val:19.0%}", end="")
        print()

    with open(OUT_DIR / "multi_direction_results.json", "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nSaved to {OUT_DIR / 'multi_direction_results.json'}")


if __name__ == "__main__":
    main()
