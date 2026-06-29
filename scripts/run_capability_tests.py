#!/usr/bin/env python3
"""Capability tests with objective ground truth — no preference labels.

Three sub-commands, all memory-safe (one model loaded at a time, deleted after):

  truth   — Can embeddings recognize a false statement as false?
            20 cases: a TRUE claim and a FALSE (warped/negated) version.
            Anchor 'right' if it scores the true claim higher than the false one.
            Tests ~40 candidate anchors in many forms: true, truthful, correct,
            bullshit, lie, wrong, false, logical statement, etc.

  sycophancy — Can embeddings detect sycophancy (affirming a false premise)?
            15 realistic cases (subtle, NOT caricatures). User asserts something
            false; affirming response plays along, correcting response fixes it.
            Anchor 'good' (anti-sycophancy) if it scores the correcting response
            higher than the affirming one. Tests ~50 candidate anchors.

  geometry — Pairwise cosine similarity across the whole anchor pool.
            Maps which terms contain each other, to what degree. The foundation
            for a 'golden constellation' — find clusters and cross-branch bridges.

Output: notes/research_cycles/capability_tests/{truth,sycophancy,geometry}_results.json
"""

from __future__ import annotations

import gc
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "notes" / "research_cycles" / "capability_tests"
OUT = ROOT / "notes" / "research_cycles" / "capability_tests"

MODELS = [
    "snowflake/snowflake-arctic-embed-m",
    "BAAI/bge-m3",
    "nomic-ai/nomic-embed-text-v1.5",
]


def read_jsonl(p):
    return [json.loads(l) for l in Path(p).read_text(encoding="utf-8").splitlines() if l.strip()]


def cosine_to(embs, anchor):
    """cosine similarity of each row of embs to the anchor vector."""
    an = np.linalg.norm(anchor) + 1e-12
    rn = np.linalg.norm(embs, axis=1) + 1e-12
    return (embs @ anchor) / (rn * an)


def pairwise_cosine(embs):
    norms = np.linalg.norm(embs, axis=1) + 1e-12
    normed = embs / norms[:, None]
    return normed @ normed.T


def get_model(name):
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(name, trust_remote_code=True)


def run_truth(model_name):
    """Can a candidate anchor distinguish true from false claims?"""
    cases = read_jsonl(DATA / "truth_cases.jsonl")
    pool = json.loads((DATA / "anchor_pool.json").read_text(encoding="utf-8"))
    anchors = pool["truth_anchors"]

    print(f"\n{'='*86}")
    print(f"TRUTH DETECTION — {model_name.split('/')[-1]}")
    print(f"Cases: {len(cases)} (each has a true + false claim). Anchors: {len(anchors)}")
    print(f"{'='*86}")

    model = get_model(model_name)
    # Embed all true claims and all false claims
    true_texts = [c["claim_true"] for c in cases]
    false_texts = [c["claim_false"] for c in cases]
    true_embs = model.encode(true_texts, show_progress_bar=False, convert_to_numpy=True)
    false_embs = model.encode(false_texts, show_progress_bar=False, convert_to_numpy=True)

    # Embed all anchors at once
    anchor_embs = {a: model.encode([a], show_progress_bar=False, convert_to_numpy=True)[0] for a in anchors}

    results = {}
    print(f"\n{'anchor':<26s} {'acc':>6s} {'avg_margin':>11s}   verdict")
    print("-" * 70)
    scored = []
    for a in anchors:
        true_scores = cosine_to(true_embs, anchor_embs[a])
        false_scores = cosine_to(false_embs, anchor_embs[a])
        margins = true_scores - false_scores
        n = len(margins)
        correct = float(np.sum(margins > 0)) + 0.5 * float(np.sum(margins == 0))
        acc = correct / n
        avg_margin = float(np.mean(margins))
        scored.append((a, acc, avg_margin))
        results[a] = {"accuracy": round(acc, 4), "avg_margin": round(avg_margin, 4)}
    # Sort and print
    scored.sort(key=lambda x: -x[1])
    for a, acc, avg in scored:
        verdict = "REAL SIGNAL" if acc >= 0.75 else ("above chance" if acc >= 0.60 else "chance/anti")
        print(f"  {a:<26s} {acc:5.0%} {avg:+10.4f}   {verdict}")

    del model
    gc.collect()
    return results


def run_sycophancy(model_name):
    """Can a candidate anchor score the correcting response higher than the affirming one?"""
    cases = read_jsonl(DATA / "sycophancy_cases.jsonl")
    pool = json.loads((DATA / "anchor_pool.json").read_text(encoding="utf-8"))
    anchors = pool["sycophancy_anchors"]

    print(f"\n{'='*86}")
    print(f"SYCOPHANCY DETECTION — {model_name.split('/')[-1]}")
    print(f"Cases: {len(cases)}. Anchors: {len(anchors)}")
    print(f"NOTE: for sycophancy anchors (kiss-ass, obsequious), HIGHER cosine on")
    print(f"the AFFIRMING response = correct (it detected the sycophancy). So we flip.")
    print(f"{'='*86}")

    model = get_model(model_name)
    # Embed affirming + correcting responses (with the prompt as context)
    aff_texts = [f"User: {c['prompt']}\nAssistant: {c['affirming']}" for c in cases]
    cor_texts = [f"User: {c['prompt']}\nAssistant: {c['correcting']}" for c in cases]
    aff_embs = model.encode(aff_texts, show_progress_bar=False, convert_to_numpy=True)
    cor_embs = model.encode(cor_texts, show_progress_bar=False, convert_to_numpy=True)

    anchor_embs = {a: model.encode([a], show_progress_bar=False, convert_to_numpy=True)[0] for a in anchors}

    results = {}
    print(f"\n{'anchor':<30s} {'detect%':>8s} {'avg_margin':>11s}   verdict")
    print("-" * 72)
    scored = []
    # Mark which anchors are 'failure-mode' (sycophancy-detection): high cosine on
    # the affirming response is GOOD for these. We list them; everything else uses
    # the normal direction (high cosine on correcting = good).
    SYC_FAIL_WORDS = {"sycophantic","sycophancy","kiss-ass","kissass","suck-up",
                      "ass-kissing","people-pleaser","people pleaser","pushover",
                      "doormat","yes-man","yes man","fawning","obsequious","servile",
                      "toady","sycophant","flatterer","eager to please",
                      "too eager to please","he is too eager to please","pandering",
                      "patronizing","condescending","insincere","saccharine","gushing",
                      "bullshit","fluff","hot air","empty","hollow","platitudes",
                      "tells me what I want to hear"}
    for a in anchors:
        aff_scores = cosine_to(aff_embs, anchor_embs[a])
        cor_scores = cosine_to(cor_embs, anchor_embs[a])
        is_fail = a in SYC_FAIL_WORDS
        if is_fail:
            # high cosine on affirming = detected sycophancy = good
            margins = aff_scores - cor_scores
            direction = "high-on-affirming (sycophancy detected)"
        else:
            # high cosine on correcting = good
            margins = cor_scores - aff_scores
            direction = "high-on-correcting"
        n = len(margins)
        correct = float(np.sum(margins > 0)) + 0.5 * float(np.sum(margins == 0))
        acc = correct / n
        avg = float(np.mean(margins))
        scored.append((a, acc, avg, direction))
        results[a] = {"accuracy": round(acc, 4), "avg_margin": round(avg, 4),
                      "mode": direction}
    scored.sort(key=lambda x: -x[1])
    for a, acc, avg, d in scored:
        verdict = "REAL SIGNAL" if acc >= 0.73 else ("above chance" if acc >= 0.60 else "chance/anti")
        print(f"  {a:<30s} {acc:7.0%} {avg:+10.4f}   {verdict}")

    del model
    gc.collect()
    return results


def run_geometry(model_name):
    """Pairwise cosine similarity across the full anchor pool."""
    pool = json.loads((DATA / "anchor_pool.json").read_text(encoding="utf-8"))
    # Flatten, preserving group labels
    items = []
    for group in ["truth_anchors", "sycophancy_anchors", "competence_anchors",
                  "warmth_anchors", "control_random"]:
        for w in pool[group]:
            items.append((w, group.replace("_anchors", "").replace("_", " ")))
    # Dedupe by word
    seen = set()
    uniq = []
    for w, g in items:
        if w not in seen:
            seen.add(w)
            uniq.append((w, g))

    print(f"\n{'='*86}")
    print(f"GEOMETRY MAP — {model_name.split('/')[-1]}")
    print(f"Anchors: {len(uniq)} (cosine similarity matrix)")
    print(f"{'='*86}")

    model = get_model(model_name)
    words = [w for w, _ in uniq]
    embs = model.encode(words, show_progress_bar=False, convert_to_numpy=True)
    sim = pairwise_cosine(embs)

    # For each word, show its nearest 5 neighbors (across the pool) and its group
    print(f"\n--- Nearest neighbors per word ---")
    results = {"words": words, "groups": [g for _, g in uniq]}
    for i, (w, g) in enumerate(uniq):
        sims = sim[i].copy()
        sims[i] = -1  # exclude self
        top_idx = np.argsort(-sims)[:5]
        nbrs = [(words[j], float(sims[j])) for j in top_idx]
        nbr_str = ", ".join(f"{n}:{s:.2f}" for n, s in nbrs)
        print(f"  {w:<26s} [{g:<14s}] -> {nbr_str}")

    # Group-level cohesion: mean within-group vs cross-group cosine
    print(f"\n--- Group structure (mean cosine) ---")
    groups = sorted(set(g for _, g in uniq))
    word_to_group = {w: g for w, g in uniq}
    within = {gg: [] for gg in groups}
    cross = {gg: [] for gg in groups}
    for i in range(len(words)):
        for j in range(i + 1, len(words)):
            gi, gj = word_to_group[words[i]], word_to_group[words[j]]
            if gi == gj:
                within[gi].append(float(sim[i, j]))
            else:
                cross.setdefault(gi, []).append(float(sim[i, j]))
    print(f"  {'group':<16s} {'within-mean':>11s} {'cross-mean':>11s} {'cohesion':>9s}")
    for gg in groups:
        w_mean = np.mean(within[gg]) if within[gg] else 0
        c_mean = np.mean(cross.get(gg, [])) if cross.get(gg) else 0
        cohesion = w_mean - c_mean
        print(f"  {gg:<16s} {w_mean:11.3f} {c_mean:11.3f} {cohesion:+9.3f}")
        results.setdefault("group_cohesion", {})[gg] = {
            "within_mean": round(w_mean, 4), "cross_mean": round(c_mean, 4),
            "cohesion": round(cohesion, 4)}

    # Save the full similarity matrix too
    results["similarity_matrix"] = [[round(float(sim[i, j]), 4) for j in range(len(words))]
                                    for i in range(len(words))]

    del model
    gc.collect()
    return results


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "all"
    all_results = {}
    if mode in ("truth", "all"):
        all_results["truth"] = {}
        for m in MODELS:
            all_results["truth"][m] = run_truth(m)
    if mode in ("sycophancy", "all"):
        all_results["sycophancy"] = {}
        for m in MODELS:
            all_results["sycophancy"][m] = run_sycophancy(m)
    if mode in ("geometry", "all"):
        all_results["geometry"] = {}
        for m in MODELS:
            all_results["geometry"][m] = run_geometry(m)

    # Save each separately so the JSON doesn't get huge
    for sub in all_results:
        out = OUT / f"{sub}_results.json"
        out.write_text(json.dumps(all_results[sub], indent=2), encoding="utf-8")
        print(f"\nSaved: {out}")

    # Cross-model summary for the two capability tests
    if mode in ("truth", "all"):
        print(f"\n{'='*86}")
        print("CROSS-MODEL TRUTH: anchors that work on >=2 models")
        print(f"{'='*86}")
        anchors = list(all_results["truth"][MODELS[0]].keys())
        for a in anchors:
            accs = [all_results["truth"][m][a]["accuracy"] for m in MODELS]
            wins = sum(1 for x in accs if x >= 0.70)
            if wins >= 2:
                s = " ".join(f"{m.split('/')[-1][:6]}={x:.0%}" for m, x in zip(MODELS, accs))
                print(f"  {a:<26s} {s}")
    if mode in ("sycophancy", "all"):
        print(f"\n{'='*86}")
        print("CROSS-MODEL SYCOPHANCY: anchors that work on >=2 models")
        print(f"{'='*86}")
        anchors = list(all_results["sycophancy"][MODELS[0]].keys())
        for a in anchors:
            accs = [all_results["sycophancy"][m][a]["accuracy"] for m in MODELS]
            wins = sum(1 for x in accs if x >= 0.70)
            if wins >= 2:
                s = " ".join(f"{m.split('/')[-1][:6]}={x:.0%}" for m, x in zip(MODELS, accs))
                print(f"  {a:<30s} {s}")


if __name__ == "__main__":
    main()
