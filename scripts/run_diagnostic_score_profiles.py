#!/usr/bin/env python3
"""Produce per-case score profiles across 15 independent axes.

Depends on: methodology/RESEARCH_CONTEXT.md and the original + warmth batteries.
Output: notes/research_cycles/diagnostic_profiles/profile_results.json
"""

from __future__ import annotations

import gc
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

BATTERY_ORIGINAL = (
    ROOT
    / "notes"
    / "research_cycles"
    / "cycle_002_potential_shaping"
    / "controlled_evaluative_axis_battery_v3_50_cases.jsonl"
)
BATTERY_WARMTH = (
    ROOT
    / "notes"
    / "research_cycles"
    / "battery_rebalancing"
    / "warmth_cases.jsonl"
)
OUTPUT_PATH = (
    ROOT / "notes" / "research_cycles" / "diagnostic_profiles" / "profile_results.json"
)

MODELS = [
    "snowflake/snowflake-arctic-embed-m",
    "BAAI/bge-m3",
    "nomic-ai/nomic-embed-text-v1.5",
]

AXES = {
    "careful": {"positive": ["Careful"], "negative": ["Reckless"]},
    "honest": {"positive": ["Honest"], "negative": ["Dishonest"]},
    "kind": {"positive": ["Kind"], "negative": ["Cruel"]},
    "wise": {"positive": ["Wise"], "negative": ["Foolish"]},
    "helpful": {"positive": ["Helpful"], "negative": ["Unhelpful"]},
    "thorough": {"positive": ["Thorough"], "negative": ["Superficial"]},
    "fair": {"positive": ["Fair"], "negative": ["Unfair"]},
    "responsible": {"positive": ["Responsible"], "negative": ["Irresponsible"]},
    "clear": {"positive": ["Clear"], "negative": ["Confusing"]},
    "good": {"positive": ["Good"], "negative": ["Bad"]},
    "hard": {"positive": ["Hard"], "negative": ["Soft"]},
    "gentle": {"positive": ["Gentle"], "negative": ["Harsh"]},
    "patient": {"positive": ["Patient"], "negative": ["Impatient"]},
    "supportive": {"positive": ["Supportive"], "negative": ["Dismissive"]},
    "constructive": {"positive": ["Constructive"], "negative": ["Destructive"]},
}


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def with_split(rows: list[dict], split: str) -> list[dict]:
    tagged = []
    for i, row in enumerate(rows):
        copied = dict(row)
        copied["_split"] = split
        copied["_split_index"] = i
        tagged.append(copied)
    return tagged


def framed(case: dict, key: str) -> str:
    return f"User: {case['prompt']}\nAssistant: {case[key]}"


def normalize(vec: np.ndarray) -> np.ndarray:
    return vec / (np.linalg.norm(vec) + 1e-12)


def compute_axis(embed_fn, positive: list[str], negative: list[str]) -> np.ndarray:
    pos_embs = embed_fn(positive)
    neg_embs = embed_fn(negative)
    return normalize(pos_embs.mean(axis=0) - neg_embs.mean(axis=0))


def accuracy_from_margins(margins: np.ndarray) -> dict:
    correct = float(np.sum(margins > 0))
    ties = float(np.sum(margins == 0))
    total = int(len(margins))
    return {
        "accuracy": (correct + 0.5 * ties) / total if total else 0.0,
        "correct": int(correct),
        "ties": int(ties),
        "total": total,
        "mean_margin": float(np.mean(margins)) if total else 0.0,
        "margins": [float(x) for x in margins],
    }


def vote_accuracy(axis_names: list[str], margin_by_axis: dict[str, np.ndarray]) -> dict:
    vote_margins = np.sum([np.sign(margin_by_axis[name]) for name in axis_names], axis=0)
    stats = accuracy_from_margins(vote_margins)
    stats["axes"] = axis_names
    stats["vote_margins"] = [float(x) for x in vote_margins]
    return stats


def split_indices(rows: list[dict], split: str) -> list[int]:
    return [i for i, row in enumerate(rows) if row["_split"] == split]


def subset_margins(margins: np.ndarray, indices: list[int]) -> np.ndarray:
    return np.array([margins[i] for i in indices], dtype=float)


def pct(value: float) -> str:
    return f"{value * 100:5.1f}%"


def main() -> None:
    from sentence_transformers import SentenceTransformer

    original = with_split(read_jsonl(BATTERY_ORIGINAL), "original")
    warmth = with_split(read_jsonl(BATTERY_WARMTH), "warmth")
    combined = original + warmth
    split_rows = {
        "original": original,
        "warmth": warmth,
        "combined": combined,
    }
    split_index_map = {
        "original": split_indices(combined, "original"),
        "warmth": split_indices(combined, "warmth"),
        "combined": list(range(len(combined))),
    }

    print("Diagnostic Score Profiles")
    print(f"Original cases: {len(original)}")
    print(f"Warmth cases:   {len(warmth)}")
    print(f"Combined cases: {len(combined)}")
    print(f"Axes: {len(AXES)}")

    results = {
        "experiment": "Diagnostic Score Profiles",
        "models": MODELS,
        "batteries": {
            "original": str(BATTERY_ORIGINAL),
            "warmth": str(BATTERY_WARMTH),
        },
        "axes": AXES,
        "results": {},
    }

    for model_name in MODELS:
        print("\n" + "=" * 100)
        print(f"MODEL: {model_name}")
        print("=" * 100)

        model = SentenceTransformer(model_name, trust_remote_code=True)

        def embed_fn(texts: list[str]) -> np.ndarray:
            return model.encode(texts, show_progress_bar=False, convert_to_numpy=True)

        better_embs = embed_fn([framed(case, "better") for case in combined])
        worse_embs = embed_fn([framed(case, "worse") for case in combined])

        axes = {
            axis_name: compute_axis(embed_fn, anchors["positive"], anchors["negative"])
            for axis_name, anchors in AXES.items()
        }
        better_scores_by_axis = {
            axis_name: better_embs @ axis for axis_name, axis in axes.items()
        }
        worse_scores_by_axis = {
            axis_name: worse_embs @ axis for axis_name, axis in axes.items()
        }
        margins_by_axis = {
            axis_name: better_scores_by_axis[axis_name] - worse_scores_by_axis[axis_name]
            for axis_name in AXES
        }

        axis_accuracy = {}
        for split_name, indices in split_index_map.items():
            axis_accuracy[split_name] = {
                axis_name: accuracy_from_margins(subset_margins(margins, indices))
                for axis_name, margins in margins_by_axis.items()
            }

        ranked_axes = sorted(
            AXES,
            key=lambda name: (
                axis_accuracy["combined"][name]["accuracy"],
                axis_accuracy["combined"][name]["mean_margin"],
            ),
            reverse=True,
        )
        best3 = ranked_axes[:3]
        best5 = ranked_axes[:5]

        vote_sets = {
            "all_15": list(AXES.keys()),
            "best_3_by_combined_accuracy": best3,
            "best_5_by_combined_accuracy": best5,
        }
        vote_results = {}
        for vote_name, vote_axes in vote_sets.items():
            vote_results[vote_name] = {}
            for split_name, indices in split_index_map.items():
                subset_by_axis = {
                    axis_name: subset_margins(margins_by_axis[axis_name], indices)
                    for axis_name in vote_axes
                }
                vote_results[vote_name][split_name] = vote_accuracy(vote_axes, subset_by_axis)

        case_profiles = {}
        wrong_majority_cases = {}
        for split_name, rows in split_rows.items():
            indices = split_index_map[split_name]
            profiles = []
            wrong_rows = []
            all15_vote_margins = vote_results["all_15"][split_name]["vote_margins"]
            for local_i, global_i in enumerate(indices):
                case = combined[global_i]
                better_scores = {
                    axis_name: float(better_scores_by_axis[axis_name][global_i])
                    for axis_name in AXES
                }
                worse_scores = {
                    axis_name: float(worse_scores_by_axis[axis_name][global_i])
                    for axis_name in AXES
                }
                margins = {
                    axis_name: float(margins_by_axis[axis_name][global_i])
                    for axis_name in AXES
                }
                correct_axes = [axis_name for axis_name, margin in margins.items() if margin > 0]
                wrong_axes = [axis_name for axis_name, margin in margins.items() if margin < 0]
                tied_axes = [axis_name for axis_name, margin in margins.items() if margin == 0]
                vote_margin = float(all15_vote_margins[local_i])
                profile = {
                    "id": case.get("id"),
                    "split": case["_split"],
                    "split_index": case["_split_index"],
                    "category": case.get("category"),
                    "phenomenon": case.get("phenomenon"),
                    "prompt": case.get("prompt"),
                    "better_scores": better_scores,
                    "worse_scores": worse_scores,
                    "margins": margins,
                    "correct_axes": correct_axes,
                    "wrong_axes": wrong_axes,
                    "tied_axes": tied_axes,
                    "num_correct_axes": len(correct_axes),
                    "num_wrong_axes": len(wrong_axes),
                    "num_tied_axes": len(tied_axes),
                    "all_axes_fail": len(correct_axes) == 0,
                    "only_one_or_two_axes_succeed": len(correct_axes) in (1, 2),
                    "all15_vote_margin": vote_margin,
                    "all15_vote_correct": vote_margin > 0,
                    "all15_vote_tie": vote_margin == 0,
                }
                profiles.append(profile)
                if vote_margin <= 0:
                    wrong_rows.append(profile)
            case_profiles[split_name] = profiles
            wrong_majority_cases[split_name] = wrong_rows

        distribution = {}
        for split_name, profiles in case_profiles.items():
            counts = {str(i): 0 for i in range(len(AXES) + 1)}
            for profile in profiles:
                counts[str(profile["num_correct_axes"])] += 1
            distribution[split_name] = counts

        print("\nAxis pairwise accuracy")
        print(f"{'axis':<15} {'original':>10} {'warmth':>10} {'combined':>10}")
        print("-" * 51)
        for axis_name in ranked_axes:
            print(
                f"{axis_name:<15} "
                f"{pct(axis_accuracy['original'][axis_name]['accuracy']):>10} "
                f"{pct(axis_accuracy['warmth'][axis_name]['accuracy']):>10} "
                f"{pct(axis_accuracy['combined'][axis_name]['accuracy']):>10}"
            )

        print("\nMajority vote accuracy")
        print(f"{'vote set':<30} {'original':>10} {'warmth':>10} {'combined':>10}")
        print("-" * 66)
        for vote_name, split_results in vote_results.items():
            print(
                f"{vote_name:<30} "
                f"{pct(split_results['original']['accuracy']):>10} "
                f"{pct(split_results['warmth']['accuracy']):>10} "
                f"{pct(split_results['combined']['accuracy']):>10}"
            )

        print(f"\nBest 3 axes by combined accuracy: {', '.join(best3)}")
        print(f"Best 5 axes by combined accuracy: {', '.join(best5)}")

        print("\nDistribution: number of axes that got each case right")
        for split_name, counts in distribution.items():
            compact = ", ".join(f"{k}:{v}" for k, v in counts.items() if v)
            print(f"{split_name}: {compact}")

        print("\nAll-15 majority-vote wrong or tied cases")
        for split_name, wrong_rows in wrong_majority_cases.items():
            print(f"\n{split_name}: {len(wrong_rows)} cases")
            for profile in wrong_rows:
                print(
                    f"- {profile['id']} | {profile['category']} | "
                    f"{profile['phenomenon']} | correct_axes={profile['num_correct_axes']} | "
                    f"vote_margin={profile['all15_vote_margin']:+.0f} | "
                    f"right={','.join(profile['correct_axes']) or 'none'}"
                )

        results["results"][model_name] = {
            "ranked_axes_by_combined_accuracy": ranked_axes,
            "best3": best3,
            "best5": best5,
            "axis_accuracy": axis_accuracy,
            "vote_results": vote_results,
            "num_correct_axes_distribution": distribution,
            "case_profiles": case_profiles,
            "wrong_or_tied_all15_majority_cases": wrong_majority_cases,
        }

        del model
        gc.collect()

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nSaved JSON: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
