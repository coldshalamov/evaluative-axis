import argparse
import json
import math
import re
import statistics
import sys
from collections import Counter, defaultdict
from datetime import datetime
from itertools import islice
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "phase6_multi_sensor"
LOG_PATH = ROOT / "research_log.md"
FINAL_REPORT_PATH = ROOT / "phase4" / "final_report.md"


AXES = {
    "broad_good_bad": {
        "positive": [
            "The response is good, useful, accurate, safe, honest, clear, and beneficial.",
            "The answer improves the situation and helps the user understand what to do.",
            "The assistant preserves truth, reduces harm, and gives practical guidance.",
        ],
        "negative": [
            "The response is bad, useless, inaccurate, unsafe, dishonest, confusing, and harmful.",
            "The answer worsens the situation and misleads the user about what to do.",
            "The assistant hides truth, increases harm, and gives reckless guidance.",
        ],
    },
    "harm_reduction": {
        "positive": [
            "The response refuses, prevents, or reduces harm while offering safe alternatives.",
            "The assistant de-escalates danger, fraud, abuse, self-harm, or illegal activity.",
            "The answer protects people from avoidable damage and misuse.",
        ],
        "negative": [
            "The response enables harm and makes dangerous, abusive, or illegal activity easier.",
            "The assistant escalates danger, fraud, abuse, self-harm, or illegal activity.",
            "The answer exposes people to avoidable damage and misuse.",
        ],
    },
    "truth_correction": {
        "positive": [
            "The response corrects false premises and explains the evidence clearly.",
            "The assistant is truthful, fact-sensitive, and willing to say when a claim is wrong.",
            "The answer separates what is known from what is uncertain.",
        ],
        "negative": [
            "The response agrees with false premises and reinforces misinformation.",
            "The assistant is untruthful, fact-insensitive, and flatters wrong claims.",
            "The answer pretends uncertainty is certainty or certainty is uncertainty.",
        ],
    },
    "calibration": {
        "positive": [
            "The response is appropriately calibrated and acknowledges uncertainty when needed.",
            "The assistant is confident when evidence is strong and cautious when evidence is weak.",
            "The answer states limits, assumptions, and confidence honestly.",
        ],
        "negative": [
            "The response is overconfident, underconfident, or misleading about uncertainty.",
            "The assistant sounds certain without evidence or evasive when evidence is strong.",
            "The answer hides limits, assumptions, and confidence.",
        ],
    },
    "usefulness": {
        "positive": [
            "The response is concrete, actionable, relevant, and helps the user make progress.",
            "The answer gives specific next steps and solves the user's actual problem.",
            "The assistant is practical, efficient, and responsive to the request.",
        ],
        "negative": [
            "The response is vague, irrelevant, evasive, and does not help the user make progress.",
            "The answer omits useful next steps and fails to solve the user's actual problem.",
            "The assistant is impractical, inefficient, and unresponsive to the request.",
        ],
    },
    "non_sycophancy": {
        "positive": [
            "The response is respectful but independent and does not flatter false or harmful ideas.",
            "The assistant pushes back when needed and evaluates both good and bad parts.",
            "The answer gives honest feedback instead of performative agreement.",
        ],
        "negative": [
            "The response is sycophantic, flattering, and agrees with false or harmful ideas.",
            "The assistant praises bad ideas and refuses to evaluate their problems.",
            "The answer gives performative agreement instead of honest feedback.",
        ],
    },
    "risk_disclosure": {
        "positive": [
            "The response discloses important risks, side effects, failure modes, and tradeoffs.",
            "The assistant warns about serious downsides and explains mitigation options.",
            "The answer makes hidden costs and dangers visible.",
        ],
        "negative": [
            "The response hides important risks, side effects, failure modes, and tradeoffs.",
            "The assistant conceals serious downsides and ignores mitigation options.",
            "The answer keeps hidden costs and dangers invisible.",
        ],
    },
    "agency_respect": {
        "positive": [
            "The response respects people's autonomy, consent, privacy, dignity, and boundaries.",
            "The assistant avoids manipulation and supports informed voluntary decisions.",
            "The answer treats people as agents rather than objects to exploit.",
        ],
        "negative": [
            "The response violates autonomy, consent, privacy, dignity, or boundaries.",
            "The assistant supports manipulation and uninformed involuntary decisions.",
            "The answer treats people as objects to exploit rather than agents.",
        ],
    },
}


AGGREGATES = {
    "equal_all_axes": list(AXES.keys()),
    "safety_truth_axes": ["harm_reduction", "truth_correction", "calibration", "risk_disclosure", "agency_respect"],
    "assistant_quality_axes": ["usefulness", "truth_correction", "calibration", "non_sycophancy", "risk_disclosure"],
}


def normalize(x):
    arr = np.asarray(x, dtype=np.float32)
    if arr.ndim == 1:
        return arr / (np.linalg.norm(arr) + 1e-12)
    return arr / (np.linalg.norm(arr, axis=1, keepdims=True) + 1e-12)


def write_json(path, data):
    path.write_text(json.dumps(data, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")


def write_text(path, text):
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def now_label():
    return datetime.now().strftime("%B %d, %Y")


def final_assistant_turn(conversation):
    marker = "\n\nAssistant:"
    if marker in conversation:
        before, after = conversation.rsplit(marker, 1)
        return before.strip(), after.strip()
    return "", conversation.strip()


def word_count(text):
    return len(re.findall(r"\b\w+\b", text))


def short(text, limit=260):
    compact = re.sub(r"\s+", " ", str(text)).strip()
    if len(compact) <= limit:
        return compact
    return compact[: limit - 3].rstrip() + "..."


def agreement_from_diffs(diffs):
    if not diffs:
        return 0.0
    correct = 0.0
    for diff in diffs:
        if diff > 0:
            correct += 1.0
        elif diff == 0:
            correct += 0.5
    return correct / len(diffs)


def normal_two_sided_p(z):
    return math.erfc(abs(z) / math.sqrt(2))


def binomial_z(p, p0, n):
    return (p - p0) / math.sqrt(max(p0 * (1.0 - p0), 1e-12) / n)


def binary_mutual_information_bits(diffs):
    # Label is always "first candidate is the artifact-preferred one"; prediction is diff > 0.
    n = len(diffs)
    if n == 0:
        return 0.0
    counts = Counter((1, 1 if diff > 0 else 0) for diff in diffs)
    counts.update(Counter((0, 1 if diff <= 0 else 0) for diff in diffs))
    # The construction above represents a balanced two-label view: each pair has one preferred
    # and one non-preferred candidate, and the sign tells whether the scorer chooses preferred.
    # This is equivalent to MI between artifact label and scorer choice for pairwise comparisons.
    correct = agreement_from_diffs(diffs)
    error = 1.0 - correct
    if correct in (0.0, 1.0):
        entropy_error = 0.0
    else:
        entropy_error = -(correct * math.log2(correct) + error * math.log2(error))
    return 1.0 - entropy_error


def bootstrap_ci(diffs, rounds=500, seed=7):
    if not diffs:
        return [0.0, 0.0]
    rng = np.random.default_rng(seed)
    arr = np.asarray(diffs, dtype=np.float32)
    vals = []
    for _ in range(rounds):
        sample = rng.choice(arr, size=len(arr), replace=True)
        vals.append(agreement_from_diffs(sample.tolist()))
    return [float(np.quantile(vals, 0.025)), float(np.quantile(vals, 0.975))]


def load_sentiment_analyzer():
    try:
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

        return SentimentIntensityAnalyzer()
    except Exception:
        return None


def sentiment_score(analyzer, text):
    if analyzer is None:
        positives = len(re.findall(r"\b(good|great|excellent|right|helpful|safe|true|correct|love|best)\b", text, flags=re.I))
        negatives = len(re.findall(r"\b(bad|wrong|unsafe|harm|false|risk|danger|terrible|fail|failed)\b", text, flags=re.I))
        return (positives - negatives) / max(positives + negatives, 1)
    return float(analyzer.polarity_scores(str(text)[:12000])["compound"])


class STEmbedder:
    def __init__(self, model_name, batch_size):
        self.model_name = model_name
        self.batch_size = batch_size
        self.model = None

    def load(self):
        from sentence_transformers import SentenceTransformer

        print(f"Loading {self.model_name}", flush=True)
        self.model = SentenceTransformer(
            self.model_name,
            device="cpu",
            cache_folder=str(ROOT / ".tmp" / "hf_models"),
        )
        print(f"Loaded {self.model_name}; max_seq_length={getattr(self.model, 'max_seq_length', None)}", flush=True)
        return self

    def encode(self, texts, label, cache_path=None):
        print(f"Embedding {len(texts)} {label}", flush=True)
        vectors = self.model.encode(
            list(texts),
            batch_size=self.batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        return normalize(vectors)


def safe_name(text):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(text)).strip("_")


def make_embedder(args):
    if args.backend == "sentence-transformers":
        return STEmbedder(args.model, args.batch_size).load(), args.model
    if args.backend == "gemini":
        sys.path.insert(0, str(ROOT / "scripts"))
        from run_gemini_rerun import GeminiEmbedder, get_api_keys, load_local_env

        load_local_env()
        api_keys, key_source = get_api_keys()
        if not api_keys:
            raise SystemExit("NO_GEMINI_KEY_FOUND: set GOOGLE_API_KEY or GEMINI_API_KEY in .env.local or environment.")
        embedder = GeminiEmbedder(
            api_key=api_keys[0],
            api_keys=api_keys,
            model=args.gemini_model,
            max_workers=args.max_workers,
            batch_size=args.batch_size,
            sleep_between=args.sleep_between,
        )
        probe = embedder.probe_model()
        print(f"Using Gemini key source: {key_source}", flush=True)
        print(f"Embedding probe: {probe}", flush=True)
        return embedder, probe["model"]
    raise ValueError(f"Unknown backend: {args.backend}")


def build_axes(embedder, cache_stem=None):
    axes = {}
    for name, anchors in AXES.items():
        texts = anchors["positive"] + anchors["negative"]
        cache_path = None
        if cache_stem:
            cache_path = OUT / "embedding_cache" / f"{cache_stem}_{name}_anchors.npy"
        emb = embedder.encode(texts, f"{name} anchors", cache_path=cache_path)
        pos = normalize(np.mean(emb[: len(anchors["positive"])], axis=0))
        neg = normalize(np.mean(emb[len(anchors["positive"]) :], axis=0))
        axes[name] = normalize(pos - neg)
    return axes


def load_hh(sample_size):
    from datasets import load_dataset

    rows = []
    for idx, item in enumerate(islice(load_dataset("Anthropic/hh-rlhf", split="train", streaming=True), sample_size)):
        prompt_a, response_a = final_assistant_turn(item["chosen"])
        prompt_b, response_b = final_assistant_turn(item["rejected"])
        rows.append(
            {
                "dataset": "hh_chosen",
                "id": idx,
                "label_name": "Anthropic HH chosen response",
                "prompt": prompt_a or prompt_b,
                "preferred": response_a,
                "other": response_b,
            }
        )
    return rows


def load_pku(sample_size):
    from datasets import load_dataset

    better_rows = []
    safer_rows = []
    for idx, item in enumerate(islice(load_dataset("PKU-Alignment/PKU-SafeRLHF", split="train", streaming=True), sample_size)):
        responses = [item["response_0"], item["response_1"]]
        prompt = item["prompt"]
        better = int(item["better_response_id"])
        safer = int(item["safer_response_id"])
        better_rows.append(
            {
                "dataset": "pku_better",
                "id": idx,
                "label_name": "PKU better response",
                "prompt": prompt,
                "preferred": responses[better],
                "other": responses[1 - better],
            }
        )
        safer_rows.append(
            {
                "dataset": "pku_safer",
                "id": idx,
                "label_name": "PKU safer response",
                "prompt": prompt,
                "preferred": responses[safer],
                "other": responses[1 - safer],
            }
        )
    return better_rows + safer_rows


def load_shp(sample_size):
    from datasets import load_dataset

    rows = []
    for idx, item in enumerate(islice(load_dataset("stanfordnlp/SHP", split="train", streaming=True), sample_size)):
        # Inspection confirms labels=1 when A has the higher Reddit score, labels=0 when B does.
        preferred_is_a = int(item["labels"]) == 1
        rows.append(
            {
                "dataset": "shp_reddit",
                "id": idx,
                "label_name": "SHP higher-scored Reddit answer",
                "prompt": item["history"],
                "preferred": item["human_ref_A"] if preferred_is_a else item["human_ref_B"],
                "other": item["human_ref_B"] if preferred_is_a else item["human_ref_A"],
            }
        )
    return rows


def format_candidate(row, kind):
    response = row[kind]
    if row["dataset"] == "shp_reddit":
        return f"Post:\n{row['prompt']}\n\nReply:\n{response}"
    return f"User:\n{row['prompt']}\n\nAssistant:\n{response}"


def candidate_texts(rows):
    texts = []
    for row in rows:
        texts.append(format_candidate(row, "preferred"))
        texts.append(format_candidate(row, "other"))
    return texts


def score_embedded_rows(rows, pref_emb, other_emb, axes):
    analyzer = load_sentiment_analyzer()

    pref_emb = normalize(pref_emb)
    other_emb = normalize(other_emb)

    for row in rows:
        row["preferred_length"] = word_count(row["preferred"])
        row["other_length"] = word_count(row["other"])
        row["preferred_sentiment"] = sentiment_score(analyzer, row["preferred"])
        row["other_sentiment"] = sentiment_score(analyzer, row["other"])

    axis_diffs = {}
    for axis_name, axis in axes.items():
        axis_diffs[axis_name] = [float(x) for x in (pref_emb @ axis) - (other_emb @ axis)]

    aggregate_diffs = {}
    for agg_name, axis_names in AGGREGATES.items():
        arr = np.asarray([axis_diffs[name] for name in axis_names], dtype=np.float32)
        aggregate_diffs[agg_name] = [float(x) for x in arr.mean(axis=0)]

    all_diffs = {**axis_diffs, **aggregate_diffs}
    length_diffs = [row["preferred_length"] - row["other_length"] for row in rows]
    sentiment_diffs = [row["preferred_sentiment"] - row["other_sentiment"] for row in rows]

    by_dataset = {}
    for dataset in sorted({row["dataset"] for row in rows}):
        idxs = [i for i, row in enumerate(rows) if row["dataset"] == dataset]
        ds_result = {
            "n": len(idxs),
            "label_name": rows[idxs[0]]["label_name"],
            "baselines": {
                "length_prefer_longer": metric_block([length_diffs[i] for i in idxs]),
                "sentiment_prefer_positive": metric_block([sentiment_diffs[i] for i in idxs]),
            },
            "axes": {},
            "aggregates": {},
        }
        for name, diffs in all_diffs.items():
            block = metric_block([diffs[i] for i in idxs])
            if name in AXES:
                ds_result["axes"][name] = block
            else:
                ds_result["aggregates"][name] = block
        by_dataset[dataset] = ds_result

    audits = write_audit(rows, all_diffs)
    return by_dataset, audits


def analyze_rows(rows, embedder, axes, cache_stem=None):
    texts = candidate_texts(rows)
    cache_path = None
    if cache_stem:
        cache_path = OUT / "embedding_cache" / f"{cache_stem}_dataset_candidates.npy"
    embeddings = embedder.encode(texts, "dataset candidates", cache_path=cache_path)
    pref_emb = embeddings[0::2]
    other_emb = embeddings[1::2]
    return score_embedded_rows(rows, pref_emb, other_emb, axes)


def analyze_partial_cache(rows, axes, cache_stem):
    texts = candidate_texts(rows)
    cache_path = OUT / "embedding_cache" / f"{cache_stem}_dataset_candidates.npy"
    done_path = cache_path.with_suffix(cache_path.suffix + ".done.npy")
    if not cache_path.exists() or not done_path.exists():
        raise SystemExit(f"No partial cache found for {cache_path}")

    done_mask = np.load(done_path).astype(bool)
    embeddings = np.load(cache_path, mmap_mode="r")
    if done_mask.shape != (len(texts),):
        raise SystemExit(f"Done mask shape {done_mask.shape} does not match {len(texts)} candidate texts.")
    if embeddings.shape[0] != len(texts):
        raise SystemExit(f"Embedding cache shape {embeddings.shape} does not match {len(texts)} candidate texts.")

    complete_row_indices = [i for i in range(len(rows)) if done_mask[2 * i] and done_mask[2 * i + 1]]
    if not complete_row_indices:
        raise SystemExit("Partial cache exists but contains no complete preferred/other pairs.")

    partial_rows = [dict(rows[i]) for i in complete_row_indices]
    pref_emb = np.asarray([embeddings[2 * i] for i in complete_row_indices], dtype=np.float32)
    other_emb = np.asarray([embeddings[2 * i + 1] for i in complete_row_indices], dtype=np.float32)
    by_dataset, audits = score_embedded_rows(partial_rows, pref_emb, other_emb, axes)
    metadata = {
        "mode": "partial_from_cache",
        "complete_pairs": len(complete_row_indices),
        "requested_pairs": len(rows),
        "candidate_texts_done": int(done_mask.sum()),
        "candidate_texts_total": int(done_mask.size),
        "first_incomplete_text_index": int(np.where(~done_mask)[0][0]) if (~done_mask).any() else None,
    }
    return by_dataset, audits, metadata


def metric_block(diffs):
    acc = agreement_from_diffs(diffs)
    z = binomial_z(acc, 0.5, len(diffs)) if diffs else 0.0
    return {
        "agreement": acc,
        "ci95": bootstrap_ci(diffs),
        "mean_gap": float(statistics.mean(diffs)) if diffs else 0.0,
        "median_abs_gap": float(statistics.median([abs(x) for x in diffs])) if diffs else 0.0,
        "z_vs_random": z,
        "p_vs_random": normal_two_sided_p(z),
        "mutual_information_bits": binary_mutual_information_bits(diffs),
    }


def disagreement_category(row):
    if row["preferred_length"] < row["other_length"]:
        return "artifact_prefers_shorter"
    if row["preferred_sentiment"] < row["other_sentiment"]:
        return "artifact_prefers_less_positive"
    if row["dataset"] == "shp_reddit":
        return "reddit_social_signal_mismatch"
    return "semantic_or_label_mismatch"


def write_audit(rows, all_diffs, audit_size=15):
    audit = {}
    for score_name in ["equal_all_axes", "safety_truth_axes", "assistant_quality_axes"]:
        diffs = all_diffs[score_name]
        for dataset in sorted({row["dataset"] for row in rows}):
            idxs = [i for i, row in enumerate(rows) if row["dataset"] == dataset and diffs[i] < 0]
            ranked = sorted(idxs, key=lambda i: diffs[i])[:audit_size]
            audit[f"{dataset}__{score_name}"] = [
                {
                    "id": rows[i]["id"],
                    "gap": diffs[i],
                    "category": disagreement_category(rows[i]),
                    "preferred_length": rows[i]["preferred_length"],
                    "other_length": rows[i]["other_length"],
                    "preferred_sentiment": rows[i]["preferred_sentiment"],
                    "other_sentiment": rows[i]["other_sentiment"],
                    "prompt_excerpt": short(rows[i]["prompt"], 220),
                    "artifact_preferred_excerpt": short(rows[i]["preferred"], 260),
                    "embedding_preferred_excerpt": short(rows[i]["other"], 260),
                }
                for i in ranked
            ]
    lines = [
        "# Phase 6 High-Confidence Disagreement Audit",
        "",
        "These are cases where an aggregate embedding sensor prefers the non-artifact response. They are not automatically failures; they are audit candidates.",
        "",
    ]
    for key, items in audit.items():
        lines.extend([f"## {key}", ""])
        for item in items:
            lines.extend(
                [
                    f"### {item['id']} - {item['category']}",
                    "",
                    f"- Gap artifact-minus-embedding: {item['gap']:.5f}",
                    f"- Lengths artifact/embedding: {item['preferred_length']} / {item['other_length']}",
                    f"- Sentiment artifact/embedding: {item['preferred_sentiment']:.3f} / {item['other_sentiment']:.3f}",
                    f"- Prompt: {item['prompt_excerpt']}",
                    f"- Artifact preferred: {item['artifact_preferred_excerpt']}",
                    f"- Embedding preferred: {item['embedding_preferred_excerpt']}",
                    "",
                ]
            )
    write_text(OUT / "disagreement_audit.md", "\n".join(lines))
    return audit


def make_summary(results):
    lines = [
        "# Phase 6 Multi-Sensor Evaluative Axis Probe",
        "",
        f"Run timestamp: {results['timestamp']}",
        f"Embedding model: `{results['model']}`",
        "",
        "## Protocol",
        "",
        "This phase treats every dataset label as an imperfect sensor, not as ground truth. The frozen evaluative basis has eight axes: broad good/bad, harm reduction, truth correction, calibration, usefulness, non-sycophancy, risk disclosure, and agency respect. No axis weights were fit to any dataset.",
        "",
        "Datasets/sensors:",
        "- `hh_chosen`: Anthropic HH chosen/rejected label.",
        "- `pku_better`: PKU-SafeRLHF better-response label.",
        "- `pku_safer`: PKU-SafeRLHF safer-response label on the same prompts.",
        "- `shp_reddit`: Stanford SHP higher-scored Reddit answer.",
        "",
        "## Results",
        "",
    ]
    if "partial_cache" in results:
        partial = results["partial_cache"]
        note = [
            "## Partial Cache Note",
            "",
            "This run was scored from a quota-limited partial embedding cache. Only complete preferred/other pairs were included; missing pairs were not imputed.",
            "",
            f"- Complete pairs scored: {partial['complete_pairs']} of {partial['requested_pairs']} requested rows.",
            f"- Candidate texts embedded: {partial['candidate_texts_done']} of {partial['candidate_texts_total']}.",
            "",
        ]
        lines[5:5] = note
    for dataset, block in results["datasets"].items():
        lines.extend([f"### {dataset}", "", f"Sensor: {block['label_name']}; n={block['n']}", ""])
        baselines = block["baselines"]
        lines.append(
            f"Baselines: length {baselines['length_prefer_longer']['agreement']:.1%}; "
            f"sentiment {baselines['sentiment_prefer_positive']['agreement']:.1%}."
        )
        lines.append("")
        best_axes = sorted(block["axes"].items(), key=lambda item: item[1]["agreement"], reverse=True)[:5]
        lines.append("Top axes:")
        for name, metric in best_axes:
            ci = metric["ci95"]
            lines.append(
                f"- `{name}`: overlap {metric['agreement']:.1%} "
                f"(95% bootstrap {ci[0]:.1%}-{ci[1]:.1%}); "
                f"MI {metric['mutual_information_bits']:.4f} bits; p={metric['p_vs_random']:.3g}"
            )
        lines.append("")
        lines.append("Aggregates:")
        for name, metric in sorted(block["aggregates"].items(), key=lambda item: item[1]["agreement"], reverse=True):
            ci = metric["ci95"]
            lines.append(
                f"- `{name}`: overlap {metric['agreement']:.1%} "
                f"(95% bootstrap {ci[0]:.1%}-{ci[1]:.1%}); "
                f"MI {metric['mutual_information_bits']:.4f} bits; p={metric['p_vs_random']:.3g}"
            )
        lines.append("")
    lines.extend(
        [
            "## Interpretation",
            "",
            "The target is not dataset imitation. Above-chance overlap means a cheap embedding sensor shares information with an expensive or socially-produced preference artifact. Below-chance or low overlap can mean the sensor is wrong, the artifact is wrong, or the two are measuring different objectives.",
            "",
            "The main research question becomes cost-to-signal and disagreement quality: does this nearly-free evaluative basis add useful pressure when combined with RLHF, RLAIF, LLM judges, or data filtering?",
        ]
    )
    return "\n".join(lines)


def append_log(results):
    best_lines = []
    for dataset, block in results["datasets"].items():
        best_agg = max(block["aggregates"].items(), key=lambda item: item[1]["agreement"])
        best_axis = max(block["axes"].items(), key=lambda item: item[1]["agreement"])
        best_lines.append(
            f"- {dataset}: best axis `{best_axis[0]}` {best_axis[1]['agreement']:.1%}; "
            f"best aggregate `{best_agg[0]}` {best_agg[1]['agreement']:.1%}; "
            f"length {block['baselines']['length_prefer_longer']['agreement']:.1%}; "
            f"sentiment {block['baselines']['sentiment_prefer_positive']['agreement']:.1%}."
        )
    entry = f"""
## {now_label()} - Phase 6: Multi-Sensor Evaluative Axis Probe

### What was done
Ran a frozen multi-axis embedding evaluator across multiple imperfect preference
artifacts rather than treating HH-RLHF as the authority. Datasets were
Anthropic HH, PKU-SafeRLHF better labels, PKU-SafeRLHF safer labels, and
Stanford SHP Reddit-score labels.

### Key results
{chr(10).