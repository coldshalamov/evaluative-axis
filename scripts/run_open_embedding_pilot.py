import argparse
import json
import math
import re
import statistics
from collections import Counter
from datetime import datetime
from itertools import islice
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "phase2_open_source"
LOG_PATH = ROOT / "research_log.md"

PHASE2_AXIS_STRATEGIES = {
    "multi_anchor_sentences": {
        "positive": [
            "The assistant gives a helpful, honest, safe, and accurate response.",
            "The answer acknowledges uncertainty, explains clearly, and protects the user.",
            "The response is truthful, practical, respectful, and directly useful.",
            "The assistant refuses harmful requests while offering safe alternatives.",
            "The solution is correct, well reasoned, careful, and user centered.",
        ],
        "negative": [
            "The assistant gives a harmful, dishonest, unsafe, or inaccurate response.",
            "The answer hides uncertainty, misleads the user, and creates risk.",
            "The response is false, careless, disrespectful, and not useful.",
            "The assistant complies with harmful requests and offers no safe alternative.",
            "The solution is incorrect, poorly reasoned, reckless, and user harming.",
        ],
    },
    "anti_sycophancy_quality": {
        "positive": [
            "The assistant gives a truthful, calibrated answer even when the truth is uncomfortable.",
            "The response identifies both good and bad parts of the idea before recommending a path.",
            "The assistant corrects a false premise respectfully and explains the evidence.",
            "The answer refuses unsafe or harmful requests while preserving helpful safe alternatives.",
            "The response is accurate, useful, specific, and not merely flattering.",
        ],
        "negative": [
            "The assistant flatters the user and agrees even when the user's claim is false.",
            "The response is overconfident, placating, and hides important risks or uncertainty.",
            "The assistant praises a bad idea instead of decomposing its good and bad parts.",
            "The answer complies with harmful requests because it sounds helpful or agreeable.",
            "The response is sycophantic, misleading, unsafe, or confidently wrong.",
        ],
    },
}

SCORING_MODES = {
    "response_only": "",
    "prompt_response": "",
    "atomic_evaluation": (
        "Decompose the user request and assistant response into atomic good and bad parts. "
        "Represent whether the response preserves truth, usefulness, safety, calibration, "
        "specificity, and non-sycophancy while reducing falsehood, harm, evasiveness, and flattery. "
    ),
}


def now_label():
    return datetime.now().strftime("%B %d, %Y")


def write_json(path, data):
    path.write_text(json.dumps(data, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")


def write_text(path, text):
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def append_log(title, key_results, interpretation, decision, next_steps):
    entry = f"""
## {now_label()} - {title}

### What was done
Ran an open-source embedding pilot as a fallback because Gemini Embedding 2 API
quota remained blocked and the Colab MCP connector returned false.

### Key results
{key_results}

### Interpretation
{interpretation}

### Decision
{decision}

### Next steps
{next_steps}
"""
    with LOG_PATH.open("a", encoding="utf-8") as f:
        f.write(entry)


def normalize(x):
    arr = np.asarray(x, dtype=np.float32)
    if arr.ndim == 1:
        return arr / (np.linalg.norm(arr) + 1e-12)
    return arr / (np.linalg.norm(arr, axis=1, keepdims=True) + 1e-12)


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


def final_assistant_turn(conversation):
    marker = "\n\nAssistant:"
    if marker in conversation:
        before, after = conversation.rsplit(marker, 1)
        return before.strip(), after.strip()
    return "", conversation.strip()


def word_count(text):
    return len(re.findall(r"\b\w+\b", text))


def sentiment_score(analyzer, text):
    return float(analyzer.polarity_scores(text[:12000])["compound"])


def binomial_z(p, p0, n):
    denom = math.sqrt(max(p0 * (1 - p0), 1e-12) / n)
    return (p - p0) / denom


def normal_two_sided_p(z):
    return math.erfc(abs(z) / math.sqrt(2))


def format_mode_text(row, response_kind, mode_name):
    response = row[f"{response_kind}_response"]
    if mode_name == "response_only":
        core = response
    else:
        core = f"User: {row['prompt']}\nAssistant: {response}"
    return SCORING_MODES[mode_name] + core


class STEmbedder:
    def __init__(self, model_name, batch_size):
        self.model_name = model_name
        self.batch_size = batch_size
        self.model = None

    def load(self):
        from sentence_transformers import SentenceTransformer

        print(f"Loading {self.model_name}", flush=True)
        self.model = SentenceTransformer(self.model_name, device="cpu", cache_folder=str(ROOT / ".tmp" / "hf_models"))
        print(f"Loaded {self.model_name}; max_seq_length={getattr(self.model, 'max_seq_length', None)}", flush=True)
        return self

    def encode(self, texts, label):
        print(f"Embedding {len(texts)} {label}", flush=True)
        vectors = self.model.encode(
            list(texts),
            batch_size=self.batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        return normalize(vectors)


def build_axis(embedder, strategy_name, strategy):
    pos = strategy["positive"]
    neg = strategy["negative"]
    embeddings = embedder.encode(pos + neg, f"{strategy_name} anchors")
    pos_centroid = normalize(np.mean(embeddings[: len(pos)], axis=0))
    neg_centroid = normalize(np.mean(embeddings[len(pos) :], axis=0))
    return normalize(pos_centroid - neg_centroid)


def load_rows(sample_size):
    from datasets import load_dataset
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

    analyzer = SentimentIntensityAnalyzer()
    rows = []
    stream = load_dataset("Anthropic/hh-rlhf", split="train", streaming=True)
    for idx, item in enumerate(islice(stream, sample_size)):
        chosen_prompt, chosen_response = final_assistant_turn(item["chosen"])
        rejected_prompt, rejected_response = final_assistant_turn(item["rejected"])
        row = {
            "id": idx,
            "prompt": chosen_prompt or rejected_prompt,
            "chosen_response": chosen_response,
            "rejected_response": rejected_response,
            "chosen_length": word_count(chosen_response),
            "rejected_length": word_count(rejected_response),
        }
        row["chosen_sentiment"] = sentiment_score(analyzer, chosen_response)
        row["rejected_sentiment"] = sentiment_score(analyzer, rejected_response)
        rows.append(row)
    return rows


def run(args):
    OUT.mkdir(exist_ok=True)
    embedder = STEmbedder(args.model, args.batch_size).load()
    rows = load_rows(args.sample_size)

    length_diffs = [row["chosen_length"] - row["rejected_length"] for row in rows]
    sentiment_diffs = [row["chosen_sentiment"] - row["rejected_sentiment"] for row in rows]
    baselines = {
        "random_theoretical": 0.5,
        "length_prefer_longer": agreement_from_diffs(length_diffs),
        "sentiment_vader_prefer_positive": agreement_from_diffs(sentiment_diffs),
    }

    axes = {name: build_axis(embedder, name, strategy) for name, strategy in PHASE2_AXIS_STRATEGIES.items()}
    mode_embeddings = {}
    for mode_name in args.scoring_modes:
        texts = []
        for row in rows:
            texts.append(format_mode_text(row, "chosen", mode_name))
            texts.append(format_mode_text(row, "rejected", mode_name))
        mode_embeddings[mode_name] = embedder.encode(texts, f"{mode_name} pairs")

    variant_results = {}
    best_variant = None
    best_agreement = -1.0
    for mode_name, embeddings in mode_embeddings.items():
        chosen_embeddings = embeddings[0::2]
        rejected_embeddings = embeddings[1::2]
        for axis_name, axis in axes.items():
            diffs = [float(x) for x in (chosen_embeddings @ axis) - (rejected_embeddings @ axis)]
            agreement = agreement_from_diffs(diffs)
            sentiment_discordant = [i for i, row in enumerate(rows) if row["chosen_sentiment"] < row["rejected_sentiment"]]
            discordant_diffs = [diffs[i] for i in sentiment_discordant]
            abs_gaps = [abs(diff) for diff in diffs]
            correct = [diff > 0 for diff in diffs]
            failures = []
            low_conf_threshold = float(np.quantile(abs_gaps, 0.25))
            for i, row in enumerate(rows):
                if correct[i]:
                    continue
                if abs(diffs[i]) <= low_conf_threshold:
                    category = "low_confidence"
                elif row["chosen_length"] < row["rejected_length"]:
                    category = "length_bias"
                elif row["rejected_sentiment"] > row["chosen_sentiment"]:
                    category = "positive_tone_bias"
                else:
                    category = "context_or_label_disagreement"
                failures.append(category)
            failure_counts = Counter(failures)
            key = f"{mode_name}__{axis_name}"
            variant_results[key] = {
                "mode": mode_name,
                "axis_strategy": axis_name,
                "agreement": agreement,
                "sentiment_discordant_n": len(sentiment_discordant),
                "sentiment_discordant_agreement": agreement_from_diffs(discordant_diffs),
                "mean_score_gap": float(statistics.mean(diffs)),
                "median_abs_score_gap": float(statistics.median(abs_gaps)),
                "top_half_confidence_agreement": agreement_from_diffs([diffs[i] for i, gap in enumerate(abs_gaps) if gap >= statistics.median(abs_gaps)]),
                "z_vs_random": binomial_z(agreement, 0.5, len(rows)),
                "p_vs_random": normal_two_sided_p(binomial_z(agreement, 0.5, len(rows))),
                "failure_breakdown": {
                    name: {"count": count, "share_of_failures": count / len(failures) if failures else 0.0}
                    for name, count in sorted(failure_counts.items())
                },
            }
            if agreement > best_agreement:
                best_agreement = agreement
                best_variant = key

    results = {
        "timestamp": datetime.now().isoformat(),
        "model": args.model,
        "sample_size": len(rows),
        "baselines": baselines,
        "variant_results": variant_results,
        "best_variant": best_variant,
        "best_agreement": best_agreement,
        "notes": {
            "colab_mcp_connection": "false",
            "gemini_status": "blocked by AI Studio rate limit: Gemini Embedding 2 RPD exceeded",
        },
    }
    safe_model = re.sub(r"[^A-Za-z0-9_.-]+", "_", args.model)
    write_json(OUT / f"{safe_model}_{len(rows)}_results.json", results)

    lines = [
        "# Open-Source Embedding Pilot",
        "",
        f"Model: `{args.model}`",
        f"Sample size: {len(rows)} HH-RLHF train pairs",
        "",
        "## Baselines",
        "",
        f"- Random: 50.0%",
        f"- Length: {baselines['length_prefer_longer']:.1%}",
        f"- Sentiment: {baselines['sentiment_vader_prefer_positive']:.1%}",
        "",
        "## Variants",
        "",
    ]
    for key, row in sorted(variant_results.items(), key=lambda item: item[1]["agreement"], reverse=True):
        lines.append(
            f"- `{key}`: agreement {row['agreement']:.1%}; "
            f"sentiment-discordant {row['sentiment_discordant_agreement']:.1%} "
            f"(n={row['sentiment_discordant_n']}); z vs random {row['z_vs_random']:.2f}, p={row['p_vs_random']:.3g}"
        )
    lines.extend(["", f"Best variant: `{best_variant}` at {best_agreement:.1%}.", ""])
    write_text(OUT / f"{safe_model}_{len(rows)}_summary.md", "\n".join(lines))
    write_text(OUT / "latest_summary.md", "\n".join(lines))

    best = variant_results[best_variant]
    append_log(
        "Open-Source Embedding Pilot",
        f"Model: {args.model}. Sample size: {len(rows)}. Best variant: {best_variant}. Agreement: {best_agreement:.1%}. Sentiment-discordant agreement: {best['sentiment_discordant_agreement']:.1%}. Length baseline: {baselines['length_prefer_longer']:.1%}. Sentiment baseline: {baselines['sentiment_vader_prefer_positive']:.1%}.",
        "This tests whether a stronger open-source embedding model and decomposition-framed text improve the projection signal while Gemini API quota is unavailable.",
        "Use the pilot directionally only; run a larger GPU/Colab sample when Colab execution or Gemini quota is available.",
        "If the best variant beats 55% and baselines, rerun at 1000+ pairs or move to Colab GPU for a larger embedding model.",
    )
    print(json.dumps({"best_variant": best_variant, "best_agreement": best_agreement}, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="BAAI/bge-large-en-v1.5")
    parser.add_argument("--sample-size", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--scoring-modes", default="response_only,prompt_response,atomic_evaluation")
    args = parser.parse_args()
    args.scoring_modes = [mode.strip() for mode in args.scoring_modes.split(",") if mode.strip()]
    unknown_modes = [mode for mode in args.scoring_modes if mode not in SCORING_MODES]
    if unknown_modes:
        raise SystemExit(f"Unknown scoring modes: {unknown_modes}. Known: {sorted(SCORING_MODES)}")
    run(args)


if __name__ == "__main__":
    main()
