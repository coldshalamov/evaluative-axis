#!/usr/bin/env python3
"""Shared scoring recipe for evaluative-axis experiments.

Confirmatory runs use a fixed recipe:
- embed responses as ``User: {prompt}\nAssistant: {response}``
- build one bipolar axis per quality term from positive and negative anchors
- score by projection onto the normalized positive-minus-negative direction
- combine multiple axes only after each axis has been scored independently

Cosine-to-positive is retained as an explicit ablation/diagnostic method,
especially for BGE-M3, not as the cross-model default. Compound anchor strings
that join multiple concepts are ablation-only; default recipes should keep
concepts as separate axes.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

DEFAULT_SCORING_METHOD = "bipolar"
COSINE_POSITIVE_METHOD = "cosine_positive"
STANDARD_FRAMING = "user_assistant"
COMPOUND_ANCHOR_POLICY = "disallowed_as_default"

SCORING_RECIPE: dict[str, Any] = {
    "default_method": DEFAULT_SCORING_METHOD,
    "default_framing": STANDARD_FRAMING,
    "response_format": "User: {prompt}\\nAssistant: {response}",
    "cosine_positive_policy": (
        "explicit BGE-M3 diagnostic or pre-registered ablation only; "
        "not a post-hoc per-axis default"
    ),
    "bipolar_policy": (
        "required for confirmatory cross-model results, unknown models, "
        "and paper headline tables unless a variant is declared separately"
    ),
    "compound_anchor_policy": (
        "do not join multiple quality concepts into one anchor string; "
        "score component axes independently and then combine scores or votes"
    ),
}


def recipe_metadata(experiment_kind: str) -> dict[str, Any]:
    """Return JSON-serializable metadata for result files."""
    return {"experiment_kind": experiment_kind, **SCORING_RECIPE}


def framed_response(case: dict[str, str], key: str) -> str:
    """Standard prompt/response framing required by the methodology."""
    return f"User: {case['prompt']}\nAssistant: {case[key]}"


def mean_vector(emb: np.ndarray) -> np.ndarray:
    """Accept a single embedding or a stack and return one vector."""
    arr = np.asarray(emb)
    if arr.ndim == 1:
        return arr
    return arr.mean(axis=0)


def normalized_bipolar_axis(pos_emb: np.ndarray, neg_emb: np.ndarray) -> np.ndarray:
    """Normalize mean(positive anchors) - mean(negative anchors)."""
    axis = mean_vector(pos_emb) - mean_vector(neg_emb)
    return axis / (np.linalg.norm(axis) + 1e-12)


def cosine_to_anchor(response_embs: np.ndarray, anchor_emb: np.ndarray) -> np.ndarray:
    """Cosine similarity from each response embedding to one anchor vector."""
    anchor = mean_vector(anchor_emb)
    anchor_norm = np.linalg.norm(anchor) + 1e-12
    response_norms = np.linalg.norm(response_embs, axis=1) + 1e-12
    return (response_embs @ anchor) / (response_norms * anchor_norm)


def bipolar_scores(
    response_embs: np.ndarray, pos_emb: np.ndarray, neg_emb: np.ndarray
) -> np.ndarray:
    """Default confirmatory score: projection onto normalized bipolar axis."""
    return response_embs @ normalized_bipolar_axis(pos_emb, neg_emb)


def cosine_positive_scores(response_embs: np.ndarray, pos_emb: np.ndarray) -> np.ndarray:
    """Diagnostic/ablation score: cosine similarity to the positive anchor."""
    return cosine_to_anchor(response_embs, pos_emb)


def pairwise_accuracy(
    better_scores: Sequence[float] | np.ndarray,
    worse_scores: Sequence[float] | np.ndarray,
) -> float:
    """Fraction where better > worse; exact ties count as half."""
    margins = np.asarray(better_scores) - np.asarray(worse_scores)
    correct = float(np.sum(margins > 0))
    ties = float(np.sum(margins == 0))
    return (correct + 0.5 * ties) / len(margins)


def resolve_scoring_method(model_name: str, purpose: str = "confirmatory") -> str:
    """Resolve allowed scoring methods without post-hoc per-axis selection."""
    if purpose in {"confirmatory", "cross_model", "unknown_model"}:
        return DEFAULT_SCORING_METHOD
    if purpose == "bge_m3_cospos_diagnostic":
        if model_name == "BAAI/bge-m3":
            return COSINE_POSITIVE_METHOD
        raise ValueError("cosine_positive diagnostic is only predeclared for BGE-M3")
    if purpose == "explicit_cospos_ablation":
        return COSINE_POSITIVE_METHOD
    raise ValueError(f"unknown scoring purpose: {purpose}")


def assert_default_anchor_texts(positive: Sequence[str], negative: Sequence[str]) -> None:
    """Guard default runs against cross-concept compound anchor strings."""
    compound_markers = (" and ", " but ", ",")
    texts = list(positive) + list(negative)
    compounds = [
        text for text in texts
        if any(marker in f" {text.lower()} " for marker in compound_markers)
    ]
    if compounds:
        raise ValueError(
            "compound anchor strings are ablation-only; score component axes "
            f"independently instead: {compounds}"
        )
