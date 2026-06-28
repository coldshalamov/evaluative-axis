#!/usr/bin/env python3
"""Generate publication-quality figures for the paper.

Figures:
1. Evaluative axis overview: how axis construction works (schematic)
2. Model landscape: accuracy by model scale for key axes
3. Anchor vocabulary comparison: single words vs ML-jargon vs baselines
4. Per-category heatmap: which axes capture which dimensions
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "paper" / "figures"
OUT.mkdir(parents=True, exist_ok=True)


def fig1_model_landscape():
    """Bar chart: best axis accuracy per model, showing frontier gap."""

    models = [
        "BGE-small\n(33M)", "GTE-base\n(109M)", "Snowflake-M\n(109M)",
        "Jina-small\n(33M)", "BGE-base\n(109M)", "Nomic\n(137M)",
        "BGE-M3\n(568M)", "Qwen3\n(600M)", "Gemini\n(??)"
    ]

    # Best targeted axis accuracy on 50-case battery
    best_targeted = [
        0.36, 0.44, 0.72,
        0.38, 0.42, 0.56,
        0.80, 0.54, 0.98
    ]

    # Raw good/bad accuracy
    good_bad = [
        0.14, 0.22, 0.48,
        0.24, 0.28, 0.12,
        0.16, 0.20, 0.26
    ]

    x = np.arange(len(models))
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 5))
    bars1 = ax.bar(x - width/2, best_targeted, width, label='Best targeted axis',
                   color='#2196F3', edgecolor='white', linewidth=0.5)
    bars2 = ax.bar(x + width/2, good_bad, width, label='Raw good/bad',
                   color='#FF9800', edgecolor='white', linewidth=0.5)

    ax.axhline(y=0.50, color='gray', linestyle='--', linewidth=1, alpha=0.5, label='Chance')
    ax.set_ylabel('Accuracy (50-case battery)', fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=9)
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=10, loc='upper left')
    ax.set_title('Evaluative Signal by Model Scale', fontsize=14, pad=10)

    # Annotate frontier gap
    ax.annotate('', xy=(8.2, 0.98), xytext=(8.2, 0.80),
                arrowprops=dict(arrowstyle='<->', color='red', lw=1.5))
    ax.text(8.5, 0.89, 'Frontier\ngap', ha='left', va='center',
            fontsize=9, color='red', fontweight='bold')

    plt.tight_layout()
    plt.savefig(OUT / "fig1_model_landscape.png", dpi=300, bbox_inches='tight')
    plt.savefig(OUT / "fig1_model_landscape.pdf", bbox_inches='tight')
    plt.close()
    print("  Saved fig1_model_landscape")


def fig2_anchor_vocabulary():
    """Grouped bar chart: single words vs ML-jargon vs baselines on 3 models."""

    categories = ['Snowflake-M\n(109M)', 'BGE-M3\n(568M)', 'Nomic\n(137M)']

    methods = {
        'Best ML-jargon axis': [0.72, 0.80, 0.56],
        '"Careful"/"Reckless"': [0.52, 0.58, 0.62],
        '"Moderate"/"Excessive"': [0.48, 0.50, 0.40],
        'Prompt-response cosine': [0.40, 0.28, 0.26],
        'Response length': [0.51, 0.51, 0.51],
        'Raw "Good"/"Bad"': [0.48, 0.16, 0.12],
    }

    x = np.arange(len(categories))
    n_methods = len(methods)
    width = 0.12
    offsets = np.linspace(-(n_methods-1)*width/2, (n_methods-1)*width/2, n_methods)

    colors = ['#1976D2', '#43A047', '#7CB342', '#EF6C00', '#9E9E9E', '#E53935']

    fig, ax = plt.subplots(figsize=(10, 5))

    for i, (name, values) in enumerate(methods.items()):
        ax.bar(x + offsets[i], values, width, label=name,
               color=colors[i], edgecolor='white', linewidth=0.5)

    ax.axhline(y=0.50, color='gray', linestyle='--', linewidth=1, alpha=0.5)
    ax.set_ylabel('Accuracy (50-case battery)', fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=11)
    ax.set_ylim(0, 0.90)
    ax.legend(fontsize=8.5, loc='upper right', ncol=2)
    ax.set_title('Anchor Vocabulary and Baselines Comparison', fontsize=14, pad=10)
    ax.text(0.02, 0.52, 'chance', fontsize=8, color='gray', transform=ax.get_yaxis_transform())

    plt.tight_layout()
    plt.savefig(OUT / "fig2_anchor_vocabulary.png", dpi=300, bbox_inches='tight')
    plt.savefig(OUT / "fig2_anchor_vocabulary.pdf", bbox_inches='tight')
    plt.close()
    print("  Saved fig2_anchor_vocabulary")


def fig3_category_heatmap():
    """Heatmap: per-category accuracy for key axes on one model (Nomic)."""

    # Load category breakdown results
    cat_path = ROOT / "notes" / "research_cycles" / "category_breakdown" / "category_results.json"
    if not cat_path.exists():
        print("  Skipping fig3: category_results.json not found")
        return

    data = json.loads(cat_path.read_text())
    model = "nomic-ai/nomic-embed-text-v1.5"
    results = data[model]

    axes = [
        "ml/anti_sycophancy",
        "ml/persona_honesty",
        "ml/harm_reduction",
        "single/careful_reckless",
        "single/moderate_excessive",
        "single/noble_base",
        "single/kind_cruel",
        "single/good_bad",
    ]

    categories = [
        "anti_sycophancy", "persona_honesty", "harm_reduction",
        "truthfulness", "reasoning_rigor", "helpfulness",
        "context_binding", "mixed"
    ]

    cat_labels = [
        "Anti-\nsycophancy", "Persona\nhonesty", "Harm\nreduction",
        "Truthfulness", "Reasoning\nrigor", "Helpfulness",
        "Context\nbinding", "Mixed\ntradeoff"
    ]

    axis_labels = [
        "ML: anti-syc.", "ML: persona", "ML: harm",
        "Careful", "Moderate", "Noble", "Kind", "Good/Bad"
    ]

    matrix = np.zeros((len(axes), len(categories)))
    for i, ax_name in enumerate(axes):
        for j, cat in enumerate(categories):
            matrix[i, j] = results[ax_name]["per_category"][cat]

    fig, ax = plt.subplots(figsize=(10, 6))
    im = ax.imshow(matrix, cmap='RdYlGn', vmin=0, vmax=1, aspect='auto')

    ax.set_xticks(np.arange(len(categories)))
    ax.set_yticks(np.arange(len(axes)))
    ax.set_xticklabels(cat_labels, fontsize=9)
    ax.set_yticklabels(axis_labels, fontsize=10)

    # Add text annotations
    for i in range(len(axes)):
        for j in range(len(categories)):
            val = matrix[i, j]
            color = 'white' if val < 0.3 or val > 0.7 else 'black'
            ax.text(j, i, f'{val:.0%}', ha='center', va='center',
                    fontsize=8, color=color, fontweight='bold')

    ax.set_title('Per-Category Accuracy (Nomic v1.5, 137M)', fontsize=14, pad=10)
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('Accuracy', fontsize=10)

    plt.tight_layout()
    plt.savefig(OUT / "fig3_category_heatmap.png", dpi=300, bbox_inches='tight')
    plt.savefig(OUT / "fig3_category_heatmap.pdf", bbox_inches='tight')
    plt.close()
    print("  Saved fig3_category_heatmap")


def fig4_gemini_targeted():
    """Bar chart: Gemini targeted axes performance."""

    axes = [
        'Anti-\nsycophancy', 'Persona\nhonesty', 'Harm\nreduction',
        'Truthfulness', 'Combined\ntargeted', 'General\nevaluative',
        'Best proxy\n(useful)', 'Sentence\ngood/bad', 'Raw\ngood/bad'
    ]

    accuracies = [0.98, 0.96, 0.94, 0.90, 0.86, 0.46, 0.42, 0.30, 0.26]

    colors = ['#1565C0' if a > 0.50 else '#E53935' for a in accuracies]

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(range(len(axes)), accuracies, color=colors,
                  edgecolor='white', linewidth=0.5)

    ax.axhline(y=0.50, color='gray', linestyle='--', linewidth=1, alpha=0.5)
    ax.set_ylabel('Accuracy (50-case battery)', fontsize=12)
    ax.set_xticks(range(len(axes)))
    ax.set_xticklabels(axes, fontsize=9)
    ax.set_ylim(0, 1.05)
    ax.set_title('Gemini Embedding 2: Targeted vs Broad Axes', fontsize=14, pad=10)
    ax.text(0.02, 0.52, 'chance', fontsize=8, color='gray', transform=ax.get_yaxis_transform())

    # Annotate p-values for significant results
    for i, (name, acc) in enumerate(zip(axes, accuracies)):
        if acc > 0.80:
            ax.text(i, acc + 0.02, f'{acc:.0%}', ha='center', va='bottom',
                    fontsize=9, fontweight='bold')

    plt.tight_layout()
    plt.savefig(OUT / "fig4_gemini_targeted.png", dpi=300, bbox_inches='tight')
    plt.savefig(OUT / "fig4_gemini_targeted.pdf", bbox_inches='tight')
    plt.close()
    print("  Saved fig4_gemini_targeted")


if __name__ == "__main__":
    print("Generating paper figures...")
    fig1_model_landscape()
    fig2_anchor_vocabulary()
    fig3_category_heatmap()
    fig4_gemini_targeted()
    print(f"\nAll figures saved to {OUT}")
