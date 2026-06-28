#!/usr/bin/env python3
"""Build a partner-readable packet from the current research-system outputs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def pct(value: float) -> str:
    return f"{value * 100:.1f}%"


def svg_escape(text: str) -> str:
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def horizontal_bar_chart_svg(
    *,
    title: str,
    subtitle: str,
    items: list[dict[str, Any]],
    width: int = 980,
    row_height: int = 46,
    left_margin: int = 300,
    right_margin: int = 120,
    top_margin: int = 88,
    bottom_margin: int = 40,
) -> str:
    height = top_margin + bottom_margin + row_height * len(items)
    chart_width = width - left_margin - right_margin
    scale_max = max(max(float(item["value"]) for item in items), 1.0)

    palette = {
        "gemini": "#2C6E49",
        "oss": "#8D99AE",
        "baseline": "#BC6C25",
        "control": "#6C757D",
        "failure": "#B02E0C",
        "mixed": "#355070",
    }

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#fcfbf7"/>',
        f'<text x="28" y="36" font-size="28" font-family="Georgia, serif" fill="#1f2933">{svg_escape(title)}</text>',
        f'<text x="28" y="64" font-size="15" font-family="Verdana, sans-serif" fill="#52606d">{svg_escape(subtitle)}</text>',
    ]

    for tick in range(0, 101, 20):
        x = left_margin + chart_width * (tick / 100)
        parts.append(
            f'<line x1="{x:.1f}" y1="{top_margin - 10}" x2="{x:.1f}" y2="{height - bottom_margin + 6}" '
            'stroke="#e6e8eb" stroke-width="1"/>'
        )
        parts.append(
            f'<text x="{x:.1f}" y="{height - 12}" text-anchor="middle" font-size="12" '
            'font-family="Verdana, sans-serif" fill="#7b8794">'
            f"{tick}%</text>"
        )

    for idx, item in enumerate(items):
        y = top_margin + idx * row_height
        value = float(item["value"])
        color = palette.get(str(item.get("style", "mixed")), "#355070")
        bar_width = chart_width * (value / scale_max) if scale_max else 0
        label = svg_escape(str(item["label"]))
        note = svg_escape(str(item.get("note", "")))
        parts.append(
            f'<text x="{left_margin - 14}" y="{y + 18}" text-anchor="end" font-size="16" '
            'font-family="Verdana, sans-serif" fill="#102a43">'
            f"{label}</text>"
        )
        if note:
            parts.append(
                f'<text x="{left_margin - 14}" y="{y + 34}" text-anchor="end" font-size="11" '
                'font-family="Verdana, sans-serif" fill="#7b8794">'
                f"{note}</text>"
            )
        parts.append(
            f'<rect x="{left_margin}" y="{y}" width="{chart_width}" height="18" rx="5" fill="#edf2f7"/>'
        )
        parts.append(
            f'<rect x="{left_margin}" y="{y}" width="{bar_width:.2f}" height="18" rx="5" fill="{color}"/>'
        )
        parts.append(
            f'<text x="{left_margin + min(bar_width + 8, chart_width - 2):.2f}" y="{y + 14}" '
            'font-size="13" font-family="Verdana, sans-serif" fill="#102a43">'
            f"{pct(value)}</text>"
        )

    parts.append("</svg>")
    return "\n".join(parts)


def build_packet_summary() -> dict[str, Any]:
    report = read_json(ROOT / "notes" / "research_system_v1" / "report" / "report.json")
    good_proxy_gemini = read_json(ROOT / "notes" / "research_system_v1" / "good_vs_proxy_conflicts_gemini_v1" / "summary.json")
    good_proxy_bge = read_json(ROOT / "notes" / "research_system_v1" / "good_vs_proxy_conflicts_bge_v1" / "summary.json")
    battery_v3 = read_json(ROOT / "notes" / "research_system_v1" / "battery_v3_gemini_direct_v1" / "summary.json")
    process_gemini = read_json(ROOT / "notes" / "research_system_v1" / "process_potential_error_repair_v1" / "summary.json")
    process_bge = read_json(ROOT / "notes" / "research_system_v1" / "process_potential_error_repair_bge_v1" / "summary.json")
    blind_review_pilot = read_json(
        ROOT / "notes" / "research_cycles" / "cycle_007_pairwise_blind_review_pilot" / "runs" / "combined_results.json"
    )
    blind_review_gemini = read_json(
        ROOT / "notes" / "research_cycles" / "cycle_008_gemini_openended_blind_review_pilot" / "runs" / "combined_results.json"
    )
    blind_review_bge_harm = read_json(
        ROOT
        / "notes"
        / "research_cycles"
        / "cycle_008_gemini_openended_blind_review_pilot"
        / "runs_bge_harm"
        / "direct_harm_reduction"
        / "analysis"
        / "aggregate.json"
    )
    local_word_sweep = read_json(
        ROOT / "notes" / "research_cycles" / "cycle_010_oss_good_vs_proxy_sweep" / "sweep_results.json"
    )

    lanes = report["lanes"]
    gates = report["gates"]
    best_local_raw = max(local_word_sweep, key=lambda row: row["comparison"]["raw_good_bad_accuracy"])
    best_local_sentence = max(local_word_sweep, key=lambda row: row["comparison"]["sentence_good_bad_accuracy"])
    best_local_proxy = max(local_word_sweep, key=lambda row: row["comparison"]["best_proxy_accuracy"])
    local_raw_under_30 = sum(
        1 for row in local_word_sweep if float(row["comparison"]["raw_good_bad_accuracy"]) < 0.30
    )

    return {
        "program_name": report["program_name"],
        "claim_gate_status": gates,
        "objective_selection": {
            "code": {
                "baseline": lanes["objective_code_gemini_curated_local"]["metrics"]["default.baseline_best_solve_rate"],
                "gemini": lanes["objective_code_gemini_curated_local"]["metrics"]["default.best_direct_solve_rate"],
                "oss_bge": lanes["objective_code_oss_colab"]["metrics"]["baai_bge_base_en_v1_5.best_direct_solve_rate"],
                "oss_mpnet": lanes["objective_code_oss_colab"]["metrics"]["sentence_transformers_all_mpnet_base_v2.best_direct_solve_rate"],
            },
            "math": {
                "baseline": lanes["objective_math_gemini_v1"]["metrics"]["default.baseline_best_solve_rate"],
                "gemini": lanes["objective_math_gemini_v1"]["metrics"]["default.best_direct_solve_rate"],
                "oss_bge": lanes["objective_math_oss_bge_v1"]["metrics"]["default.best_direct_solve_rate"],
            },
            "tool_interpretation": {
                "baseline": lanes["tool_interpretation_gemini_v1"]["metrics"]["default.baseline_best_solve_rate"],
                "gemini": lanes["tool_interpretation_gemini_v1"]["metrics"]["default.best_direct_solve_rate"],
                "oss_bge": lanes["tool_interpretation_oss_bge_v1"]["metrics"]["default.best_direct_solve_rate"],
            },
        },
        "word_vs_targeted": {
            "gemini_raw_good_bad": good_proxy_gemini["comparison"]["raw_good_bad_accuracy"],
            "gemini_sentence_good_bad": good_proxy_gemini["comparison"]["sentence_good_bad_accuracy"],
            "gemini_best_proxy_name": good_proxy_gemini["comparison"]["best_proxy_axis"],
            "gemini_best_proxy_accuracy": good_proxy_gemini["comparison"]["best_proxy_accuracy"],
            "bge_raw_good_bad": good_proxy_bge["comparison"]["raw_good_bad_accuracy"],
            "bge_best_proxy_name": good_proxy_bge["comparison"]["best_proxy_axis"],
            "bge_best_proxy_accuracy": good_proxy_bge["comparison"]["best_proxy_accuracy"],
            "gemini_direct_general_evaluative": battery_v3["metrics"]["direct_general_evaluative"]["accuracy"],
            "gemini_direct_combined": battery_v3["metrics"]["direct_combined"]["accuracy"],
            "gemini_direct_category_axis": battery_v3["metrics"]["direct_category_axis"]["accuracy"],
            "gemini_direct_truthfulness": battery_v3["metrics"]["direct_truthfulness"]["accuracy"],
            "gemini_direct_harm_reduction": battery_v3["metrics"]["direct_harm_reduction"]["accuracy"],
            "gemini_direct_persona_honesty": battery_v3["metrics"]["direct_persona_honesty"]["accuracy"],
            "gemini_direct_anti_sycophancy": battery_v3["metrics"]["direct_anti_sycophancy"]["accuracy"],
        },
        "process_signal": {
            "gemini_category_axis_dense": process_gemini["metrics"]["category_axis"]["dense_reward_localization_score"],
            "gemini_combined_dense": process_gemini["metrics"]["combined"]["dense_reward_localization_score"],
            "gemini_error_drop": process_gemini["metrics"]["category_axis"]["error_drop_accuracy"],
            "gemini_repair_rise": process_gemini["metrics"]["category_axis"]["repair_rise_accuracy"],
            "gemini_final_answer_only_combined_dense": process_gemini["metrics"]["final_answer_only_combined"]["dense_reward_localization_score"],
            "gemini_sentiment_dense": process_gemini["metrics"]["sentiment"]["dense_reward_localization_score"],
            "gemini_length_dense": process_gemini["metrics"]["length"]["dense_reward_localization_score"],
            "bge_category_axis_dense": process_bge["metrics"]["category_axis"]["dense_reward_localization_score"],
            "bge_combined_dense": process_bge["metrics"]["combined"]["dense_reward_localization_score"],
        },
        "blind_review_pilot": {
            f"{row['focus_method']}__vs__{row['baseline_method']}": {
                "focus_wins": row["focus_wins"],
                "baseline_wins": row["baseline_wins"],
                "ties": row["ties"],
                "focus_win_rate_decided": row["focus_win_rate_decided"],
            }
            for row in blind_review_pilot
        },
        "blind_review_gemini_followup": {
            f"{row['focus_method']}__vs__{row['baseline_method']}": {
                "focus_wins": row["focus_wins"],
                "baseline_wins": row["baseline_wins"],
                "ties": row["ties"],
                "focus_win_rate_decided": row["focus_win_rate_decided"],
            }
            for row in blind_review_gemini
        },
        "blind_review_bge_harm_followup": {
            row["comparison"].replace(" ", "_"): {
                "focus_wins": row["focus_wins"],
                "baseline_wins": row["baseline_wins"],
                "ties": row["ties"],
                "focus_win_rate_decided": row["focus_win_rate_decided"],
            }
            for row in blind_review_bge_harm
        },
        "local_word_sweep": {
            "best_local_raw_model": best_local_raw["model"],
            "best_local_raw_good_bad": best_local_raw["comparison"]["raw_good_bad_accuracy"],
            "best_local_sentence_model": best_local_sentence["model"],
            "best_local_sentence_good_bad": best_local_sentence["comparison"]["sentence_good_bad_accuracy"],
            "best_local_proxy_model": best_local_proxy["model"],
            "best_local_proxy_name": best_local_proxy["comparison"]["best_proxy_axis"],
            "best_local_proxy_accuracy": best_local_proxy["comparison"]["best_proxy_accuracy"],
            "local_models_raw_good_bad_below_30_count": local_raw_under_30,
            "local_models_total": len(local_word_sweep),
        },
    }


def build_figures(summary: dict[str, Any], output_dir: Path) -> dict[str, str]:
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    objective = summary["objective_selection"]
    selection_items = [
        {"label": "Code baseline", "value": objective["code"]["baseline"], "style": "baseline", "note": "best cheap baseline"},
        {"label": "Code Gemini", "value": objective["code"]["gemini"], "style": "gemini", "note": "best direct method"},
        {"label": "Code BGE-base", "value": objective["code"]["oss_bge"], "style": "oss", "note": "cheap OSS"},
        {"label": "Math baseline", "value": objective["math"]["baseline"], "style": "baseline", "note": "best cheap baseline"},
        {"label": "Math Gemini", "value": objective["math"]["gemini"], "style": "gemini", "note": "best direct method"},
        {"label": "Math BGE-base", "value": objective["math"]["oss_bge"], "style": "oss", "note": "cheap OSS"},
        {"label": "Tool baseline", "value": objective["tool_interpretation"]["baseline"], "style": "baseline", "note": "best cheap baseline"},
        {"label": "Tool Gemini", "value": objective["tool_interpretation"]["gemini"], "style": "gemini", "note": "best direct method"},
        {"label": "Tool BGE-base", "value": objective["tool_interpretation"]["oss_bge"], "style": "oss", "note": "cheap OSS"},
    ]
    selection_svg = horizontal_bar_chart_svg(
        title="Objective Reranking: Gemini Beats Cheap Baselines And OSS",
        subtitle="Solve-rate on frozen code, math, and tool-interpretation reranking suites.",
        items=selection_items,
    )
    selection_path = figures_dir / "figure_selection_lift.svg"
    write_text(selection_path, selection_svg)

    word = summary["word_vs_targeted"]
    evaluative_items = [
        {"label": "Gemini raw good/bad", "value": word["gemini_raw_good_bad"], "style": "failure", "note": "50-case conflict battery"},
        {"label": "Gemini sentence good/bad", "value": word["gemini_sentence_good_bad"], "style": "failure", "note": "same battery"},
        {"label": f"Gemini best proxy ({word['gemini_best_proxy_name']})", "value": word["gemini_best_proxy_accuracy"], "style": "mixed", "note": "best nearby word axis"},
        {"label": "Gemini general evaluative", "value": word["gemini_direct_general_evaluative"], "style": "mixed", "note": "richer broad axis"},
        {"label": "Gemini combined targeted axes", "value": word["gemini_direct_combined"], "style": "gemini", "note": "same 50 cases"},
        {"label": "Gemini category-routed axis", "value": word["gemini_direct_category_axis"], "style": "gemini", "note": "same 50 cases"},
        {"label": "Gemini truthfulness axis", "value": word["gemini_direct_truthfulness"], "style": "gemini", "note": "same 50 cases"},
        {"label": "Gemini harm-reduction axis", "value": word["gemini_direct_harm_reduction"], "style": "gemini", "note": "same 50 cases"},
        {"label": "Gemini persona-honesty axis", "value": word["gemini_direct_persona_honesty"], "style": "gemini", "note": "same 50 cases"},
        {"label": "Gemini anti-sycophancy axis", "value": word["gemini_direct_anti_sycophancy"], "style": "gemini", "note": "same 50 cases"},
    ]
    evaluative_svg = horizontal_bar_chart_svg(
        title="Raw good/bad Fails, Targeted Evaluative Axes Work",
        subtitle="All scores are on the exact same 50-case length-balanced conflict battery.",
        items=evaluative_items,
        row_height=42,
    )
    evaluative_path = figures_dir / "figure_word_vs_targeted.svg"
    write_text(evaluative_path, evaluative_svg)

    process = summary["process_signal"]
    process_items = [
        {"label": "Gemini combined process score", "value": process["gemini_combined_dense"], "style": "gemini", "note": "dense localization score"},
        {"label": "Gemini category-axis process score", "value": process["gemini_category_axis_dense"], "style": "gemini", "note": "dense localization score"},
        {"label": "BGE combined process score", "value": process["bge_combined_dense"], "style": "oss", "note": "cheap OSS"},
        {"label": "BGE category-axis process score", "value": process["bge_category_axis_dense"], "style": "oss", "note": "cheap OSS"},
        {"label": "Gemini sentiment control", "value": process["gemini_sentiment_dense"], "style": "control", "note": "cheap baseline"},
        {"label": "Gemini length control", "value": process["gemini_length_dense"], "style": "control", "note": "cheap baseline"},
        {"label": "Gemini final-answer-only", "value": process["gemini_final_answer_only_combined_dense"], "style": "control", "note": "not process-aware"},
    ]
    process_svg = horizontal_bar_chart_svg(
        title="Process Signal Is Real But Still Below The Frozen Training Gate",
        subtitle="Gemini tracks injected error and repair far better than cheap controls or cheap OSS, but the precommitted gate remains unmet.",
        items=process_items,
    )
    process_path = figures_dir / "figure_process_signal.svg"
    write_text(process_path, process_svg)

    return {
        "selection_lift": str(selection_path.relative_to(output_dir)).replace("\\", "/"),
        "word_vs_targeted": str(evaluative_path.relative_to(output_dir)).replace("\\", "/"),
        "process_signal": str(process_path.relative_to(output_dir)).replace("\\", "/"),
    }


def build_brief(summary: dict[str, Any], figures: dict[str, str]) -> str:
    gates = summary["claim_gate_status"]
    objective = summary["objective_selection"]
    word = summary["word_vs_targeted"]
    process = summary["process_signal"]
    blind_review = summary["blind_review_pilot"]
    blind_review_gemini = summary["blind_review_gemini_followup"]
    blind_review_bge_harm = summary["blind_review_bge_harm_followup"]
    local_words = summary["local_word_sweep"]
    lines = [
        "# Partner Packet V1",
        "",
        "## Thesis",
        "",
        "This repo tests whether evaluative embedding geometry can provide a cheap, deterministic signal for answer selection and eventually training.",
        "",
        "The current evidence supports a narrower and more defensible claim than the original strongest version:",
        "",
        "- embedding scoring can improve objective reranking across several domains;",
        "- stronger embedding models materially outperform cheap OSS embedders on the same frozen tasks;",
        "- raw one-word `good/bad` is not enough in the current zero-shot setup;",
        "- process sensitivity exists, but dense training-readiness is not yet established.",
        "",
        "## Current Claim Gate Status",
        "",
        f"- `capacity_code_gate`: `{gates['capacity_code_gate']['status']}`",
        f"- `behavior_basis_gate`: `{gates['behavior_basis_gate']['status']}`",
        f"- `capacity_cross_domain_gate`: `{gates['capacity_cross_domain_gate']['status']}`",
        f"- `cross_domain_selection_gate`: `{gates['cross_domain_selection_gate']['status']}`",
        f"- `process_potential_gate`: `{gates['process_potential_gate']['status']}`",
        f"- `training_readiness_gate`: `{gates['training_readiness_gate']['status']}`",
        "",
        "## Figure Set",
        "",
        f"![Objective Selection Lift]({figures['selection_lift']})",
        "",
        f"![Word Vs Targeted Axes]({figures['word_vs_targeted']})",
        "",
        f"![Process Signal]({figures['process_signal']})",
        "",
        "## Evidence Ladder",
        "",
        "### 1. Objective selection is now real evidence, not just dataset overlap",
        "",
        f"- code: Gemini `{pct(objective['code']['gemini'])}` vs best baseline `{pct(objective['code']['baseline'])}` vs BGE-base `{pct(objective['code']['oss_bge'])}`",
        f"- math: Gemini `{pct(objective['math']['gemini'])}` vs best baseline `{pct(objective['math']['baseline'])}` vs BGE-base `{pct(objective['math']['oss_bge'])}`",
        f"- tool interpretation: Gemini `{pct(objective['tool_interpretation']['gemini'])}` vs best baseline `{pct(objective['tool_interpretation']['baseline'])}` vs BGE-base `{pct(objective['tool_interpretation']['oss_bge'])}`",
        "",
        "This is the strongest external-facing part of the repo right now because the end metric is objective: the selected candidate either solves the task or it does not.",
        "",
        "### 2. The capability gap is large",
        "",
        "- Gemini clears every current cross-domain selection gate.",
        "- Cheap OSS encoders either match baseline at best or fall far short of Gemini.",
        "- On code, `all-mpnet-base-v2` falls below baseline while Gemini reaches 83.3%.",
        "",
        "This supports a real model-quality dependency even though parameter count is not directly observed here.",
        "",
        "### 3. The repo now has an honest negative result on raw `good/bad`",
        "",
        f"- Gemini raw `good/bad`: `{pct(word['gemini_raw_good_bad'])}` on the 50-case conflict battery",
        f"- Gemini sentence `This response is good/bad.`: `{pct(word['gemini_sentence_good_bad'])}`",
        f"- Gemini best nearby proxy (`{word['gemini_best_proxy_name']}`): `{pct(word['gemini_best_proxy_accuracy'])}`",
        f"- BGE raw `good/bad`: `{pct(word['bge_raw_good_bad'])}`",
        f"- best local raw `good/bad`: `{pct(local_words['best_local_raw_good_bad'])}` on `{local_words['best_local_raw_model']}`",
        f"- best local nearby proxy: `{pct(local_words['best_local_proxy_accuracy'])}` on `{local_words['best_local_proxy_model']}` via `{local_words['best_local_proxy_name']}`",
        "",
        f"Seven of the eight local models stayed below 30% on raw `good/bad`, so this is no longer only a Gemini-plus-BGE story.",
        "",
        "That result rules out the easiest overclaim. Under the current zero-shot measurement interface, raw one-word `good/bad` is not the strong signal the project hoped for.",
        "",
        "### 4. Targeted evaluative axes are strong on the same hard battery",
        "",
        f"- combined targeted axes: `{pct(word['gemini_direct_combined'])}`",
        f"- category-routed axis: `{pct(word['gemini_direct_category_axis'])}`",
        f"- truthfulness axis: `{pct(word['gemini_direct_truthfulness'])}`",
        f"- harm-reduction axis: `{pct(word['gemini_direct_harm_reduction'])}`",
        f"- persona-honesty axis: `{pct(word['gemini_direct_persona_honesty'])}`",
        f"- anti-sycophancy axis: `{pct(word['gemini_direct_anti_sycophancy'])}`",
        "",
        "So the useful current story is not \"good/bad already works by itself.\" It is that richer evaluative geometry is recoverable and practically useful.",
        "",
        "### 5. The bridge toward training is promising but incomplete",
        "",
        f"- Gemini process `error_drop_accuracy`: `{pct(process['gemini_error_drop'])}`",
        f"- Gemini process `repair_rise_accuracy`: `{pct(process['gemini_repair_rise'])}`",
        f"- Gemini combined dense localization: `{pct(process['gemini_combined_dense'])}`",
        f"- Gemini category-axis dense localization: `{pct(process['gemini_category_axis_dense'])}`",
        f"- Gemini final-answer-only dense localization: `{pct(process['gemini_final_answer_only_combined_dense'])}`",
        f"- BGE combined dense localization: `{pct(process['bge_combined_dense'])}`",
        "",
        "This is meaningful because the signal reacts to the bad step and the repair step, while final-answer-only and cheap lexical controls collapse. But the frozen gate still fails, so the repo should not yet claim dense-reward readiness.",
        "",
        "### 6. The open-ended lane improves with stronger embeddings but is still not decisive",
        "",
        f"- `direct_category_axis` vs `length`: `{pct(blind_review['direct_category_axis__vs__length']['focus_win_rate_decided'])}` decided win rate",
        f"- `direct_category_axis` vs `random`: `{pct(blind_review['direct_category_axis__vs__random']['focus_win_rate_decided'])}`",
        f"- `direct_category_axis` vs `refusal_heuristic`: `{pct(blind_review['direct_category_axis__vs__refusal_heuristic']['focus_win_rate_decided'])}`",
        f"- `direct_anti_sycophancy` vs `length`: `{pct(blind_review['direct_anti_sycophancy__vs__length']['focus_win_rate_decided'])}`",
        f"- `direct_anti_sycophancy` vs `refusal_heuristic`: `{pct(blind_review['direct_anti_sycophancy__vs__refusal_heuristic']['focus_win_rate_decided'])}`",
        "",
        f"- Gemini `direct_harm_reduction` vs `random`: `{pct(blind_review_gemini['direct_harm_reduction__vs__random']['focus_win_rate_decided'])}`",
        f"- Gemini `direct_harm_reduction` vs `length`: `{pct(blind_review_gemini['direct_harm_reduction__vs__length']['focus_win_rate_decided'])}`",
        f"- Gemini `direct_harm_reduction` vs `refusal_heuristic`: `{pct(blind_review_gemini['direct_harm_reduction__vs__refusal_heuristic']['focus_win_rate_decided'])}`",
        f"- matched BGE `direct_harm_reduction` vs `random`: `{pct(blind_review_bge_harm['direct_harm_reduction_vs_random']['focus_win_rate_decided'])}`",
        "",
        "This lane is still exploratory because it inherits the old length-biased candidate pool and uses blinded LLM adjudication rather than human gold review. But it is now more informative: stronger embeddings materially improve the open-ended blind-review results on the same pool, even though the lane still loses to length and refusal heuristics.",
        "",
        "## What This Packet Supports",
        "",
        "- evaluative embedding geometry is a credible cheap selection signal on objective tasks;",
        "- stronger embedding models appear much more suitable for the method than cheap OSS embedders;",
        "- a scalar-plus-basis view is more supported than a pure raw-word `good/bad` view;",
        "- process-aware scoring is plausible enough to justify more training-adjacent work;",
        "- the cheap OSS open-ended lane is now better characterized and should not be overclaimed.",
        "- the raw-word broad-axis story is now mapped across the local model family rather than only Gemini plus one BGE baseline.",
        "",
        "## What It Does Not Support Yet",
        "",
        "- that raw `good/bad` alone is already a robust evaluator;",
        "- that HH agreement alone proves anything decisive;",
        "- that the current process signal is strong enough for dense reward training;",
        "- that the repo already has blind human-adjudicated proof on open-ended generation;",
        "- that cheap open-source embedders are already good enough for the open-ended selection problem.",
        "",
        "## Best Next Moves",
        "",
        "1. Expand the process-potential suite from 12 traces toward 30-50 traces, especially for reasoning rigor, persona honesty, and harm-reduction edge cases.",
        "2. Expand the objective reranking suites from small pilots toward 30-50 tasks per domain while preserving objective end metrics.",
        "3. Build a clean no-leakage open-ended reranking packet with blind pairwise judging once candidate pools are length-balanced.",
        "4. Build the fresh no-leakage length-controlled open-ended pool so the stronger embedding family can be tested on a cleaner intervention benchmark.",
        "",
        "## Artifact Map",
        "",
        "- Source report: `notes/research_system_v1/report/report.md`",
        "- Packet summary JSON: `paper/partner_packet_v1/packet_summary.json`",
        "- Figure directory: `paper/partner_packet_v1/figures/`",
        "- This brief: `paper/partner_packet_v1/brief.md`",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    output_dir = ROOT / "paper" / "partner_packet_v1"
    output_dir.mkdir(parents=True, exist_ok=True)

    summary = build_packet_summary()
    figures = build_figures(summary, output_dir)
    brief = build_brief(summary, figures)

    write_json(output_dir / "packet_summary.json", summary)
    write_text(output_dir / "brief.md", brief)

    print(f"Wrote packet to {output_dir}")


if __name__ == "__main__":
    main()
