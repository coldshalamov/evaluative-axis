#!/usr/bin/env python3
"""Objective code reranking pilot with hidden unit tests."""

from __future__ import annotations

import argparse
import ast
import json
import random
import re
import statistics
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from run_gemini_rerun import GeminiEmbedder, get_api_keys, load_local_env, normalize, quota_retry_delay  # noqa: E402


CODE_AXES: dict[str, dict[str, list[str]]] = {
    "broad_good_bad": {
        "positive": [
            "The answer is good, useful, accurate, careful, and beneficial.",
            "The response solves the real task well and avoids avoidable mistakes.",
            "The assistant preserves what is good and reduces what is bad.",
        ],
        "negative": [
            "The answer is bad, useless, inaccurate, careless, or harmful.",
            "The response fails to solve the real task and creates avoidable mistakes.",
            "The assistant amplifies bad parts and loses good parts.",
        ],
    },
    "code_quality": {
        "positive": [
            "The code is correct, clear, robust, and handles edge cases.",
            "The implementation matches the specification and passes the tests.",
            "The function is safe, readable, and solves the requested problem.",
        ],
        "negative": [
            "The code is buggy, unclear, fragile, or misses edge cases.",
            "The implementation does not match the specification and fails the tests.",
            "The function is unsafe, confusing, or fails to solve the requested problem.",
        ],
    },
}


STYLE_PROMPTS = [
    "Write a concise but correct implementation.",
    "Write a robust implementation that handles edge cases carefully.",
    "Write a clear and readable implementation that still stays correct.",
]


GENERATION_MODEL = "gemini-2.0-flash"
MAX_GENERATION_RETRIES = 6


SAFE_BUILTINS: dict[str, Any] = {
    "abs": abs,
    "all": all,
    "any": any,
    "bool": bool,
    "dict": dict,
    "enumerate": enumerate,
    "filter": filter,
    "float": float,
    "int": int,
    "isinstance": isinstance,
    "len": len,
    "list": list,
    "map": map,
    "max": max,
    "min": min,
    "range": range,
    "reversed": reversed,
    "set": set,
    "sorted": sorted,
    "str": str,
    "sum": sum,
    "tuple": tuple,
    "zip": zip,
    "ValueError": ValueError,
}


RISKY_NAMES = {
    "__import__",
    "compile",
    "eval",
    "exec",
    "getattr",
    "globals",
    "help",
    "input",
    "locals",
    "open",
    "setattr",
    "vars",
}

RISKY_ROOTS = {
    "builtins",
    "ctypes",
    "os",
    "pathlib",
    "requests",
    "shutil",
    "socket",
    "subprocess",
    "sys",
    "urllib",
}


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")


def word_count(text: str) -> int:
    return len(re.findall(r"[A-Za-z_0-9]+", text))


def strip_code_fences(text: str) -> str:
    raw = text.strip()
    match = re.search(r"```(?:python)?\s*(.*?)```", raw, flags=re.IGNORECASE | re.DOTALL)
    if match:
        return match.group(1).strip()
    return raw


def raises_value_error(fn, *args):
    try:
        fn(*args)
    except ValueError:
        return True
    except Exception:
        return False
    return False


def safe_eval_check(check: str, fn) -> bool:
    env = {
        "__builtins__": {},
        "fn": fn,
        "raises_value_error": raises_value_error,
        "True": True,
        "False": False,
        "None": None,
    }
    return bool(eval(check, env, {}))


class UnsafeCodeError(RuntimeError):
    pass


class SafetyVisitor(ast.NodeVisitor):
    def visit_Import(self, node: ast.Import) -> Any:
        raise UnsafeCodeError("Imports are not allowed in generated code.")

    def visit_ImportFrom(self, node: ast.ImportFrom) -> Any:
        raise UnsafeCodeError("Imports are not allowed in generated code.")

    def visit_Call(self, node: ast.Call) -> Any:
        if isinstance(node.func, ast.Name) and node.func.id in RISKY_NAMES:
            raise UnsafeCodeError(f"Disallowed call: {node.func.id}")
        self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute) -> Any:
        root = node
        while isinstance(root, ast.Attribute):
            root = root.value
        if isinstance(root, ast.Name) and root.id in RISKY_ROOTS:
            raise UnsafeCodeError(f"Disallowed module or root object: {root.id}")
        self.generic_visit(node)


def load_function(code: str, function_name: str):
    code = strip_code_fences(code)
    tree = ast.parse(code, mode="exec")
    SafetyVisitor().visit(tree)
    has_target = any(isinstance(node, ast.FunctionDef) and node.name == function_name for node in tree.body)
    if not has_target:
        raise UnsafeCodeError(f"Generated code does not define `{function_name}`.")
    globals_dict = {"__builtins__": SAFE_BUILTINS}
    locals_dict: dict[str, Any] = {}
    exec(compile(tree, "<generated_code>", "exec"), globals_dict, locals_dict)
    fn = locals_dict.get(function_name) or globals_dict.get(function_name)
    if not callable(fn):
        raise UnsafeCodeError(f"`{function_name}` is not callable after execution.")
    return fn, code


def evaluate_candidate(task: dict[str, Any], code: str) -> dict[str, Any]:
    try:
        fn, clean_code = load_function(code, task["function_name"])
    except Exception as exc:
        return {
            "status": "invalid",
            "clean_code": strip_code_fences(code),
            "passed": 0,
            "total": len(task["hidden_checks"]),
            "pass_rate": 0.0,
            "error": f"{type(exc).__name__}: {exc}",
            "check_results": [],
        }

    check_results = []
    passed = 0
    for check in task["hidden_checks"]:
        try:
            ok = safe_eval_check(check, fn)
        except Exception as exc:
            ok = False
            check_results.append({"check": check, "passed": False, "error": f"{type(exc).__name__}: {exc}"})
        else:
            check_results.append({"check": check, "passed": ok, "error": ""})
        if ok:
            passed += 1

    total = len(task["hidden_checks"])
    return {
        "status": "ok",
        "clean_code": clean_code,
        "passed": passed,
        "total": total,
        "pass_rate": passed / total if total else 0.0,
        "error": "",
        "check_results": check_results,
    }


class GeminiTextClient:
    def __init__(self, api_keys: list[str], model: str = GENERATION_MODEL):
        self.api_keys = api_keys
        self.model = model
        self.session = requests.Session()

    def generate_text(self, prompt: str, temperature: float = 0.8) -> str:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent"
        body = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {
                "temperature": temperature,
                "topP": 0.95,
            },
        }
        for attempt in range(MAX_GENERATION_RETRIES):
            key = self.api_keys[attempt % len(self.api_keys)]
            response = self.session.post(
                url,
                headers={"Content-Type": "application/json", "x-goog-api-key": key},
                json=body,
                timeout=90,
            )
            if response.ok:
                data = response.json()
                candidates = data.get("candidates", [])
                if not candidates:
                    raise RuntimeError(f"No candidates returned: {data}")
                parts = candidates[0].get("content", {}).get("parts", [])
                text = "".join(part.get("text", "") for part in parts if isinstance(part, dict)).strip()
                if not text:
                    raise RuntimeError("Empty generateContent response.")
                return text
            if response.status_code in {429, 500, 502, 503, 504} and attempt < MAX_GENERATION_RETRIES - 1:
                delay = quota_retry_delay(response, attempt, default_cap=90) if response.status_code == 429 else min(30, 2 ** attempt)
                print(f"Retrying generation after HTTP {response.status_code}; sleeping {delay:.1f}s", flush=True)
                time.sleep(delay)
                continue
            raise RuntimeError(f"Gemini generateContent failed: HTTP {response.status_code}: {response.text[:400]}")
        raise RuntimeError("Gemini generateContent exhausted retries.")


def build_generation_prompt(task: dict[str, Any], style_prompt: str) -> str:
    examples = "\n".join(f"- {example}" for example in task.get("visible_examples", []))
    return (
        "Write Python code only. Do not use markdown fences. Do not include explanation.\n"
        "Do not import anything. Do not use files, networking, subprocesses, or external libraries.\n"
        f"{style_prompt}\n"
        f"Task:\n{task['prompt']}\n"
        f"Examples:\n{examples}\n"
        f"Required function name: {task['function_name']}\n"
        "Return only the function implementation."
    )


def build_critique_prompt(task: dict[str, Any], code: str) -> str:
    return (
        "Review the following Python function for likely strengths and likely failure modes.\n"
        "Do not choose a winner or compare it to another answer.\n"
        "Describe:\n"
        "- specification match;\n"
        "- likely edge cases handled or missed;\n"
        "- likely bugs or ambiguity;\n"
        "- readability / simplicity tradeoffs;\n"
        "- overall likelihood that the function really satisfies the task.\n"
        "Avoid labels like winner, better, worse, chosen, or rejected.\n\n"
        f"Task:\n{task['prompt']}\n\n"
        f"Code:\n{code}\n"
    )


def build_direct_text(task: dict[str, Any], code: str) -> str:
    return f"Task:\n{task['prompt']}\n\nCode:\n{code}"


def build_critique_text(task: dict[str, Any], critique: str) -> str:
    return f"Task:\n{task['prompt']}\n\nReview:\n{critique}"


def select_best_candidate(candidates: list[dict[str, Any]], method: str, rng: random.Random) -> dict[str, Any]:
    if method == "random":
        return rng.choice(candidates)
    if method == "length":
        return max(candidates, key=lambda row: (row["length_words"], row["candidate_id"]))
    if method == "direct_broad":
        return max(candidates, key=lambda row: (row["scores"]["direct_broad"], row["candidate_id"]))
    if method == "direct_code":
        return max(candidates, key=lambda row: (row["scores"]["direct_code"], row["candidate_id"]))
    if method == "critique_broad":
        return max(candidates, key=lambda row: (row["scores"]["critique_broad"], row["candidate_id"]))
    if method == "critique_code":
        return max(candidates, key=lambda row: (row["scores"]["critique_code"], row["candidate_id"]))
    raise ValueError(f"Unknown method: {method}")


def summarize_method(rows: list[dict[str, Any]]) -> dict[str, float | int]:
    solved = sum(1 for row in rows if row["selected_passed"] == row["selected_total"])
    avg_pass_rate = statistics.mean(row["selected_pass_rate"] for row in rows) if rows else 0.0
    avg_regret = statistics.mean(row["best_pass_rate"] - row["selected_pass_rate"] for row in rows) if rows else 0.0
    top_hit_rate = statistics.mean(1.0 if row["selected_is_best_available"] else 0.0 for row in rows) if rows else 0.0
    return {
        "tasks": len(rows),
        "tasks_solved": solved,
        "solve_rate": solved / len(rows) if rows else 0.0,
        "avg_selected_pass_rate": avg_pass_rate,
        "avg_regret_vs_best_available": avg_regret,
        "best_candidate_hit_rate": top_hit_rate,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tasks",
        type=Path,
        default=ROOT / "notes" / "research_cycles" / "cycle_003_objective_code_reranking" / "code_tasks_v1.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "notes" / "research_cycles" / "cycle_003_objective_code_reranking" / "pilot_results",
    )
    parser.add_argument("--limit-tasks", type=int, default=6)
    parser.add_argument("--candidates-per-task", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260625)
    parser.add_argument("--embed-model", default=None)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-workers", type=int, default=2)
    parser.add_argument("--mode", choices=["generated", "curated"], default="generated")
    args = parser.parse_args()

    load_local_env()
    api_keys, key_source = get_api_keys()
    if not api_keys:
        raise SystemExit("No Gemini API key found in environment or .env.local.")

    payload = read_json(args.tasks)
    tasks = payload["tasks"][: args.limit_tasks]
    styles = STYLE_PROMPTS[: args.candidates_per_task]
    rng = random.Random(args.seed)
    use_critiques = args.mode == "generated"

    generator = GeminiTextClient(api_keys=api_keys, model=GENERATION_MODEL)
    embedder = GeminiEmbedder(
        api_key=api_keys[0],
        api_keys=api_keys,
        model=args.embed_model,
        batch_size=args.batch_size,
        max_workers=args.max_workers,
    )
    probe = embedder.probe_model()

    axis_vectors = {}
    for axis_name, anchors in CODE_AXES.items():
        texts = anchors["positive"] + anchors["negative"]
        embs = embedder.encode(texts, label=f"{axis_name} anchors", cache_path=args.output / "embedding_cache" / f"{axis_name}_anchors.npy")
        pos = normalize(embs[: len(anchors["positive"])].mean(axis=0))
        neg = normalize(embs[len(anchors["positive"]) :].mean(axis=0))
        axis_vectors[axis_name] = normalize(pos - neg)

    candidate_rows: list[dict[str, Any]] = []
    direct_texts: list[str] = []
    critique_texts: list[str] = []

    for task in tasks:
        if args.mode == "generated":
            print(f"Generating candidates for {task['id']}...", flush=True)
            task_candidates = []
            for candidate_index, style_prompt in enumerate(styles, start=1):
                generated = generator.generate_text(build_generation_prompt(task, style_prompt), temperature=0.8)
                task_candidates.append(
                    {
                        "id": f"C{candidate_index}",
                        "label": style_prompt,
                        "raw_generation": generated,
                    }
                )
        else:
            print(f"Loading curated candidates for {task['id']}...", flush=True)
            task_candidates = task.get("candidates", [])[: args.candidates_per_task]
            if not task_candidates:
                raise SystemExit(f"No curated candidates found for task {task['id']}")

        for task_candidate in task_candidates:
            raw_code = task_candidate.get("raw_generation") or task_candidate["code"]
            evaluation = evaluate_candidate(task, raw_code)
            critique = ""
            if use_critiques:
                critique = generator.generate_text(build_critique_prompt(task, evaluation["clean_code"]), temperature=0.2).strip()
            candidate_id = task_candidate["id"]
            candidate_row = {
                "task_id": task["id"],
                "candidate_id": candidate_id,
                "style_prompt": task_candidate.get("label", ""),
                "raw_generation": raw_code,
                "code": evaluation["clean_code"],
                "critique": critique,
                "evaluation": evaluation,
                "length_words": word_count(evaluation["clean_code"]),
            }
            direct_texts.append(build_direct_text(task, evaluation["clean_code"]))
            if use_critiques:
                critique_texts.append(build_critique_text(task, critique))
            candidate_rows.append(candidate_row)

    direct_embs = embedder.encode(
        direct_texts,
        label="direct code texts",
        cache_path=args.output / "embedding_cache" / "direct_texts.npy",
    )
    critique_embs = None
    if use_critiques:
        critique_embs = embedder.encode(
            critique_texts,
            label="critique texts",
            cache_path=args.output / "embedding_cache" / "critique_texts.npy",
        )

    for idx, row in enumerate(candidate_rows):
        row["scores"] = {
            "direct_broad": float(direct_embs[idx] @ axis_vectors["broad_good_bad"]),
            "direct_code": float(direct_embs[idx] @ axis_vectors["code_quality"]),
        }
        if use_critiques and critique_embs is not None:
            row["scores"]["critique_broad"] = float(critique_embs[idx] @ axis_vectors["broad_good_bad"])
            row["scores"]["critique_code"] = float(critique_embs[idx] @ axis_vectors["code_quality"])

    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in candidate_rows:
        grouped.setdefault(row["task_id"], []).append(row)

    methods = ["random", "length", "direct_broad", "direct_code"]
    if use_critiques:
        methods.extend(["critique_broad", "critique_code"])
    selection_rows: list[dict[str, Any]] = []
    summary_rows: dict[str, list[dict[str, Any]]] = {method: [] for method in methods}

    for task in tasks:
        task_candidates = grouped[task["id"]]
        best_pass_rate = max(row["evaluation"]["pass_rate"] for row in task_candidates)
        best_candidate_ids = sorted(
            row["candidate_id"]
            for row in task_candidates
            if row["evaluation"]["pass_rate"] == best_pass_rate
        )
        for method in methods:
            selected = select_best_candidate(task_candidates, method, rng)
            row = {
                "task_id": task["id"],
                "method": method,
                "selected_candidate_id": selected["candidate_id"],
                "selected_passed": selected["evaluation"]["passed"],
                "selected_total": selected["evaluation"]["total"],
                "selected_pass_rate": selected["evaluation"]["pass_rate"],
                "selected_is_best_available": selected["candidate_id"] in best_candidate_ids,
                "best_pass_rate": best_pass_rate,
                "best_candidate_ids": best_candidate_ids,
            }
            selection_rows.append(row)
            summary_rows[method].append(row)

    method_summary = {method: summarize_method(rows) for method, rows in summary_rows.items()}

    args.output.mkdir(parents=True, exist_ok=True)
    write_json(
        args.output / "config.json",
        {
            "dataset_name": payload.get("dataset_name"),
            "purpose": payload.get("purpose"),
            "mode": args.mode,
            "generation_model": GENERATION_MODEL,
            "embedding_probe": probe,
            "key_source": key_source,
            "limit_tasks": args.limit_tasks,
            "candidates_per_task": args.candidates_per_task,
            "seed": args.seed,
        },
    )
    write_json(args.output / "candidates.json", candidate_rows)
    write_json(args.output / "selections.json", selection_rows)
    write_json(args.output / "summary.json", method_summary)

    lines = [
        "# Objective Code Reranking Pilot",
        "",
        f"Generation model: `{GENERATION_MODEL}`",
        f"Embedding model: `{probe['model']}`",
        f"Tasks: {len(tasks)}",
        f"Candidates per task: {len(styles)}",
        "",
        "## Method Results",
        "",
        "| Method | Solve rate | Tasks solved | Avg selected pass rate | Best-candidate hit rate | Avg regret vs best |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for method in methods:
        row = method_summary[method]
        lines.append(
            f"| `{method}` | {row['solve_rate']:.1%} | {row['tasks_solved']} / {row['tasks']} | "
            f"{row['avg_selected_pass_rate']:.1%} | {row['best_candidate_hit_rate']:.1%} | "
            f"{row['avg_regret_vs_best_available']:.3f} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation Rule",
            "",
            "This pilot is objective at the final metric layer because hidden unit tests decide whether the selected candidate is actually good. It is still a small pilot and should be read as directional evidence, not a publishable final benchmark.",
        ]
    )
    (args.output / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
