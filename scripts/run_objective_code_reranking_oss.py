#!/usr/bin/env python3
"""Objective code reranking pilot with open-source embedding models."""

from __future__ import annotations

import argparse
import ast
import json
import random
import re
import statistics
from pathlib import Path
from typing import Any

import numpy as np
from sentence_transformers import SentenceTransformer


ROOT = Path(__file__).resolve().parents[1]


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


DEFAULT_MODELS = [
    "sentence-transformers/all-mpnet-base-v2",
    "BAAI/bge-base-en-v1.5",
]


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")


def normalize(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float32)
    norm = np.linalg.norm(v)
    if norm == 0:
        return v
    return v / norm


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


def build_direct_text(task: dict[str, Any], code: str) -> str:
    return f"Task:\n{task['prompt']}\n\nCode:\n{code}"


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


def encode_texts(model: SentenceTransformer, texts: list[str]) -> np.ndarray:
    return model.encode(
        texts,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    )


def run_model(model_name: str, payload: dict[str, Any], seed: int) -> dict[str, Any]:
    print(f"\n=== MODEL: {model_name} ===", flush=True)
    model = SentenceTransformer(model_name)
    axis_vectors = {}
    for axis_name, anchors in CODE_AXES.items():
        texts = anchors["positive"] + anchors["negative"]
        embs = encode_texts(model, texts)
        pos_count = len(anchors["positive"])
        pos = normalize(embs[:pos_count].mean(axis=0))
        neg = normalize(embs[pos_count:].mean(axis=0))
        axis_vectors[axis_name] = normalize(pos - neg)

    candidate_rows: list[dict[str, Any]] = []
    direct_texts: list[str] = []
    tasks = payload["tasks"]

    for task in tasks:
        for task_candidate in task["candidates"]:
            evaluation = evaluate_candidate(task, task_candidate["code"])
            candidate_rows.append(
                {
                    "task_id": task["id"],
                    "candidate_id": task_candidate["id"],
                    "label": task_candidate["label"],
                    "code": evaluation["clean_code"],
                    "evaluation": evaluation,
                    "length_words": word_count(evaluation["clean_code"]),
                }
            )
            direct_texts.append(build_direct_text(task, evaluation["clean_code"]))

    direct_embs = encode_texts(model, direct_texts)
    for idx, row in enumerate(candidate_rows):
        row["scores"] = {
            "direct_broad": float(direct_embs[idx] @ axis_vectors["broad_good_bad"]),
            "direct_code": float(direct_embs[idx] @ axis_vectors["code_quality"]),
        }

    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in candidate_rows:
        grouped.setdefault(row["task_id"], []).append(row)

    rng = random.Random(seed)
    methods = ["random", "length", "direct_broad", "direct_code"]
    selection_rows: list[dict[str, Any]] = []
    summary_rows: dict[str, list[dict[str, Any]]] = {method: [] for method in methods}

    for task in tasks:
        task_candidates = grouped[task["id"]]
        best_pass_rate = max(row["evaluation"]["pass_rate"] for row in task_candidates)
        best_candidate_ids = sorted(
            row["candidate_id"] for row in task_candidates if row["evaluation"]["pass_rate"] == best_pass_rate
        )
        for method in methods:
            if method == "random":
                selected = rng.choice(task_candidates)
            elif method == "length":
                selected = max(task_candidates, key=lambda row: (row["length_words"], row["candidate_id"]))
            elif method == "direct_broad":
                selected = max(task_candidates, key=lambda row: (row["scores"]["direct_broad"], row["candidate_id"]))
            else:
                selected = max(task_candidates, key=lambda row: (row["scores"]["direct_code"], row["candidate_id"]))

            result_row = {
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
            selection_rows.append(result_row)
            summary_rows[method].append(result_row)

    method_summary = {method: summarize_method(rows) for method, rows in summary_rows.items()}
    print(json.dumps(method_summary, indent=2), flush=True)
    return {
        "model": model_name,
        "summary": method_summary,
        "selections": selection_rows,
        "candidates": candidate_rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tasks",
        type=Path,
        default=ROOT / "notes" / "research_cycles" / "cycle_003_objective_code_reranking" / "code_candidates_v1.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "notes" / "research_cycles" / "cycle_003_objective_code_reranking" / "oss_results",
    )
    parser.add_argument("--seed", type=int, default=20260625)
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    args = parser.parse_args()

    payload = read_json(args.tasks)
    results = [run_model(model_name, payload, args.seed) for model_name in args.models]

    args.output.mkdir(parents=True, exist_ok=True)
    write_json(args.output / "results.json", results)
    summary = {row["model"]: row["summary"] for row in results}
    write_json(args.output / "summary.json", summary)
    print(f"\nSaved results to {args.output}", flush=True)


if __name__ == "__main__":
    main()
