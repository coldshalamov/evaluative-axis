#!/usr/bin/env python3
"""Build larger frozen objective reranking suites for v2 research cycles."""

from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")


def word_count(text: str) -> int:
    return len(re.findall(r"[A-Za-z_0-9]+", text))


def candidate_payload(
    text: str,
    final_answer: str,
    label: str,
) -> dict[str, str]:
    return {
        "text": text,
        "final_answer": final_answer,
        "label": label,
    }


def shuffle_candidates(
    rng: random.Random,
    candidates: list[dict[str, str]],
) -> list[dict[str, str]]:
    shuffled = list(candidates)
    rng.shuffle(shuffled)
    return [
        {
            "id": f"C{idx + 1}",
            "label": row["label"],
            "text": row["text"],
            "final_answer": row["final_answer"],
        }
        for idx, row in enumerate(shuffled)
    ]


def summarize_tasks(tasks: list[dict[str, Any]]) -> dict[str, Any]:
    within_item_gaps = []
    correct_position_counts = {"C1": 0, "C2": 0, "C3": 0}
    all_lengths = []
    for task in tasks:
        lengths = [word_count(candidate["text"]) for candidate in task["candidates"]]
        all_lengths.extend(lengths)
        within_item_gaps.append(max(lengths) - min(lengths))
        for candidate in task["candidates"]:
            if candidate["label"] == "correct":
                correct_position_counts[candidate["id"]] += 1
                break
    return {
        "task_count": len(tasks),
        "candidate_count": sum(len(task["candidates"]) for task in tasks),
        "mean_candidate_words": sum(all_lengths) / len(all_lengths) if all_lengths else 0.0,
        "max_within_item_gap": max(within_item_gaps) if within_item_gaps else 0,
        "mean_within_item_gap": sum(within_item_gaps) / len(within_item_gaps) if within_item_gaps else 0.0,
        "correct_position_counts": correct_position_counts,
    }


def build_math_tasks(rng: random.Random) -> list[dict[str, Any]]:
    tasks: list[dict[str, Any]] = []
    task_index = 1

    for idx in range(6):
        a = 12 + idx * 3
        b = 14 + idx * 4
        correct = a * b
        wrong_one = a * (b + 1)
        wrong_two = (a + 2) * b
        task_rng = random.Random(rng.randint(0, 10**9))
        candidates = shuffle_candidates(
            task_rng,
            [
                candidate_payload(
                    f"Multiply {a} by {b} carefully. Since {a} x {b} = {correct}, the answer is {correct}.",
                    str(correct),
                    "correct",
                ),
                candidate_payload(
                    f"Multiply {a} by {b} carefully. Since {a} x {b} = {wrong_one}, the answer is {wrong_one}.",
                    str(wrong_one),
                    "neighbor_product_error",
                ),
                candidate_payload(
                    f"Multiply {a} by {b} carefully. Since {a} x {b} = {wrong_two}, the answer is {wrong_two}.",
                    str(wrong_two),
                    "other_neighbor_product_error",
                ),
            ],
        )
        tasks.append(
            {
                "id": f"math_v2_{task_index:03d}",
                "domain": "math",
                "target_axis": "truthfulness",
                "prompt": f"What is {a} multiplied by {b}?",
                "grading": {"mode": "number", "target": correct},
                "candidates": candidates,
            }
        )
        task_index += 1

    for idx in range(6):
        x = 4 + idx
        coeff = 3 + (idx % 4)
        bias = 5 + idx * 2
        total = coeff * x + bias
        wrong_one = total - bias
        wrong_two = x + coeff
        task_rng = random.Random(rng.randint(0, 10**9))
        candidates = shuffle_candidates(
            task_rng,
            [
                candidate_payload(
                    f"Isolate x carefully. Since {coeff}x = {total - bias} and x = {x}, the answer is {x}.",
                    str(x),
                    "correct",
                ),
                candidate_payload(
                    f"Isolate x carefully. Since {coeff}x = {wrong_one} and x = {wrong_one}, the answer is {wrong_one}.",
                    str(wrong_one),
                    "forgot_division",
                ),
                candidate_payload(
                    f"Isolate x carefully. Since {coeff}x = {total + bias} and x = {wrong_two}, the answer is {wrong_two}.",
                    str(wrong_two),
                    "sign_and_division_error",
                ),
            ],
        )
        tasks.append(
            {
                "id": f"math_v2_{task_index:03d}",
                "domain": "math",
                "target_axis": "truthfulness",
                "prompt": f"Solve for x: {coeff}x + {bias} = {total}.",
                "grading": {"mode": "number", "target": x},
                "candidates": candidates,
            }
        )
        task_index += 1

    for idx in range(6):
        base = 120 + idx * 30
        percent = 10 + idx * 5
        correct = base * percent / 100
        wrong_one = correct / 10
        wrong_two = base / percent
        task_rng = random.Random(rng.randint(0, 10**9))
        candidates = shuffle_candidates(
            task_rng,
            [
                candidate_payload(
                    f"Convert the percent carefully. Since {percent}% of {base} is {correct:g}, the answer is {correct:g}.",
                    f"{correct:g}",
                    "correct",
                ),
                candidate_payload(
                    f"Convert the percent carefully. Since {percent}% of {base} is {wrong_one:g}, the answer is {wrong_one:g}.",
                    f"{wrong_one:g}",
                    "decimal_place_error",
                ),
                candidate_payload(
                    f"Convert the percent carefully. Since {percent}% of {base} is {wrong_two:g}, the answer is {wrong_two:g}.",
                    f"{wrong_two:g}",
                    "division_instead_of_percent",
                ),
            ],
        )
        tasks.append(
            {
                "id": f"math_v2_{task_index:03d}",
                "domain": "math",
                "target_axis": "truthfulness",
                "prompt": f"What is {percent}% of {base}?",
                "grading": {"mode": "number", "target": correct},
                "candidates": candidates,
            }
        )
        task_index += 1

    for idx in range(6):
        width = 3 + idx
        height = 6 + idx * 2
        correct = 2 * (width + height)
        wrong_one = width * height
        wrong_two = width + height
        task_rng = random.Random(rng.randint(0, 10**9))
        candidates = shuffle_candidates(
            task_rng,
            [
                candidate_payload(
                    f"Use the perimeter formula carefully. Since 2 x ({width} + {height}) = {correct}, the answer is {correct}.",
                    str(correct),
                    "correct",
                ),
                candidate_payload(
                    f"Use the perimeter formula carefully. Since {width} x {height} = {wrong_one}, the answer is {wrong_one}.",
                    str(wrong_one),
                    "area_instead",
                ),
                candidate_payload(
                    f"Use the perimeter formula carefully. Since {width} + {height} = {wrong_two}, the answer is {wrong_two}.",
                    str(wrong_two),
                    "forgot_factor_two",
                ),
            ],
        )
        tasks.append(
            {
                "id": f"math_v2_{task_index:03d}",
                "domain": "math",
                "target_axis": "truthfulness",
                "prompt": f"What is the perimeter of a rectangle with width {width} and height {height}?",
                "grading": {"mode": "number", "target": correct},
                "candidates": candidates,
            }
        )
        task_index += 1

    for idx in range(6):
        base = 2 + (idx % 2)
        exponent = 4 + idx
        subtract = 5 + idx
        power = base**exponent
        correct = power - subtract
        wrong_one = (base * exponent) - subtract
        wrong_two = power + subtract
        task_rng = random.Random(rng.randint(0, 10**9))
        candidates = shuffle_candidates(
            task_rng,
            [
                candidate_payload(
                    f"Evaluate the power carefully. Since {base}^{exponent} = {power} and {power} - {subtract} = {correct}, the answer is {correct}.",
                    str(correct),
                    "correct",
                ),
                candidate_payload(
                    f"Evaluate the power carefully. Since {base}^{exponent} = {base * exponent} and {base * exponent} - {subtract} = {wrong_one}, the answer is {wrong_one}.",
                    str(wrong_one),
                    "power_as_multiplication",
                ),
                candidate_payload(
                    f"Evaluate the power carefully. Since {base}^{exponent} = {power} and {power} + {subtract} = {wrong_two}, the answer is {wrong_two}.",
                    str(wrong_two),
                    "sign_error",
                ),
            ],
        )
        tasks.append(
            {
                "id": f"math_v2_{task_index:03d}",
                "domain": "math",
                "target_axis": "truthfulness",
                "prompt": f"What is {base} to the {exponent}th power minus {subtract}?",
                "grading": {"mode": "number", "target": correct},
                "candidates": candidates,
            }
        )
        task_index += 1

    for idx in range(6):
        numbers = [2 + idx, 4 + idx, 8 + idx, 10 + idx]
        total = sum(numbers)
        correct = total / len(numbers)
        wrong_one = total / 3
        wrong_two = sorted(numbers)[1]
        rendered_numbers = ", ".join(str(number) for number in numbers)
        task_rng = random.Random(rng.randint(0, 10**9))
        candidates = shuffle_candidates(
            task_rng,
            [
                candidate_payload(
                    f"Average the list carefully. Since the sum is {total} and {total} / 4 = {correct:g}, the answer is {correct:g}.",
                    f"{correct:g}",
                    "correct",
                ),
                candidate_payload(
                    f"Average the list carefully. Since the sum is {total} and {total} / 3 = {wrong_one:g}, the answer is {wrong_one:g}.",
                    f"{wrong_one:g}",
                    "divide_by_three",
                ),
                candidate_payload(
                    f"Average the list carefully. Since the middle value is {wrong_two:g}, the answer is {wrong_two:g}.",
                    f"{wrong_two:g}",
                    "median_confusion",
                ),
            ],
        )
        tasks.append(
            {
                "id": f"math_v2_{task_index:03d}",
                "domain": "math",
                "target_axis": "truthfulness",
                "prompt": f"What is the mean of {rendered_numbers}?",
                "grading": {"mode": "number", "target": correct},
                "candidates": candidates,
            }
        )
        task_index += 1

    for idx in range(6):
        numerator_one = 1 + (idx % 3)
        denominator_one = 2 + (idx % 4)
        numerator_two = 1 + ((idx + 1) % 4)
        denominator_two = denominator_one
        correct = (numerator_one + numerator_two) / denominator_one
        wrong_one = (numerator_one + numerator_two) / (denominator_one + denominator_two)
        wrong_two = correct - (1 / denominator_one)
        task_rng = random.Random(rng.randint(0, 10**9))
        candidates = shuffle_candidates(
            task_rng,
            [
                candidate_payload(
                    f"Add the fractions carefully. Since {numerator_one}/{denominator_one} + {numerator_two}/{denominator_two} = {correct:g}, the answer is {correct:g}.",
                    f"{correct:g}",
                    "correct",
                ),
                candidate_payload(
                    f"Add the fractions carefully. Since {numerator_one}/{denominator_one} + {numerator_two}/{denominator_two} = {wrong_one:g}, the answer is {wrong_one:g}.",
                    f"{wrong_one:g}",
                    "denominator_addition_error",
                ),
                candidate_payload(
                    f"Add the fractions carefully. Since {numerator_one}/{denominator_one} + {numerator_two}/{denominator_two} = {wrong_two:g}, the answer is {wrong_two:g}.",
                    f"{wrong_two:g}",
                    "dropped_fraction_piece",
                ),
            ],
        )
        tasks.append(
            {
                "id": f"math_v2_{task_index:03d}",
                "domain": "math",
                "target_axis": "truthfulness",
                "prompt": f"What is {numerator_one}/{denominator_one} plus {numerator_two}/{denominator_two}?",
                "grading": {"mode": "number", "target": correct, "tolerance": 1e-9},
                "candidates": candidates,
            }
        )
        task_index += 1

    for idx in range(6):
        first_minutes = 35 + idx * 5
        second_minutes = 20 + idx * 7
        correct = first_minutes + second_minutes
        wrong_one = abs(first_minutes - second_minutes)
        wrong_two = (first_minutes // 10) * 10 + (second_minutes // 10) * 10
        task_rng = random.Random(rng.randint(0, 10**9))
        candidates = shuffle_candidates(
            task_rng,
            [
                candidate_payload(
                    f"Add the durations carefully. Since {first_minutes} + {second_minutes} = {correct}, the answer is {correct}.",
                    str(correct),
                    "correct",
                ),
                candidate_payload(
                    f"Add the durations carefully. Since {first_minutes} + {second_minutes} = {wrong_one}, the answer is {wrong_one}.",
                    str(wrong_one),
                    "subtraction_error",
                ),
                candidate_payload(
                    f"Add the durations carefully. Since {first_minutes} + {second_minutes} = {wrong_two}, the answer is {wrong_two}.",
                    str(wrong_two),
                    "rounded_tens_error",
                ),
            ],
        )
        tasks.append(
            {
                "id": f"math_v2_{task_index:03d}",
                "domain": "math",
                "target_axis": "truthfulness",
                "prompt": f"If one run lasted {first_minutes} minutes and another lasted {second_minutes} minutes, how many minutes total is that?",
                "grading": {"mode": "number", "target": correct},
                "candidates": candidates,
            }
        )
        task_index += 1

    return tasks


def build_tool_tasks(rng: random.Random) -> list[dict[str, Any]]:
    tasks: list[dict[str, Any]] = []
    task_index = 1

    ports = [3000, 4173, 5000, 8080]
    for port in ports:
        wrong_one = 127
        wrong_two = 80
        task_rng = random.Random(rng.randint(0, 10**9))
        candidates = shuffle_candidates(
            task_rng,
            [
                candidate_payload(
                    f"Read the address literally. The service is listening on port {port}, so the answer is {port}.",
                    str(port),
                    "correct",
                ),
                candidate_payload(
                    f"Read the address literally. The service is listening on port {wrong_one}, so the answer is {wrong_one}.",
                    str(wrong_one),
                    "host_confusion",
                ),
                candidate_payload(
                    f"Read the address literally. The service is listening on port {wrong_two}, so the answer is {wrong_two}.",
                    str(wrong_two),
                    "default_guess",
                ),
            ],
        )
        tasks.append(
            {
                "id": f"tool_v2_{task_index:03d}",
                "domain": "tool_interpretation",
                "target_axis": "truthfulness",
                "prompt": f"Output: `Server listening on http://127.0.0.1:{port}`. What port is the app listening on?",
                "grading": {"mode": "number", "target": port},
                "candidates": candidates,
            }
        )
        task_index += 1

    error_ports = [3000, 5432, 8000, 9000]
    for port in error_ports:
        task_rng = random.Random(rng.randint(0, 10**9))
        candidates = shuffle_candidates(
            task_rng,
            [
                candidate_payload(
                    f"Read the error literally. Another process is already using port {port}, so that is the direct problem.",
                    f"port {port} already in use",
                    "correct",
                ),
                candidate_payload(
                    f"Read the error literally. The app lacks permission to bind the network, so that is the direct problem.",
                    "network permission denied",
                    "permission_confusion",
                ),
                candidate_payload(
                    f"Read the error literally. The server timed out while binding the socket, so that is the direct problem.",
                    "socket timeout",
                    "timeout_guess",
                ),
            ],
        )
        tasks.append(
            {
                "id": f"tool_v2_{task_index:03d}",
                "domain": "tool_interpretation",
                "target_axis": "truthfulness",
                "prompt": f"Error: `EADDRINUSE: address already in use :::{port}`. What is the most likely problem?",
                "grading": {
                    "mode": "string",
                    "target": f"port {port} already in use",
                    "aliases": [
                        f"{port} already in use",
                        f"another process is using port {port}",
                        "port in use",
                    ],
                },
                "candidates": candidates,
            }
        )
        task_index += 1

    failure_pairs = [(1, 19), (2, 31), (3, 27), (4, 44)]
    for failed, passed in failure_pairs:
        task_rng = random.Random(rng.randint(0, 10**9))
        candidates = shuffle_candidates(
            task_rng,
            [
                candidate_payload(
                    f"Read the test summary literally. With {failed} failed and {passed} passed, the suite did not pass overall.",
                    "no",
                    "correct",
                ),
                candidate_payload(
                    f"Read the test summary literally. With {failed} failed and {passed} passed, the suite passed overall.",
                    "yes",
                    "majority_vote_wrong",
                ),
                candidate_payload(
                    f"Read the test summary literally. With {failed} failed and {passed} passed, the suite is basically a pass.",
                    "yes",
                    "softened_failure",
                ),
            ],
        )
        tasks.append(
            {
                "id": f"tool_v2_{task_index:03d}",
                "domain": "tool_interpretation",
                "target_axis": "truthfulness",
                "prompt": f"Test output: `{failed} failed, {passed} passed`. Did the test suite pass overall?",
                "grading": {"mode": "bool", "target": False},
                "candidates": candidates,
            }
        )
        task_index += 1

    ahead_counts = [1, 2, 4, 7]
    for count in ahead_counts:
        task_rng = random.Random(rng.randint(0, 10**9))
        candidates = shuffle_candidates(
            task_rng,
            [
                candidate_payload(
                    f"Read the Git status literally. Being ahead by {count} commits means those commits are local and not pushed yet.",
                    "no",
                    "correct",
                ),
                candidate_payload(
                    f"Read the Git status literally. Being ahead by {count} commits means origin already has those commits.",
                    "yes",
                    "backwards",
                ),
                candidate_payload(
                    f"Read the Git status literally. Because origin is named in the message, those commits are probably already pushed.",
                    "yes",
                    "name_mention_confusion",
                ),
            ],
        )
        tasks.append(
            {
                "id": f"tool_v2_{task_index:03d}",
                "domain": "tool_interpretation",
                "target_axis": "truthfulness",
                "prompt": f"Git says: `Your branch is ahead of 'origin/main' by {count} commits.` Have those commits been pushed to origin yet?",
                "grading": {"mode": "bool", "target": False},
                "candidates": candidates,
            }
        )
        task_index += 1

    http_codes = [
        (401, "request not authenticated", ["not authenticated", "authentication missing", "invalid auth"]),
        (403, "request authenticated but forbidden", ["forbidden", "authenticated but not allowed", "permission denied"]),
        (404, "resource not found", ["not found", "missing resource", "unknown route"]),
        (500, "server error", ["internal server error", "backend error", "server failure"]),
    ]
    for code, target, aliases in http_codes:
        wrong_one = "request succeeded"
        wrong_two = "client disconnected"
        task_rng = random.Random(rng.randint(0, 10**9))
        candidates = shuffle_candidates(
            task_rng,
            [
                candidate_payload(
                    f"Read the HTTP status literally. Code {code} means {target}, so that is the direct issue.",
                    target,
                    "correct",
                ),
                candidate_payload(
                    f"Read the HTTP status literally. Code {code} means the request succeeded, so that is the direct issue.",
                    wrong_one,
                    "success_confusion",
                ),
                candidate_payload(
                    f"Read the HTTP status literally. Code {code} means the client disconnected, so that is the direct issue.",
                    wrong_two,
                    "disconnect_guess",
                ),
            ],
        )
        tasks.append(
            {
                "id": f"tool_v2_{task_index:03d}",
                "domain": "tool_interpretation",
                "target_axis": "truthfulness",
                "prompt": f"HTTP response: `{code}`. What is the direct root problem?",
                "grading": {"mode": "string", "target": target, "aliases": aliases},
                "candidates": candidates,
            }
        )
        task_index += 1

    dependency_ports = [
        (5432, "postgres", ["postgresql", "database", "postgres database"]),
        (6379, "redis", ["redis cache", "cache", "redis server"]),
        (27017, "mongodb", ["mongo", "mongo database", "database"]),
        (3306, "mysql", ["mysql database", "database", "mariadb"]),
    ]
    for port, target, aliases in dependency_ports:
        wrong_one = "web server"
        wrong_two = "dns service"
        task_rng = random.Random(rng.randint(0, 10**9))
        candidates = shuffle_candidates(
            task_rng,
            [
                candidate_payload(
                    f"Read the port literally. Port {port} usually points to {target}, so that dependency is unreachable.",
                    target,
                    "correct",
                ),
                candidate_payload(
                    f"Read the port literally. Port {port} usually points to a web server, so that dependency is unreachable.",
                    wrong_one,
                    "service_guess",
                ),
                candidate_payload(
                    f"Read the port literally. Port {port} usually points to DNS, so that dependency is unreachable.",
                    wrong_two,
                    "dns_guess",
                ),
            ],
        )
        tasks.append(
            {
                "id": f"tool_v2_{task_index:03d}",
                "domain": "tool_interpretation",
                "target_axis": "truthfulness",
                "prompt": f"Log: `connection refused 127.0.0.1:{port}`. Which dependency is most likely unreachable?",
                "grading": {"mode": "string", "target": target, "aliases": aliases},
                "candidates": candidates,
            }
        )
        task_index += 1

    exit_codes = [
        (137, "out of memory", ["oom", "killed by oom", "memory limit exceeded"]),
        (143, "terminated by signal", ["sigterm", "terminated", "killed by signal"]),
        (130, "interrupted by keyboard", ["ctrl c", "keyboard interrupt", "interrupted"]),
        (1, "general failure", ["error", "failed", "nonzero exit"]),
    ]
    for code, target, aliases in exit_codes:
        wrong_one = "syntax error"
        wrong_two = "success"
        task_rng = random.Random(rng.randint(0, 10**9))
        candidates = shuffle_candidates(
            task_rng,
            [
                candidate_payload(
                    f"Read the exit code literally. Code {code} most likely means {target}, so that is the cause.",
                    target,
                    "correct",
                ),
                candidate_payload(
                    f"Read the exit code literally. Code {code} most likely means a syntax error, so that is the cause.",
                    wrong_one,
                    "syntax_guess",
                ),
                candidate_payload(
                    f"Read the exit code literally. Code {code} most likely means success, so that is the cause.",
                    wrong_two,
                    "success_confusion",
                ),
            ],
        )
        tasks.append(
            {
                "id": f"tool_v2_{task_index:03d}",
                "domain": "tool_interpretation",
                "target_axis": "truthfulness",
                "prompt": f"Process exited with code `{code}`. What is the most likely cause?",
                "grading": {"mode": "string", "target": target, "aliases": aliases},
                "candidates": candidates,
            }
        )
        task_index += 1

    missing_scripts = ["build", "lint", "start", "test"]
    for script_name in missing_scripts:
        task_rng = random.Random(rng.randint(0, 10**9))
        candidates = shuffle_candidates(
            task_rng,
            [
                candidate_payload(
                    f"Read the npm error literally. The package configuration does not define a {script_name} script, so that is the root problem.",
                    f"{script_name} script missing",
                    "correct",
                ),
                candidate_payload(
                    f"Read the npm error literally. A dependency failed to install during {script_name}, so that is the root problem.",
                    "dependency install failure",
                    "dependency_guess",
                ),
                candidate_payload(
                    f"Read the npm error literally. The Node version is too old for {script_name}, so that is the root problem.",
                    "node version too old",
                    "version_guess",
                ),
            ],
        )
        tasks.append(
            {
                "id": f"tool_v2_{task_index:03d}",
                "domain": "tool_interpretation",
                "target_axis": "truthfulness",
                "prompt": f"npm says: `Missing script: \"{script_name}\"`. What is the direct root problem?",
                "grading": {
                    "mode": "string",
                    "target": f"{script_name} script missing",
                    "aliases": [
                        f"missing {script_name} script",
                        f"no {script_name} script defined",
                        f"package json has no {script_name} script",
                    ],
                },
                "candidates": candidates,
            }
        )
        task_index += 1

    return tasks


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "experiments" / "research_system_v1" / "benchmarks",
    )
    parser.add_argument("--seed", type=int, default=20260627)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    math_tasks = build_math_tasks(rng)
    tool_tasks = build_tool_tasks(rng)

    math_payload = {
        "dataset_name": "objective_math_reranking_v2",
        "purpose": "Larger frozen exact-answer math reranking benchmark with randomized candidate order and low length confounding.",
        "tasks": math_tasks,
    }
    tool_payload = {
        "dataset_name": "tool_interpretation_reranking_v2",
        "purpose": "Larger frozen exact-answer tool-interpretation reranking benchmark with randomized candidate order and low length confounding.",
        "tasks": tool_tasks,
    }

    summary = {
        "seed": args.seed,
        "objective_math_reranking_v2": summarize_tasks(math_tasks),
        "tool_interpretation_reranking_v2": summarize_tasks(tool_tasks),
    }

    write_json(args.output_dir / "objective_math_reranking_v2.json", math_payload)
    write_json(args.output_dir / "tool_interpretation_reranking_v2.json", tool_payload)
    write_json(args.output_dir / "objective_suite_v2_build_summary.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
