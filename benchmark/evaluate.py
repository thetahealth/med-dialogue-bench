"""
Evaluate simulation results using LLM-as-judge.

Usage:
    python -m benchmark.evaluate dial_001.json dial_004.json
    python -m benchmark.evaluate dial_001.json --judge-model qwen3.5-plus
    python -m benchmark.evaluate dial_001.json --repeat 3 --output results.json
"""

import argparse
import json
import sys
from pathlib import Path

from shared.clients import create_client, DEFAULT_CLIENT, CLIENT_NAMES
from benchmark.metrics import (
    _llm_judge_diagnosis_and_misled,
    _llm_judge_info_disclosure_turns,
    info_coverage_single,
)


def evaluate_single(case: dict, client, model: str) -> dict:
    """Evaluate a single case, return per-case metrics."""
    gt = case.get("ground_truth_diagnosis", "")
    predicted = case.get("diagnosis", "")
    dialogue = case.get("dialogue", [])
    key_info = case.get("key_information", [])
    rounds = case.get("rounds", 0)

    result = {"case_id": case.get("case_id", "?"), "doctor_model": case.get("doctor_model", "?")}

    # Judge call 1: diagnosis accuracy + misled
    if gt and predicted and dialogue:
        judge1 = _llm_judge_diagnosis_and_misled(gt, predicted, dialogue, client, model, key_info)
        result["semantic_accuracy"] = judge1["correct"]
        result["misled"] = judge1["misled"]
    else:
        result["semantic_accuracy"] = None
        result["misled"] = None
        result["skip_reason_judge1"] = "missing gt/predicted/dialogue"

    # Judge call 2: info disclosure → coverage + efficiency
    if key_info and dialogue:
        cov = info_coverage_single(key_info, dialogue, rounds, client, model)
        result["coverage"] = cov["coverage"]
        result["efficiency"] = cov["efficiency"]
        result["disclosure_turns"] = cov["disclosure_turns"]
        result["covered_count"] = cov["covered_count"]
        result["total_count"] = cov["total_count"]
        result["optimal_stop_turn"] = cov["optimal_stop_turn"]
        result["wasted_turns"] = cov["wasted_turns"]
    else:
        result["coverage"] = None
        result["efficiency"] = None
        result["skip_reason_judge2"] = "missing key_info/dialogue"

    # Hardcoded metrics
    result["rounds"] = rounds
    result["terminated_by"] = case.get("terminated_by", "")

    return result


def print_result(r: dict):
    """Pretty-print one case's evaluation result."""
    print(f"\n{'='*60}")
    print(f"Case: {r['case_id']}  Doctor: {r['doctor_model']}")
    print(f"{'='*60}")

    # Judge 1
    if r["semantic_accuracy"] is not None:
        acc = "CORRECT" if r["semantic_accuracy"] else "INCORRECT"
        misled = r["misled"]
        misled_str = "N/A" if misled is None else ("YES" if misled else "NO")
        print(f"  Diagnosis:  {acc}  |  Misled: {misled_str}")
    else:
        print(f"  Diagnosis:  SKIPPED ({r.get('skip_reason_judge1', '')})")

    # Judge 2
    if r["coverage"] is not None:
        print(f"  Coverage:   {r['coverage']:.4f} ({r['covered_count']}/{r['total_count']})")
        print(f"  Efficiency: {r['efficiency']:.4f}")
        print(f"  Disclosure: {r['disclosure_turns']}")
        print(f"  Optimal stop: turn {r['optimal_stop_turn']}  |  Wasted: {r['wasted_turns']}")
    else:
        print(f"  Coverage:   SKIPPED ({r.get('skip_reason_judge2', '')})")

    # Hardcoded
    print(f"  Rounds:     {r['rounds']}  |  Terminated by: {r['terminated_by']}")


def print_summary(results: list):
    """Print aggregate summary for multiple results."""
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    acc_results = [r for r in results if r["semantic_accuracy"] is not None]
    if acc_results:
        correct = sum(1 for r in acc_results if r["semantic_accuracy"])
        print(f"  Accuracy: {correct}/{len(acc_results)}")
    cov_results = [r for r in results if r["coverage"] is not None]
    if cov_results:
        avg_cov = sum(r["coverage"] for r in cov_results) / len(cov_results)
        avg_eff = sum(r["efficiency"] for r in cov_results) / len(cov_results)
        print(f"  Avg Coverage:   {avg_cov:.4f}")
        print(f"  Avg Efficiency: {avg_eff:.4f}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate simulation results")
    parser.add_argument("files", nargs="+", help="Simulation JSON files to evaluate")
    parser.add_argument("--client", default=DEFAULT_CLIENT, choices=CLIENT_NAMES)
    parser.add_argument("--judge-model", default=None, help="Judge model (default: client default)")
    parser.add_argument("--output", "-o", default=None, help="Write results to JSON file")
    parser.add_argument("--repeat", "-n", type=int, default=1, help="Repeat evaluation N times (default: 1)")
    args = parser.parse_args()

    client, model = create_client(args.client, args.judge_model)
    print(f"Judge: {model} via {args.client}")

    # Resolve file paths once
    paths = []
    for filepath in args.files:
        path = Path(filepath)
        if not path.exists():
            print(f"WARNING: {filepath} not found, skipping", file=sys.stderr)
            continue
        paths.append(path)

    all_runs = []
    for run_idx in range(args.repeat):
        if args.repeat > 1:
            print(f"\n{'#'*60}")
            print(f"RUN {run_idx + 1}/{args.repeat}")
            print(f"{'#'*60}")

        results = []
        for path in paths:
            case = json.loads(path.read_text())
            print(f"\nEvaluating {path.name} ...")
            r = evaluate_single(case, client, model)
            r["file"] = path.name
            results.append(r)
            print_result(r)

        if len(results) > 1:
            print_summary(results)

        all_runs.append(results)

    # Write output
    if args.output:
        output_data = {
            "judge_model": model,
            "client": args.client,
            "repeat": args.repeat,
            "runs": all_runs,
        }
        out_path = Path(args.output)
        out_path.write_text(json.dumps(output_data, indent=2, ensure_ascii=False))
        print(f"\nResults written to {out_path}")


if __name__ == "__main__":
    main()
