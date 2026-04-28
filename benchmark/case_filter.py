#!/usr/bin/env python3
"""
Case filtering script: select diagnosable cases via empirical baseline validation.

Method:
  For each case, count correct diagnoses across M models × R rounds.
  Keep cases with >= threshold correct out of total attempts.

Usage:
  # 3-model filtering (original)
  python -m benchmark.case_filter \
    --cases data/cases.json \
    --models gpt-5.4 claude-opus-4-6 qwen3.5-plus \
    --results-dir results_non_dynamic \
    --threshold 2 \
    --output data/cases_filtered.json

  # 4-model filtering (with Gemini)
  python -m benchmark.case_filter \
    --cases data/cases.json \
    --models gpt-5.4 claude-opus-4-6 qwen3.5-plus gemini-3.1-pro \
    --results-dir results_non_dynamic \
    --threshold 2 \
    --output data/cases_filtered.json
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path


def load_cases(cases_path: str) -> list:
    with open(cases_path) as f:
        return json.load(f)


def collect_baseline_results(
    case_ids: list[str],
    models: list[str],
    results_dir: Path,
) -> dict[str, dict[str, list[bool]]]:
    """Collect baseline accuracy per case per model across R1 + R2.

    Returns:
        {case_id: {model: [bool, bool, ...]}}  — list of correct/incorrect per run
    """
    data = defaultdict(lambda: defaultdict(list))

    for model in models:
        for run in ["baseline", "baseline_r2"]:
            # baseline_r2 may have an extra "baseline" subdirectory
            run_dir = results_dir / model / run
            if not run_dir.exists():
                continue
            # Check for nested baseline subdirectory (experiment.py artifact)
            nested = run_dir / "baseline"
            if nested.exists() and nested.is_dir():
                run_dir = nested
            for cid in case_ids:
                eval_path = run_dir / f"eval_{cid}.json"
                if not eval_path.exists():
                    continue
                with open(eval_path) as f:
                    ev = json.load(f)
                correct = bool(ev.get("semantic_accuracy"))
                data[cid][model].append(correct)

    return data


def filter_cases(
    cases: list,
    results: dict,
    models: list[str],
    threshold: int,
) -> tuple[list, list, list]:
    """Filter cases by threshold.

    Returns:
        (kept, excluded, report_rows)
    """
    kept = []
    excluded = []
    report_rows = []

    for case in cases:
        cid = case["Case_ID"]
        total_correct = 0
        total_attempts = 0
        model_details = {}

        for model in models:
            runs = results.get(cid, {}).get(model, [])
            correct_count = sum(runs)
            model_details[model] = f"{correct_count}/{len(runs)}"
            total_correct += correct_count
            total_attempts += len(runs)

        passed = total_correct >= threshold
        if passed:
            kept.append(case)
        else:
            excluded.append(case)

        report_rows.append({
            "case_id": cid,
            "total_correct": total_correct,
            "total_attempts": total_attempts,
            "passed": passed,
            **model_details,
        })

    return kept, excluded, report_rows


def print_report(
    report_rows: list,
    models: list[str],
    threshold: int,
    kept_count: int,
    total_count: int,
):
    print(f"\n{'='*80}")
    print(f"Case Filtering Report")
    print(f"{'='*80}")
    print(f"Models: {', '.join(models)}")
    print(f"Threshold: >= {threshold} correct")
    print(f"Result: {kept_count}/{total_count} cases kept\n")

    # Header
    model_headers = "  ".join(f"{m[:12]:>12}" for m in models)
    print(f"{'Case':>6}  {model_headers}  {'Total':>7}  {'Status':>8}")
    print("-" * (6 + 2 + 14 * len(models) + 2 + 7 + 2 + 8))

    for row in sorted(report_rows, key=lambda r: (-r["total_correct"], r["case_id"])):
        model_vals = "  ".join(f"{row[m]:>12}" for m in models)
        status = "✅ KEEP" if row["passed"] else "❌ EXCL"
        total_str = f"{row['total_correct']}/{row['total_attempts']}"
        print(f"{row['case_id']:>6}  {model_vals}  {total_str:>7}  {status:>8}")

    # Summary by score
    print(f"\n--- Distribution ---")
    from collections import Counter
    score_dist = Counter(r["total_correct"] for r in report_rows)
    max_attempts = max(r["total_attempts"] for r in report_rows)
    for score in range(max_attempts, -1, -1):
        count = score_dist.get(score, 0)
        if count > 0:
            bar = "█" * count
            status = "KEEP" if score >= threshold else "EXCL"
            print(f"  {score}/{max_attempts}: {count:>3} cases {bar} [{status}]")


def main():
    parser = argparse.ArgumentParser(description="Filter cases by baseline diagnosability")
    parser.add_argument("--cases", required=True, help="Source cases JSON (e.g. data/cases.json)")
    parser.add_argument("--models", nargs="+", required=True,
                        help="Model directory names (e.g. gpt-5.4 claude-opus-4-6)")
    parser.add_argument("--results-dir", default="results_non_dynamic",
                        help="Results root directory")
    parser.add_argument("--threshold", type=int, default=2,
                        help="Minimum correct count to keep (default: 2)")
    parser.add_argument("--output", help="Output filtered cases JSON")
    parser.add_argument("--report", help="Output detailed report as markdown")
    args = parser.parse_args()

    cases = load_cases(args.cases)
    case_ids = [c["Case_ID"] for c in cases]
    results_dir = Path(args.results_dir)

    print(f"Loaded {len(cases)} cases from {args.cases}")
    print(f"Models: {args.models}")
    print(f"Results dir: {results_dir}")

    # Check available data
    for model in args.models:
        for run in ["baseline", "baseline_r2"]:
            run_dir = results_dir / model / run
            if run_dir.exists():
                nested = run_dir / "baseline"
                search_dir = nested if nested.exists() and nested.is_dir() else run_dir
                count = len(list(search_dir.glob("eval_*.json")))
                print(f"  {model}/{run}: {count} eval files")
            else:
                print(f"  {model}/{run}: (not found)")

    results = collect_baseline_results(case_ids, args.models, results_dir)
    kept, excluded, report_rows = filter_cases(cases, results, args.models, args.threshold)

    print_report(report_rows, args.models, args.threshold, len(kept), len(cases))

    if args.output:
        with open(args.output, "w") as f:
            json.dump(kept, f, indent=2, ensure_ascii=False)
        print(f"\n✅ Saved {len(kept)} filtered cases to {args.output}")

    if args.report:
        with open(args.report, "w") as f:
            f.write(f"# Case Filtering Report\n\n")
            f.write(f"**Date:** Generated by benchmark/case_filter.py\n")
            f.write(f"**Source:** {args.cases} ({len(cases)} cases)\n")
            f.write(f"**Models:** {', '.join(args.models)}\n")
            f.write(f"**Threshold:** >= {args.threshold} correct\n")
            f.write(f"**Result:** {len(kept)}/{len(cases)} cases kept\n\n")

            f.write(f"## Per-case Results\n\n")
            headers = "| Case | " + " | ".join(args.models) + " | Total | Status |\n"
            sep = "|:---:|" + ":---:|" * len(args.models) + ":---:|:---:|\n"
            f.write(headers)
            f.write(sep)
            for row in sorted(report_rows, key=lambda r: r["case_id"]):
                model_vals = " | ".join(row[m] for m in args.models)
                total_str = f"{row['total_correct']}/{row['total_attempts']}"
                status = "✅" if row["passed"] else "❌"
                f.write(f"| {row['case_id']} | {model_vals} | {total_str} | {status} |\n")

            f.write(f"\n## Kept Cases ({len(kept)})\n\n")
            f.write(", ".join(c["Case_ID"] for c in kept))
            f.write(f"\n\n## Excluded Cases ({len(excluded)})\n\n")
            f.write(", ".join(c["Case_ID"] for c in excluded))
            f.write("\n")

        print(f"✅ Saved report to {args.report}")


if __name__ == "__main__":
    main()
