"""
Generate behavioral scripts for the 85 cases.

Two modes, driven by --batch / --custom:

    # Default: generate all 6 packed batches covering every dim+level pair
    # (1 case x 6 LLM calls covers all 31 unique pairs used across experiments)
    python -m case_generator.run_scripts --input cases.json --model gpt-4.1

    # Single batch (1-6)
    python -m case_generator.run_scripts --input cases.json --model gpt-4.1 --batch 3

    # Ad-hoc custom config (still a single LLM call per case, same merge schema)
    python -m case_generator.run_scripts --input cases.json --model gpt-4.1 \\
        --custom --disclosure withholding --cognition accurate \\
        --logic fabricating --attitude compliant \\
        --purpose symptom_relief --expression normal

Common options:
    --offset N / --count N        slice the cases list
    --save-every N                checkpoint frequency (default 5)
    --dry-run                     show the plan, skip LLM calls
    --language zh|en              prompt language (default en)

Output: rewrites --output (default: overwrite --input). Cases get a nested
`scripts[dim][level] = {...}` entry per generated level, so repeat runs are
idempotent and resumable.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

from .experiment_configs import (
    ALL_LEVELS,
    DEFAULTS,
    GENERATION_BATCHES,
    config_label,
    get_unique_dim_levels,
)
from .script_generator import ScriptGenerator
from shared.clients import CLIENT_NAMES, DEFAULT_CLIENT


def save(cases: List[Dict], output: str) -> None:
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", encoding="utf-8") as f:
        json.dump(cases, f, ensure_ascii=False, indent=2)


def case_needs_batch(case: Dict, batch: Dict[str, str]) -> bool:
    """True if any dim+level in this batch is missing from the case."""
    scripts = case.get("scripts", {})
    for dim, level in batch.items():
        if level not in scripts.get(dim, {}):
            return True
    return False


def merge_scripts(case: Dict, result: Dict[str, Any], batch: Dict[str, str]) -> None:
    """Merge generated scripts into case["scripts"][dim][level]."""
    scripts = case.setdefault("scripts", {})
    for dim, level in batch.items():
        if dim in result:
            dim_scripts = scripts.setdefault(dim, {})
            if level not in dim_scripts:
                dim_scripts[level] = result[dim]


def count_coverage(cases: List[Dict]) -> Dict[tuple, int]:
    """Count how many cases have each dim+level pair."""
    from collections import Counter
    coverage = Counter()
    for c in cases:
        for dim, levels in c.get("scripts", {}).items():
            for level in levels:
                coverage[(dim, level)] += 1
    return coverage


def _build_custom_batch(args: argparse.Namespace) -> Dict[str, str]:
    return {
        "disclosure": args.disclosure,
        "cognition": args.cognition,
        "logic": args.logic,
        "attitude": args.attitude,
        "purpose": args.purpose,
        "expression": args.expression,
    }


def _add_dim_args(parser: argparse.ArgumentParser) -> None:
    for dim, levels in ALL_LEVELS.items():
        parser.add_argument(
            f"--{dim}",
            choices=levels,
            default=DEFAULTS[dim],
            help=f"{dim} level (default: {DEFAULTS[dim]} = baseline). "
                 f"Only used with --custom.",
        )


def main():
    parser = argparse.ArgumentParser(
        description="Generate APAF behavioral scripts for cases (batched or custom)."
    )
    parser.add_argument("--input", type=str, required=True,
                        help="Input JSON file (must have key_information).")
    parser.add_argument("--output", type=str,
                        help="Output JSON file (default: overwrite input).")
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--count", type=int,
                        help="Process only N cases after offset.")
    parser.add_argument("--batch", type=int, choices=range(1, 7),
                        help="Run only batch N (1-6). Mutually exclusive with --custom.")
    parser.add_argument("--custom", action="store_true",
                        help="Run a single ad-hoc config built from --disclosure etc.")
    parser.add_argument("--save-every", type=int, default=5)
    parser.add_argument("--client", choices=CLIENT_NAMES, default=DEFAULT_CLIENT)
    parser.add_argument("--model", type=str)
    parser.add_argument("--language", choices=["zh", "en"], default="en")
    parser.add_argument("--dry-run", action="store_true")
    _add_dim_args(parser)

    args = parser.parse_args()

    if args.custom and args.batch is not None:
        parser.error("--custom and --batch are mutually exclusive")

    # Resolve the batch list
    if args.custom:
        batches = [_build_custom_batch(args)]
        batch_ids = ["custom"]
    elif args.batch is not None:
        batches = [GENERATION_BATCHES[args.batch - 1]]
        batch_ids = [args.batch]
    else:
        batches = GENERATION_BATCHES
        batch_ids = list(range(1, 7))

    # Load cases
    with open(args.input, "r", encoding="utf-8") as f:
        all_cases = json.load(f)
    print(f"Loaded {len(all_cases)} cases from {args.input}")

    output = args.output or args.input

    # Slice
    cases = all_cases
    if args.offset:
        cases = cases[args.offset:]
    if args.count:
        cases = cases[:args.count]
    if args.offset or args.count:
        print(f"Selected {len(cases)} cases (offset={args.offset}, count={args.count})")

    # Check key_information
    has_key_info = sum(1 for c in cases if c.get("key_information"))
    if has_key_info == 0:
        print("No cases have key_information. Run run_key_info.py first.", file=sys.stderr)
        sys.exit(1)
    print(f"{has_key_info}/{len(cases)} cases have key_information")

    # Plan summary
    print(f"\n{'='*60}")
    print(f"Plan: {len(batches)} batch(es) x {len(cases)} cases")
    print(f"{'='*60}")
    for bid, batch in zip(batch_ids, batches):
        label = config_label(batch)
        needs = sum(1 for c in cases if c.get("key_information") and case_needs_batch(c, batch))
        print(f"  Batch {bid}: {label}")
        print(f"           {needs}/{len(cases)} cases need generation")
    print()

    if args.dry_run:
        print("Coverage summary across all experiment dim+level pairs:")
        all_pairs = get_unique_dim_levels()
        coverage = count_coverage(cases)
        for dim, level in all_pairs:
            n = coverage.get((dim, level), 0)
            status = "OK" if n == len(cases) else f"{n}/{len(cases)}"
            print(f"  {dim}={level}: {status}")
        print(f"\nTotal unique dim+level pairs: {len(all_pairs)}")
        return

    # Generate
    gen = ScriptGenerator(
        client_name=args.client,
        model=args.model,
        language=args.language,
    )

    total_generated = 0

    for bid, batch in zip(batch_ids, batches):
        label = config_label(batch)
        print(f"\n{'-'*60}")
        print(f"Batch {bid}: {label}")
        print(f"{'-'*60}")

        batch_generated = 0
        for i, case in enumerate(cases):
            key_info = case.get("key_information", [])
            if not key_info:
                print(f"  [{i+1}/{len(cases)}] SKIP (no key_info)")
                continue

            if not case_needs_batch(case, batch):
                print(f"  [{i+1}/{len(cases)}] SKIP (already done)")
                continue

            print(f"  [{i+1}/{len(cases)}] generating...", end=" ", flush=True)
            result = gen.generate_scripts(case, batch)

            if result:
                merge_scripts(case, result, batch)
                batch_generated += 1
                total_generated += 1
                print("OK")
            else:
                print("FAILED")

            if batch_generated > 0 and batch_generated % args.save_every == 0:
                save(all_cases, output)
                print(f"  [checkpoint] Saved after {batch_generated} cases in batch {bid}")

        if batch_generated > 0:
            save(all_cases, output)
        print(f"  Batch {bid} done: {batch_generated} cases generated")

    # Final summary
    print(f"\n{'='*60}")
    print(f"All done. {total_generated} total generations across {len(batches)} batch(es).")
    print(f"Output: {output}")

    all_pairs = get_unique_dim_levels()
    coverage = count_coverage(cases)
    complete = sum(1 for p in all_pairs if coverage.get(p, 0) == len(cases))
    print(f"Coverage: {complete}/{len(all_pairs)} dim+level pairs fully covered")


if __name__ == "__main__":
    main()
