"""
Load AgentClinic JSONL cases, extract key_information, save to cases.json.

End-to-end script with resume support and periodic saves.

Usage:
    # Load from JSONL, extract all 107 cases
    python -m case_generator.run_key_info \
        --input path/to/agentclinic_medqa.jsonl

    # Try the first 5 cases
    python -m case_generator.run_key_info \
        --input path/to/agentclinic_medqa.jsonl \
        --count 5

    # Resume (read existing cases.json and skip already-extracted)
    python -m case_generator.run_key_info \
        --input path/to/agentclinic_medqa.jsonl \
        --resume

    # Custom model and output path
    python -m case_generator.run_key_info \
        --input path/to/agentclinic_medqa.jsonl \
        --client nebula --model gpt-4o \
        --output data/cases.json
"""

import argparse
import json
import sys
from pathlib import Path

from .agentclinic_loader import AgentClinicLoader
from .script_generator import ScriptGenerator
from shared.clients import CLIENT_NAMES, DEFAULT_CLIENT


def save(cases: list[dict], output: str) -> None:
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", encoding="utf-8") as f:
        json.dump(cases, f, ensure_ascii=False, indent=2)


def main():
    parser = argparse.ArgumentParser(
        description="Load AgentClinic cases → extract key_information via LLM"
    )
    parser.add_argument("--input", type=str, required=True,
                        help="AgentClinic JSONL file")
    parser.add_argument("--output", type=str, default="cases.json",
                        help="Output JSON file (default: cases.json)")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from existing output file (skip already extracted)")
    parser.add_argument("--offset", type=int, default=0,
                        help="Skip first N cases (default: 0)")
    parser.add_argument("--count", type=int,
                        help="Only process N cases (after offset)")
    parser.add_argument("--save-every", type=int, default=1,
                        help="Save progress every N extractions (default: 1)")
    parser.add_argument("--client", choices=CLIENT_NAMES, default=DEFAULT_CLIENT,
                        help=f"LLM client (default: {DEFAULT_CLIENT})")
    parser.add_argument("--model", type=str,
                        help="Override model name")
    parser.add_argument("--language", choices=["zh", "en"], default="en",
                        help="Language for prompts (default: en)")
    args = parser.parse_args()

    # --- Step 1: Load cases ---
    if args.resume and Path(args.output).exists():
        # Resume mode: load from existing output
        with open(args.output, "r", encoding="utf-8") as f:
            cases = json.load(f)
        done = sum(1 for c in cases if c.get("key_information"))
        print(f"Resume: loaded {len(cases)} cases from {args.output} ({done} already extracted)")
    else:
        # Fresh load from JSONL
        loader = AgentClinicLoader()
        cases = loader.load(file_path=args.input)
        if not cases:
            print("No cases loaded.", file=sys.stderr)
            sys.exit(1)

    if args.offset:
        cases = cases[args.offset:]
    if args.count:
        cases = cases[:args.count]
    if args.offset or args.count:
        print(f"Selected {len(cases)} cases (offset={args.offset}, count={args.count})")

    # --- Step 2: Extract key_information ---
    gen = ScriptGenerator(
        client_name=args.client,
        model=args.model,
        language=args.language,
    )

    extracted = 0
    total = len(cases)
    print(f"\n=== Extracting key_information ({total} cases) ===")

    for i, case in enumerate(cases):
        if case.get("key_information"):
            print(f"  [{i + 1}/{total}] SKIP (already has key_info)")
            continue

        print(f"  [{i + 1}/{total}] Extracting...", end=" ", flush=True)
        key_info = gen.extract_key_info(case)
        case["key_information"] = key_info
        extracted += 1
        print(f"OK ({len(key_info)} items)")

        # Periodic save
        if extracted % args.save_every == 0:
            save(cases, args.output)
            print(f"  [checkpoint] Saved after {extracted} extractions")

    # --- Step 3: Final save ---
    save(cases, args.output)
    done = sum(1 for c in cases if c.get("key_information"))
    print(f"\nDone! {extracted} newly extracted, {done}/{total} total with key_info")
    print(f"Output: {args.output}")


if __name__ == "__main__":
    main()
