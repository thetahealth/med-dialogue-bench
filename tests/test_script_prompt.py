"""
Render the full script_prompt.jinja prompt for manual inspection.

Provide levels for the 6 APAF dimensions; the rendered prompt only includes
the design blocks for non-baseline dimensions.

Usage:
    # Single dimension: disclosure=withholding, everything else baseline
    python tests/test_script_prompt.py --disclosure withholding

    # Multi-dimensional
    python tests/test_script_prompt.py --disclosure withholding --attitude evasive

    # Specific case + Chinese template
    python tests/test_script_prompt.py --index 0 --lang zh --cognition web_self_diagnosis

    # Feed from raw JSONL (fake key_info)
    python tests/test_script_prompt.py --jsonl /path/to.jsonl --disclosure selective
"""

import argparse

from _common import (
    get_jinja_env, load_cases, load_cases_json, make_fake_key_info, pick_case,
    print_case_header, print_separator,
)


def render_scripts(jinja_env, case: dict, key_info: list[str], lang: str,
                   config: dict) -> str:
    template = jinja_env.get_template("script_prompt.jinja")
    return template.render(
        language=lang,
        key_information=key_info,
        diagnosis=case.get("Correct_Diagnosis", ""),
        social_history=case.get("Social_History", ""),
        **config,
    )


def main():
    parser = argparse.ArgumentParser(description="Render script_prompt for inspection")
    parser.add_argument("--data", type=str, default=None,
                        help="Path to cases.json (default: project root cases.json)")
    parser.add_argument("--jsonl", type=str, default=None,
                        help="Path to agentclinic_medqa.jsonl (uses fake key_info)")
    parser.add_argument("--index", type=int, default=None,
                        help="Case index (0-based). Random if not specified.")
    parser.add_argument("--lang", choices=["zh", "en"], default="en",
                        help="Language (default: en)")

    # 6 dimension arguments
    parser.add_argument("--disclosure", default="responsive")
    parser.add_argument("--cognition", default="accurate")
    parser.add_argument("--logic", default="consistent")
    parser.add_argument("--attitude", default="compliant")
    parser.add_argument("--purpose", default="symptom_relief")
    parser.add_argument("--expression", default="normal")

    args = parser.parse_args()

    config = {
        "disclosure": args.disclosure,
        "cognition": args.cognition,
        "logic": args.logic,
        "attitude": args.attitude,
        "purpose": args.purpose,
        "expression": args.expression,
    }

    # Non-baseline dims for display
    non_baselines = {
        "disclosure": ["responsive"],
        "cognition": ["accurate"],
        "logic": ["consistent"],
        "attitude": ["compliant"],
        "purpose": ["symptom_relief"],
        "expression": ["clear", "normal"],
    }
    active = [f"{d}={l}" for d, l in config.items() if l not in non_baselines[d]]

    if args.jsonl:
        cases = load_cases(args.jsonl)
        use_real_key_info = False
    else:
        cases = load_cases_json(args.data) if args.data else load_cases_json()
        use_real_key_info = True

    idx, case = pick_case(cases, args.index)
    key_info = case.get("key_information", []) if use_real_key_info else make_fake_key_info(case)
    if not key_info:
        print(f"WARNING: Case #{idx} has no key_information, falling back to fake key_info")
        key_info = make_fake_key_info(case)

    print_case_header(idx, case)
    print(f"key_information: {len(key_info)} items ({'real' if use_real_key_info else 'fake'})")
    print(f"Config: {', '.join(active) if active else 'all baseline (no scripts needed)'}")

    jinja_env = get_jinja_env()
    print_separator(f"lang={args.lang}, {', '.join(active) if active else 'baseline'}")
    print(render_scripts(jinja_env, case, key_info, args.lang, config))


if __name__ == "__main__":
    main()
