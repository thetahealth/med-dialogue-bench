"""
Render the full key_info_prompt.jinja prompt for manual inspection.

Usage:
    python tests/test_key_info_prompt.py                  # random case
    python tests/test_key_info_prompt.py --index 5        # specific index
    python tests/test_key_info_prompt.py --lang zh        # Chinese template
"""

import argparse

from _common import (
    get_jinja_env, load_cases, pick_case,
    print_case_header, print_separator,
)


def render_key_info(jinja_env, case: dict, lang: str) -> str:
    template = jinja_env.get_template("key_info_prompt.jinja")
    return template.render(
        language=lang,
        demographics=case.get("Demographics", ""),
        primary_symptom=case.get("Primary_Symptom", ""),
        secondary_symptoms=case.get("Secondary_Symptoms", ""),
        history=case.get("History", ""),
        past_medical_history=case.get("Past_Medical_History", ""),
        medications=case.get("Medications", ""),
        social_history=case.get("Social_History", ""),
        family_history=case.get("Family_History", ""),
        review_of_systems=case.get("Review_of_Systems", ""),
        correct_diagnosis=case.get("Correct_Diagnosis", ""),
        objective_for_doctor=case.get("Objective_for_Doctor", ""),
    )


def main():
    parser = argparse.ArgumentParser(description="Render key_info_prompt with AgentClinic data")
    parser.add_argument("--data", type=str, default=None,
                        help="Path to agentclinic_medqa.jsonl")
    parser.add_argument("--index", type=int, default=None,
                        help="Case index (0-based). Random if not specified.")
    parser.add_argument("--lang", choices=["zh", "en"], default="en",
                        help="Language (default: en)")
    args = parser.parse_args()

    cases = load_cases(args.data) if args.data else load_cases()
    idx, case = pick_case(cases, args.index)

    print_case_header(idx, case)
    print_separator(f"key_info_prompt.jinja  [case={idx}, lang={args.lang}]")

    jinja_env = get_jinja_env()
    print(render_key_info(jinja_env, case, args.lang))


if __name__ == "__main__":
    main()
