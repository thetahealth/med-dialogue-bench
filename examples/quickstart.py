"""End-to-end demo: run one case with one behavioral config against one doctor model.

Cost: approximately $0.01 - $0.05. Time: ~1 minute.

Usage:
    cp .env.example .env
    python examples/quickstart.py
"""

from __future__ import annotations

import json
import pathlib
import sys
from dataclasses import asdict

from dotenv import load_dotenv

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from benchmark.runner import DoctorBenchmarkRunner


def main() -> None:
    load_dotenv()

    cases = json.loads((ROOT / "data" / "cases_filtered_85.json").read_text())
    case = dict(cases[0])

    case["APAF_Logic"] = "fabricating"
    case["APAF_Cognition"] = "accurate"
    case["APAF_Expression"] = "normal"
    case["APAF_Disclosure"] = "responsive"
    case["APAF_Attitude"] = "compliant"
    case["APAF_Purpose"] = "symptom_relief"

    runner = DoctorBenchmarkRunner(
        doctor_model="gpt-4o",
        patient_model="gpt-4o-mini",
        max_turns=8,
        language="en",
    )
    result = runner.run_single(case)

    print(json.dumps(asdict(result), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
