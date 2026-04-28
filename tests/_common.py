"""Shared utilities for prompt rendering tests.

Reuses data loading and flattening from case_generator to avoid duplication.
"""

import random
import sys
from pathlib import Path

from jinja2 import Environment, FileSystemLoader

# Allow imports from project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from case_generator.agentclinic_loader import load_jsonl, convert_case  # noqa: E402
from case_generator.script_generator import DIMENSION_LEVELS  # noqa: E402, F401

# === Constants ===

JSONL_PATH = "/Users/admin/workspace/AgentClinic/agentclinic_medqa.jsonl"
CASES_JSON_PATH = str(Path(__file__).resolve().parent.parent / "cases.json")

TEMPLATE_DIR = Path(__file__).resolve().parent.parent / "case_generator"


def load_cases(path: str = JSONL_PATH) -> list[dict]:
    """Load and flatten all cases from JSONL."""
    raw_records = load_jsonl(path)
    return [convert_case(r, i + 1) for i, r in enumerate(raw_records)]


def load_cases_json(path: str = CASES_JSON_PATH) -> list[dict]:
    """Load cases from processed cases.json (with key_information)."""
    import json
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def make_fake_key_info(case: dict) -> list[str]:
    """Construct a simulated key_information list from flat case fields.

    NOT LLM-generated — just a quick approximation for testing template rendering.
    """
    items = []
    if case.get("Demographics"):
        items.append(case["Demographics"])
    if case.get("Primary_Symptom"):
        items.append(f"Chief complaint: {case['Primary_Symptom']}")
    for s in case.get("Secondary_Symptoms", "").split(";"):
        s = s.strip()
        if s:
            items.append(s)
    if case.get("Past_Medical_History"):
        items.append(f"PMH: {case['Past_Medical_History']}")
    if case.get("Medications"):
        items.append(f"Medications: {case['Medications']}")
    if case.get("Social_History"):
        items.append(f"Social: {case['Social_History']}")
    if case.get("Review_of_Systems"):
        items.append(f"ROS: {case['Review_of_Systems']}")
    return items


def get_jinja_env() -> Environment:
    return Environment(loader=FileSystemLoader(str(TEMPLATE_DIR)))


def pick_case(cases: list[dict], index: int | None) -> tuple[int, dict]:
    """Select a case by index (or random)."""
    if index is not None:
        if index < 0 or index >= len(cases):
            raise IndexError(f"index {index} out of range [0, {len(cases) - 1}]")
        return index, cases[index]
    idx = random.randint(0, len(cases) - 1)
    return idx, cases[idx]


def print_separator(title: str):
    print(f"\n{'=' * 70}")
    print(f"  {title}")
    print(f"{'=' * 70}\n")


def print_case_header(idx: int, case: dict):
    print(f"Case #{idx} | Diagnosis: {case.get('Correct_Diagnosis', '')}")
    print(f"Demographics: {case.get('Demographics', '')}")
    print(f"Primary Symptom: {case.get('Primary_Symptom', '')}")
