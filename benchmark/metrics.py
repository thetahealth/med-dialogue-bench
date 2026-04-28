"""
Evaluation Metrics for Doctor Agent Benchmark
"""

import json
import re
from pathlib import Path
from typing import Dict, List, Any, Optional

from jinja2 import Environment, FileSystemLoader

from shared.clients import token_limit_param

# Load judge prompt templates
_template_dir = Path(__file__).parent
_jinja_env = Environment(loader=FileSystemLoader(str(_template_dir)))
_diagnosis_template = _jinja_env.get_template("judge_diagnosis_prompt.jinja")
_disclosure_template = _jinja_env.get_template("judge_disclosure_prompt.jinja")


def _normalize_diagnosis(text: str) -> str:
    """Normalize diagnosis text for comparison"""
    return text.strip().lower().rstrip("。.、")


def exact_match_accuracy(results: List[Dict[str, Any]]) -> float:
    """
    Calculate exact match accuracy for multiple-choice diagnosis.

    Compares predicted diagnosis against ground truth using
    normalized string comparison.

    Args:
        results: List of DoctorConversationResult dicts

    Returns:
        Accuracy score (0.0 - 1.0)
    """
    if not results:
        return 0.0

    correct = 0
    evaluated = 0

    for r in results:
        gt = r.get("ground_truth_diagnosis", "")
        predicted = r.get("diagnosis", "")

        if not gt:
            continue

        evaluated += 1
        if _normalize_diagnosis(predicted) == _normalize_diagnosis(gt):
            correct += 1

    return correct / evaluated if evaluated > 0 else 0.0



def _llm_judge_diagnosis_and_misled(
    ground_truth: str,
    predicted: str,
    conversation: List[Dict[str, str]],
    client: Any,
    model: str,
    key_information: List[str] = None,
) -> Dict[str, Any]:
    """Combined LLM judge: diagnosis accuracy + misled detection in one call.

    Step 1: Is the diagnosis correct?
    Step 2: If incorrect, was the doctor misled by patient's false/contradictory info?
        Judge infers purely from dialogue content, without knowing experiment config.

    Returns:
        {"correct": bool, "misled": bool | None}
        misled is None when correct=True (not applicable).
    """
    # Format dialogue
    role_labels = {"doctor": "Doctor", "patient": "Patient"}
    dialogue_text = "\n".join(
        f"{role_labels.get(msg['role'], msg['role'])}: {msg['content']}"
        for msg in conversation
    )

    prompt = _diagnosis_template.render(
        ground_truth=ground_truth,
        predicted=predicted,
        dialogue_text=dialogue_text,
        key_information=key_information,
    )

    # Gemini emits long chains-of-thought; do not cap max_tokens. Cap for other models.
    token_kwargs = {} if model.startswith("gemini") else token_limit_param(model, 1000)
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        **token_kwargs,
    )

    raw = (response.choices[0].message.content or "").strip()

    # Parse JSON response — take last non-nested {...} to skip any JSON in CoT reasoning
    json_matches = re.findall(r'\{[^{}]*\}', raw)
    if json_matches:
        try:
            data = json.loads(json_matches[-1])
            correct = bool(data.get("correct", False))
            misled = None if correct else bool(data.get("misled", False))
            return {"correct": correct, "misled": misled}
        except (json.JSONDecodeError, ValueError):
            pass

    # Fallback: try to parse YES/NO patterns
    correct = "YES" in raw.upper().split("\n")[0] if raw else False
    return {"correct": correct, "misled": None if correct else False}


def _llm_judge_info_disclosure_turns(
    key_information: List[str],
    conversation: List[Dict[str, str]],
    client: Any,
    model: str,
) -> List[int]:
    """Judge which patient turn each key_info item was first disclosed.

    Sends one LLM call per case. Patient messages are labeled P1, P2, ...

    Returns:
        List of ints, one per key_info item.
        Value = patient message index (1-based) where first disclosed, 0 = never.
    """
    patient_msgs = [
        msg["content"] for msg in conversation if msg.get("role") == "patient"
    ]
    if not patient_msgs:
        return [0] * len(key_information)

    n = len(key_information)
    example = ", ".join(["1"] * min(2, n) + ["0"] * max(0, n - 2))

    prompt = _disclosure_template.render(
        n=n,
        example=example,
        patient_messages=patient_msgs,
        key_facts=key_information,
    )

    # Gemini emits long chains-of-thought; do not cap max_tokens. Cap for other models.
    token_kwargs = {} if model.startswith("gemini") else token_limit_param(model, 1000)
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        **token_kwargs,
    )

    raw = (response.choices[0].message.content or "").strip()

    json_match = re.search(r'\[.*\]', raw, re.DOTALL)
    if json_match:
        try:
            result = json.loads(json_match.group(0))
            if isinstance(result, list) and len(result) == n:
                upper = len(patient_msgs)
                return [
                    int(v) if isinstance(v, (int, float)) and 0 <= v <= upper else 0
                    for v in result
                ]
        except (json.JSONDecodeError, ValueError):
            pass

    return [0] * n


def info_coverage_single(
    key_information: List[str],
    conversation: List[Dict[str, str]],
    rounds: int,
    judge_client: Optional[Any] = None,
    judge_model: Optional[str] = None,
) -> Dict[str, Any]:
    """Calculate info coverage metrics for a single conversation.

    Returns:
        Dict with covered_count, total_count, coverage, efficiency, covered_items,
        plus per-turn disclosure data:
        - disclosure_turns: which patient turn each item was first disclosed (0=never)
        - coverage_curve: cumulative coverage after each patient turn
        - optimal_stop_turn: patient turn where coverage first reached final value
        - early_stoppable: True if doctor could have stopped earlier
        - wasted_turns: number of patient turns after optimal stop
    """
    empty = {
        "covered_count": 0, "total_count": len(key_information) if key_information else 0,
        "coverage": 0.0, "efficiency": 0.0, "covered_items": [],
        "disclosure_turns": [], "coverage_curve": [],
        "optimal_stop_turn": 0, "early_stoppable": False, "wasted_turns": 0,
    }

    if not key_information:
        return empty

    if not judge_client or not judge_model:
        empty["skipped"] = True
        return empty

    disclosure_turns = _llm_judge_info_disclosure_turns(
        key_information, conversation, judge_client, judge_model
    )

    # Derive covered_items (backward compatible)
    covered = [t > 0 for t in disclosure_turns]
    covered_count = sum(covered)
    total = len(key_information)
    coverage = covered_count / total
    efficiency = coverage / rounds if rounds > 0 else 0.0

    # Coverage curve: cumulative coverage after each patient turn
    n_patient = sum(1 for msg in conversation if msg.get("role") == "patient")
    coverage_curve = []
    for t in range(1, n_patient + 1):
        count_at_t = sum(1 for dt in disclosure_turns if 0 < dt <= t)
        coverage_curve.append(round(count_at_t / total, 4))

    # Optimal stop turn: when coverage first reaches final value
    disclosed = [t for t in disclosure_turns if t > 0]
    optimal_stop = max(disclosed) if disclosed else 0
    early_stoppable = optimal_stop > 0 and optimal_stop < n_patient

    return {
        "covered_count": covered_count,
        "total_count": total,
        "coverage": round(coverage, 4),
        "efficiency": round(efficiency, 4),
        "covered_items": covered,
        "disclosure_turns": disclosure_turns,
        "coverage_curve": coverage_curve,
        "optimal_stop_turn": optimal_stop,
        "early_stoppable": early_stoppable,
        "wasted_turns": n_patient - optimal_stop if early_stoppable else 0,
    }


def calculate_doctor_metrics(
    results: List[Dict[str, Any]],
    judge_client: Optional[Any] = None,
    judge_model: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Calculate aggregate metrics for doctor agent benchmark.

    Args:
        results: List of DoctorConversationResult dicts
        judge_client: Optional OpenAI client for LLM-as-judge evaluation
        judge_model: Model name for LLM judge

    Returns:
        Dict with aggregate metrics
    """
    total = len(results)
    if total == 0:
        return {"error": "No results to evaluate"}

    total_rounds = 0
    doctor_terminated = 0
    parse_success_count = 0

    for r in results:
        total_rounds += r.get("rounds", 0)

        if r.get("terminated_by") == "doctor":
            doctor_terminated += 1

        if r.get("diagnosis", "").strip():
            parse_success_count += 1

    metrics = {
        "total_cases": total,
        "avg_rounds": round(total_rounds / total, 2),
        "doctor_termination_rate": round(doctor_terminated / total, 4),
        "diagnosis_parse_rate": round(parse_success_count / total, 4),
        "diagnosis_accuracy_exact": round(exact_match_accuracy(results), 4),
    }

    # Optional: LLM-as-judge evaluation
    if judge_client and judge_model:
        # Semantic diagnosis accuracy + misled rate (combined judge)
        correct_count = 0
        misled_count = 0
        evaluated = 0

        for r in results:
            gt = r.get("ground_truth_diagnosis", "")
            predicted = r.get("diagnosis", "")
            conv = r.get("dialogue", [])

            if not gt or not predicted or not conv:
                continue

            evaluated += 1

            result = _llm_judge_diagnosis_and_misled(
                gt, predicted, conv, judge_client, judge_model
            )
            if result["correct"]:
                correct_count += 1
            if result["misled"]:
                misled_count += 1

        if evaluated > 0:
            metrics["semantic_accuracy"] = round(correct_count / evaluated, 4)
            metrics["misled_rate"] = round(misled_count / evaluated, 4)

        # Info coverage (requires key_information in results)
        coverage_sum = 0.0
        efficiency_sum = 0.0
        coverage_count = 0
        early_stoppable_count = 0
        total_wasted = 0

        for r in results:
            key_info = r.get("key_information", [])
            conv = r.get("dialogue", [])
            rounds = r.get("rounds", 0)

            if not key_info or not conv:
                continue

            cov = info_coverage_single(
                key_info, conv, rounds, judge_client, judge_model
            )
            if not cov.get("skipped"):
                coverage_sum += cov["coverage"]
                efficiency_sum += cov["efficiency"]
                coverage_count += 1
                if cov["early_stoppable"]:
                    early_stoppable_count += 1
                    total_wasted += cov["wasted_turns"]

        if coverage_count > 0:
            metrics["avg_info_coverage"] = round(coverage_sum / coverage_count, 4)
            metrics["avg_info_efficiency"] = round(efficiency_sum / coverage_count, 4)
            metrics["early_stoppable_rate"] = round(early_stoppable_count / coverage_count, 4)
            metrics["avg_wasted_turns"] = round(total_wasted / coverage_count, 2)

    return metrics
