"""
Experiment Orchestrator — run APAF experiments end-to-end.

Pipeline: script generation → simulation → evaluation → aggregation.
Supports parallel execution and checkpoint-based resume.

Usage:
    python -m benchmark.experiment --list
    python -m benchmark.experiment --cases data/cases.json --configs baseline --count 10
    python -m benchmark.experiment --cases data/cases.json --group single --concurrency 10
"""

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from case_generator.experiment_configs import (
    DEFAULTS,
    EXPERIMENT_CONFIGS,
    config_label,
)
from case_generator.script_generator import ScriptGenerator, config_is_all_baseline
from shared.clients import CLIENT_NAMES, DEFAULT_CLIENT, create_client


# ---------------------------------------------------------------------------
# Config registry: map config name → config dict
# ---------------------------------------------------------------------------

def _build_config_registry() -> Dict[str, Dict[str, str]]:
    """Build name→config lookup from EXPERIMENT_CONFIGS."""
    registry = {}
    for config in EXPERIMENT_CONFIGS:
        name = config_label(config)
        registry[name] = config
    return registry


CONFIG_REGISTRY = _build_config_registry()


def _config_group(config: Dict[str, str]) -> str:
    """Determine group of a config: baseline, single, or multi."""
    non_default = sum(1 for d, l in config.items() if l != DEFAULTS.get(d))
    if non_default == 0:
        return "baseline"
    elif non_default == 1:
        return "single"
    else:
        return "multi"


def resolve_configs(
    names: Optional[List[str]] = None,
    group: Optional[str] = None,
) -> List[Tuple[str, Dict[str, str]]]:
    """Resolve CLI arguments to a list of (config_name, config_dict) pairs.

    Args:
        names: Explicit config names (e.g. ["baseline", "disclosure=withholding"])
        group: Group filter ("baseline", "single", "multi", "all")

    Returns:
        List of (name, config_dict) tuples
    """
    if names:
        result = []
        for name in names:
            if name not in CONFIG_REGISTRY:
                available = ", ".join(sorted(CONFIG_REGISTRY.keys()))
                raise ValueError(f"Unknown config: '{name}'. Available: {available}")
            result.append((name, CONFIG_REGISTRY[name]))
        return result

    if group:
        if group == "all":
            return list(CONFIG_REGISTRY.items())
        return [
            (name, cfg) for name, cfg in CONFIG_REGISTRY.items()
            if _config_group(cfg) == group
        ]

    raise ValueError("Specify --configs or --group")


# ---------------------------------------------------------------------------
# Phase 1: Script generation and caching
# ---------------------------------------------------------------------------

SCRIPTS_DIR = Path("data/scripts")


def _merge_scripts_into_case(
    case: Dict[str, Any],
    scripts_result: Dict[str, Any],
    config: Dict[str, str],
) -> Dict[str, Any]:
    """Create a case copy with scripts embedded (simulate.py compatible format).

    Format: case["scripts"][dim] = {"type": level, "instruction": "...", ...}
    """
    case_copy = dict(case)
    scripts = {}
    for dim, level in config.items():
        if dim in scripts_result:
            entry = dict(scripts_result[dim])
            entry["type"] = level
            scripts[dim] = entry
        else:
            # Baseline dim or missing from result: store type only
            scripts[dim] = {"type": level}
    case_copy["scripts"] = scripts
    return case_copy


def generate_or_load_scripts(
    case: Dict[str, Any],
    config: Dict[str, str],
    config_name: str,
    generator: Optional[ScriptGenerator],
    scripts_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """Load cached scripts or generate new ones.

    Returns case dict with embedded scripts (ready for simulate.py).
    Saves to {scripts_dir}/{config_name}/case_{case_id}.json.
    """
    base_dir = scripts_dir or SCRIPTS_DIR
    case_id = case.get("Case_ID", "unknown")
    script_path = base_dir / config_name / f"case_{case_id}.json"

    if script_path.exists():
        with open(script_path, "r", encoding="utf-8") as f:
            return json.load(f)

    if generator is None:
        raise RuntimeError(
            f"Scripts missing for case {case_id}/{config_name} "
            "and no script generator configured"
        )
    result = generator.generate_scripts(case, config)
    case_with_scripts = _merge_scripts_into_case(case, result, config)

    # Cache
    script_path.parent.mkdir(parents=True, exist_ok=True)
    with open(script_path, "w", encoding="utf-8") as f:
        json.dump(case_with_scripts, f, ensure_ascii=False, indent=2)

    return case_with_scripts


# ---------------------------------------------------------------------------
# Phase 2: Dialogue simulation
# ---------------------------------------------------------------------------

def run_dialogue(
    case_with_scripts: Dict[str, Any],
    client_name: str,
    doctor_model: str,
    patient_model: str,
    language: str = "en",
    max_turns: int = 20,
    prompt_style: str = "minimal",
    dynamic_state: bool = False,
    generic_scripts: bool = False,
) -> Dict[str, Any]:
    """Run a doctor-patient dialogue. Extracted from simulate.py.

    Args:
        case_with_scripts: Case dict with embedded scripts (from generate_or_load_scripts)
        client_name: API client name (e.g. "nebula")
        doctor_model: Model name for doctor
        patient_model: Model name for patient
        language: "zh" or "en"
        max_turns: Maximum dialogue turns
        prompt_style: "minimal" or "guided"
        dynamic_state: Enable dynamic patient state changes during dialogue

    Returns:
        Dict matching simulate.py output format (case_id, diagnosis, dialogue, etc.)
    """
    from doctor_agent.agent import DoctorAgent
    from patient_agent.agent import MedicalCase, PatientAgent, PatientProfile

    # Read APAF config from scripts
    scripts = case_with_scripts.get("scripts", {})
    config = dict(DEFAULTS)
    for dim in config:
        if dim in scripts and "type" in scripts[dim]:
            config[dim] = scripts[dim]["type"]

    case = MedicalCase.from_test_case(case_with_scripts)
    profile = PatientProfile(
        attitude=config["attitude"],
        logic=config["logic"],
        cognition=config["cognition"],
        expression=config["expression"],
        purpose=config["purpose"],
        disclosure=config["disclosure"],
    )

    patient = PatientAgent(
        case=case, profile=profile,
        client_name=client_name, model=patient_model,
        language=language,
        dynamic_state=dynamic_state,
        generic_scripts=generic_scripts,
    )
    doctor = DoctorAgent(
        client_name=client_name, model=doctor_model,
        language=language,
        max_turns=max_turns,
        prompt_style=prompt_style,
    )

    # --- Dialogue loop (from simulate.py) ---
    history = []
    dialogue_log = []

    # Patient opening
    opening = patient.respond()
    if opening.is_empty:
        return _build_output(
            case, config, doctor, patient, history, dialogue_log,
            rounds=0, terminated_by="empty_response", diagnosis_text="",
            language=language, prompt_style=prompt_style,
        )

    history.append({"role": "patient", "content": opening.text})
    dialogue_log.append({
        "role": "patient", "content": opening.text,
        "thought": opening.thought, "state": opening.state,
    })

    terminated_by = "max_turns"
    rounds = 0

    while rounds < max_turns:
        # Doctor turn
        result = doctor.inquiry_turn(history)
        rounds += 1

        if not result.text and not result.is_terminated:
            terminated_by = "empty_response"
            break

        if result.is_terminated:
            if result.text:
                history.append({"role": "doctor", "content": result.text})
                dialogue_log.append({"role": "doctor", "content": result.text})
            terminated_by = "doctor"
            break

        history.append({"role": "doctor", "content": result.text})
        dialogue_log.append({"role": "doctor", "content": result.text})

        # Patient turn
        patient_resp = patient.respond(result.text)
        if patient_resp.is_empty:
            terminated_by = "empty_response"
            break

        history.append({"role": "patient", "content": patient_resp.text})
        dialogue_log.append({
            "role": "patient", "content": patient_resp.text,
            "thought": patient_resp.thought, "state": patient_resp.state,
        })

    # --- Diagnosis ---
    diagnosis_text = ""
    if terminated_by != "empty_response":
        diagnosis = doctor.diagnose(history)
        diagnosis_text = diagnosis.diagnosis

    return _build_output(
        case, config, doctor, patient, history, dialogue_log,
        rounds, terminated_by, diagnosis_text, language, prompt_style,
    )


def _build_output(
    case, config, doctor, patient, history, dialogue_log,
    rounds, terminated_by, diagnosis_text, language, prompt_style,
) -> Dict[str, Any]:
    """Build output dict matching simulate.py format."""
    return {
        "case_id": case.case_id,
        "ground_truth_diagnosis": case.ground_truth_diagnosis,
        "key_information": case.key_information,
        "config": config,
        "doctor_model": doctor.model,
        "patient_model": patient.model,
        "prompt_style": prompt_style,
        "language": language,
        "rounds": rounds,
        "terminated_by": terminated_by,
        "diagnosis": diagnosis_text,
        "dialogue": dialogue_log,
        "state_history": patient.state_history,
    }


# ---------------------------------------------------------------------------
# Three-phase pipeline for a single (case, config)
# ---------------------------------------------------------------------------

def _make_generic_scripts(config: Dict[str, str]) -> Dict[str, Any]:
    """Create scripts with type only (no case-specific instruction) for ablation."""
    scripts = {}
    for dim, level in config.items():
        scripts[dim] = {"type": level}
    return scripts


def _save_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def _load_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def process_one(
    case: Dict[str, Any],
    config: Dict[str, str],
    config_name: str,
    output_dir: Path,
    *,
    script_generator: Optional[ScriptGenerator],
    client_name: str,
    doctor_model: str,
    patient_model: str,
    judge_client: Any,
    judge_model: str,
    language: str,
    max_turns: int,
    prompt_style: str,
    phases: str,
    dynamic_state: bool = False,
    scripts_dir: Optional[Path] = None,
    generic_scripts: bool = False,
) -> Dict[str, Any]:
    """Run the three-phase pipeline for one (case, config) pair.

    Args:
        phases: "all", "script", "simulate", or "evaluate"
        generic_scripts: Skip case-specific script generation, use type-only scripts (for ablation)

    Returns:
        Status dict with case_id, config_name, phases completed, errors.
    """
    from benchmark.evaluate import evaluate_single

    case_id = case.get("Case_ID", "unknown")
    base_scripts_dir = scripts_dir or SCRIPTS_DIR
    status = {"case_id": case_id, "config": config_name, "phases": []}

    try:
        # Phase 1: Script generation
        if generic_scripts:
            case_with_scripts = dict(case)
            case_with_scripts["scripts"] = _make_generic_scripts(config)
            status["phases"].append("script")
        elif phases in ("all", "script", "simulate"):
            case_with_scripts = generate_or_load_scripts(
                case, config, config_name, script_generator,
                scripts_dir=base_scripts_dir,
            )
            status["phases"].append("script")
        else:
            # evaluate-only: need scripts for eval data
            script_path = base_scripts_dir / config_name / f"case_{case_id}.json"
            if script_path.exists():
                case_with_scripts = _load_json(script_path)
            else:
                status["error"] = "scripts not found"
                return status

        if phases == "script":
            return status

        # Phase 2: Simulation
        dial_path = output_dir / config_name / f"dial_{case_id}.json"
        if phases in ("all", "simulate"):
            if not dial_path.exists():
                result = run_dialogue(
                    case_with_scripts,
                    client_name, doctor_model,
                    patient_model,
                    language, max_turns, prompt_style,
                    dynamic_state=dynamic_state,
                    generic_scripts=generic_scripts,
                )
                _save_json(dial_path, result)
            status["phases"].append("simulate")

        if phases == "simulate":
            return status

        # Phase 3: Evaluation
        eval_path = output_dir / config_name / f"eval_{case_id}.json"
        if phases in ("all", "evaluate"):
            if not eval_path.exists():
                if not dial_path.exists():
                    status["error"] = "dialogue not found for evaluation"
                    return status
                dial_data = _load_json(dial_path)
                eval_result = evaluate_single(dial_data, judge_client, judge_model)
                _save_json(eval_path, eval_result)
            status["phases"].append("evaluate")

    except Exception as e:
        status["error"] = str(e)

    return status


# ---------------------------------------------------------------------------
# Phase 4: Aggregation
# ---------------------------------------------------------------------------

def aggregate_results(
    output_dir: Path,
    configs: List[Tuple[str, Dict[str, str]]],
    doctor_model: str,
    patient_model: str,
    judge_model: str,
) -> Dict[str, Any]:
    """Read eval files and compute per-config aggregate metrics.

    Writes summary.json to output_dir.
    """
    summary = {
        "doctor_model": doctor_model,
        "patient_model": patient_model,
        "judge_model": judge_model,
        "configs": {},
    }

    for config_name, _ in configs:
        config_dir = output_dir / config_name
        if not config_dir.exists():
            continue

        eval_files = sorted(config_dir.glob("eval_*.json"))
        if not eval_files:
            continue

        evals = [_load_json(f) for f in eval_files]

        # Compute aggregate metrics
        n = len(evals)
        acc_vals = [e for e in evals if e.get("semantic_accuracy") is not None]
        correct = sum(1 for e in acc_vals if e["semantic_accuracy"])
        misled_vals = [e for e in evals if e.get("misled") is True]

        cov_vals = [e for e in evals if e.get("coverage") is not None]
        eff_vals = [e for e in evals if e.get("efficiency") is not None]

        rounds_vals = [e["rounds"] for e in evals if "rounds" in e]
        doc_term = sum(1 for e in evals if e.get("terminated_by") == "doctor")
        empty_resp = sum(1 for e in evals if e.get("terminated_by") == "empty_response")

        metrics = {"cases_completed": n}
        if empty_resp > 0:
            metrics["empty_response_count"] = empty_resp

        if acc_vals:
            metrics["semantic_accuracy"] = round(correct / len(acc_vals), 4)
            metrics["misled_rate"] = round(len(misled_vals) / len(acc_vals), 4)
        if cov_vals:
            metrics["avg_coverage"] = round(
                sum(e["coverage"] for e in cov_vals) / len(cov_vals), 4
            )
        if eff_vals:
            metrics["avg_efficiency"] = round(
                sum(e["efficiency"] for e in eff_vals) / len(eff_vals), 4
            )
        if rounds_vals:
            metrics["avg_rounds"] = round(sum(rounds_vals) / len(rounds_vals), 2)
        if n > 0:
            metrics["doctor_termination_rate"] = round(doc_term / n, 4)

        summary["configs"][config_name] = metrics

    # Write summary
    summary_path = output_dir / "summary.json"
    _save_json(summary_path, summary)
    return summary


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def run_experiment(
    cases: List[Dict[str, Any]],
    configs: List[Tuple[str, Dict[str, str]]],
    output_dir: Path,
    *,
    concurrency: int = 5,
    script_generator: Optional[ScriptGenerator] = None,
    client_name: str = DEFAULT_CLIENT,
    doctor_model: str = "qwen3.5-plus",
    patient_model: str = "claude-opus-4-5-20251101",
    judge_client: Any = None,
    judge_model: str = "qwen3-max",
    language: str = "en",
    max_turns: int = 20,
    prompt_style: str = "minimal",
    phases: str = "all",
    prompt_version: Optional[str] = None,
    dynamic_state: bool = False,
    generic_scripts: bool = False,
) -> Dict[str, Any]:
    """Run the experiment pipeline across all (case, config) pairs.

    Uses ThreadPoolExecutor for parallel execution.
    """
    scripts_dir = Path(f"data/scripts_{prompt_version}") if prompt_version else SCRIPTS_DIR
    total = len(cases) * len(configs)
    print(f"\n{'#'*60}")
    print(f"Experiment: {len(configs)} configs x {len(cases)} cases = {total} tasks")
    print(f"Phases: {phases} | Concurrency: {concurrency}")
    print(f"Client: {client_name}")
    print(f"Doctor: {doctor_model} | Patient: {patient_model} | Dynamic: {dynamic_state}")
    if phases in ("all", "evaluate"):
        print(f"Judge: {judge_model}")
    print(f"Output: {output_dir}")
    print(f"{'#'*60}\n")

    completed = 0
    errors = 0
    start_time = time.time()

    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = {}
        for config_name, config in configs:
            for case in cases:
                future = pool.submit(
                    process_one,
                    case, config, config_name, output_dir,
                    script_generator=script_generator,
                    client_name=client_name,
                    doctor_model=doctor_model,
                    patient_model=patient_model,
                    judge_client=judge_client,
                    judge_model=judge_model,
                    language=language,
                    max_turns=max_turns,
                    prompt_style=prompt_style,
                    phases=phases,
                    dynamic_state=dynamic_state,
                    scripts_dir=scripts_dir,
                    generic_scripts=generic_scripts,
                )
                futures[future] = (case.get("Case_ID", "?"), config_name)

        for future in as_completed(futures):
            case_id, cfg_name = futures[future]
            try:
                status = future.result()
                if "error" in status:
                    errors += 1
                    print(f"  ERROR {case_id}/{cfg_name}: {status['error']}")
                completed += 1
                elapsed = time.time() - start_time
                rate = completed / elapsed if elapsed > 0 else 0
                eta = (total - completed) / rate if rate > 0 else 0
                print(
                    f"\r  [{completed}/{total}] "
                    f"{errors} errors | "
                    f"{elapsed:.0f}s elapsed | "
                    f"ETA {eta:.0f}s",
                    end="", flush=True,
                )
            except Exception as e:
                errors += 1
                completed += 1
                print(f"\n  EXCEPTION {case_id}/{cfg_name}: {e}")

    elapsed = time.time() - start_time
    print(f"\n\nDone: {completed} tasks in {elapsed:.0f}s ({errors} errors)")

    # Aggregate
    if phases in ("all", "evaluate"):
        summary = aggregate_results(
            output_dir, configs, doctor_model, patient_model, judge_model,
        )
        print(f"\nSummary saved to {output_dir / 'summary.json'}")

        # Print summary table
        print(f"\n{'Config':<35s} {'N':>4s} {'Acc':>6s} {'Misled':>7s} "
              f"{'Cov':>6s} {'Eff':>6s} {'Rnd':>5s}")
        print("-" * 75)
        for cfg_name, metrics in summary.get("configs", {}).items():
            n = metrics.get("cases_completed", 0)
            acc = metrics.get("semantic_accuracy", -1)
            mis = metrics.get("misled_rate", -1)
            cov = metrics.get("avg_coverage", -1)
            eff = metrics.get("avg_efficiency", -1)
            rnd = metrics.get("avg_rounds", -1)
            print(
                f"{cfg_name:<35s} {n:>4d} "
                f"{acc:>6.2%} {mis:>7.2%} "
                f"{cov:>6.3f} {eff:>6.4f} {rnd:>5.1f}"
            )
        return summary

    return {}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Run APAF experiment pipeline: scripts → simulate → evaluate → aggregate"
    )
    parser.add_argument("--cases", type=str, help="Cases JSON file (e.g. data/cases.json)")
    parser.add_argument("--configs", nargs="+", type=str,
                        help='Config names (e.g. baseline "disclosure=withholding")')
    parser.add_argument("--group", type=str,
                        choices=["all", "baseline", "single", "multi"],
                        help="Run all configs in a group")
    parser.add_argument("--count", type=int, help="Only use first N cases")
    parser.add_argument("--phase", type=str, default="all",
                        choices=["all", "script", "simulate", "evaluate"],
                        help="Which phases to run (default: all)")
    parser.add_argument("--concurrency", type=int, default=5,
                        help="Parallel workers (default: 5)")
    parser.add_argument("--output-dir", type=str, default="results",
                        help="Results directory (default: results)")
    parser.add_argument("--client", type=str, default=DEFAULT_CLIENT,
                        choices=CLIENT_NAMES, help="API client for patient/judge/script agents")
    parser.add_argument("--doctor-model", type=str, default="qwen3.5-plus")
    parser.add_argument("--patient-model", type=str,
                        default="claude-opus-4-5-20251101")
    parser.add_argument("--judge-model", type=str, default="qwen3-max")
    parser.add_argument("--script-model", type=str, default="gpt-5.2")
    parser.add_argument("--language", choices=["zh", "en"], default="en")
    parser.add_argument("--max-turns", type=int, default=20)
    parser.add_argument("--prompt-style", choices=["minimal", "guided"],
                        default="minimal")
    parser.add_argument("--dynamic", action="store_true",
                        help="Enable dynamic patient state changes during dialogue")
    parser.add_argument("--generic-scripts", action="store_true",
                        help="Skip case-specific script generation, use generic prompts (for ablation)")
    parser.add_argument("--prompt-version", type=str, default=None,
                        help="Patient prompt template version suffix (e.g. 'v1' loads prompt_v1.jinja)")
    parser.add_argument("--list", action="store_true",
                        help="List all configs and exit")
    args = parser.parse_args()

    # List mode
    if args.list:
        print(f"Total configs: {len(CONFIG_REGISTRY)}\n")
        current_group = None
        for name, cfg in CONFIG_REGISTRY.items():
            group = _config_group(cfg)
            if group != current_group:
                current_group = group
                print(f"\n=== {group} ===")
            non_default = {d: l for d, l in cfg.items() if l != DEFAULTS.get(d)}
            desc = ", ".join(f"{d}={l}" for d, l in non_default.items()) or "all defaults"
            print(f"  {name:<40s} {desc}")
        return

    if not args.cases:
        parser.error("--cases is required (use --list to see configs)")

    # Load cases
    with open(args.cases, "r", encoding="utf-8") as f:
        cases = json.load(f)
    if args.count:
        cases = cases[:args.count]
    print(f"Loaded {len(cases)} cases from {args.cases}")

    # Resolve configs
    configs = resolve_configs(names=args.configs, group=args.group)
    print(f"Selected {len(configs)} configs")

    # Create clients
    output_dir = Path(args.output_dir)

    script_gen = None
    if args.phase in ("all", "script", "simulate") and not args.generic_scripts:
        script_gen = ScriptGenerator(
            client_name=args.client,
            model=args.script_model,
            language=args.language,
            prompt_version=args.prompt_version,
        )

    judge_client_obj = None
    if args.phase in ("all", "evaluate"):
        judge_client_obj, judge_model_resolved = create_client(
            args.client, args.judge_model,
        )
    else:
        judge_model_resolved = args.judge_model

    # Run
    run_experiment(
        cases=cases,
        configs=configs,
        output_dir=output_dir,
        concurrency=args.concurrency,
        script_generator=script_gen,
        client_name=args.client,
        doctor_model=args.doctor_model,
        patient_model=args.patient_model,
        judge_client=judge_client_obj,
        judge_model=judge_model_resolved,
        language=args.language,
        max_turns=args.max_turns,
        prompt_style=args.prompt_style,
        phases=args.phase,
        prompt_version=args.prompt_version,
        dynamic_state=args.dynamic,
        generic_scripts=args.generic_scripts,
    )


if __name__ == "__main__":
    main()
