"""
Benchmark Runner - Run doctor agent benchmarks against adversarial patients
"""

import json
import time
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field, asdict

from patient_agent import PatientAgent, PatientProfile, MedicalCase
from .metrics import calculate_doctor_metrics


# --- Data Loading ---


def load_test_cases(file_path: str) -> List[Dict[str, Any]]:
    """
    Load test cases from Excel, CSV, or JSON file.

    Expected columns:
    - Case_ID: Unique case identifier
    - Profile: User profile (name,gender,age,occupation)
    - T_Location, T_Primary, T_Severity, T_Onset, T_Character,
      T_Radiation, T_Associations, T_Modifiers
    - APAF_Attitude, APAF_Logic, APAF_Cognition, APAF_Expression,
      APAF_Purpose, APAF_Disclosure (optional, for adversarial profiles)

    Args:
        file_path: Path to Excel (.xlsx), CSV, or JSON file

    Returns:
        List of test case dicts
    """
    path = Path(file_path)

    if path.suffix.lower() == ".xlsx":
        try:
            import pandas as pd
            df = pd.read_excel(file_path)
            return df.to_dict("records")
        except ImportError:
            raise ImportError(
                "pandas and openpyxl required for Excel files. "
                "Install with: pip install pandas openpyxl"
            )

    elif path.suffix.lower() == ".csv":
        try:
            import pandas as pd
            df = pd.read_csv(file_path)
            return df.to_dict("records")
        except ImportError:
            raise ImportError(
                "pandas required for CSV files. Install with: pip install pandas"
            )

    elif path.suffix.lower() == ".json":
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)

    else:
        raise ValueError(f"Unsupported file format: {path.suffix}")


# --- Doctor Agent Benchmark ---


@dataclass
class DoctorConversationResult:
    """Result of a single doctor agent benchmark conversation"""
    case_id: str
    ground_truth_diagnosis: str
    diagnosis_result: Optional[Dict[str, Any]]
    rounds: int
    terminated_by: str  # "doctor", "max_turns"
    adversarial_score: float = 0.0
    key_information: List[str] = field(default_factory=list)
    state_history: List[Dict[str, Any]] = field(default_factory=list)
    conversation: List[Dict[str, str]] = field(default_factory=list)
    error: Optional[str] = None


class DoctorBenchmarkRunner:
    """
    Run doctor agent benchmark (inquiry + diagnosis).

    Usage:
        runner = DoctorBenchmarkRunner(
            doctor_model="gpt-4o",
            patient_model="gpt-4o-mini",
        )
        results = runner.run("test_cases.json")
        metrics = runner.get_metrics()
    """

    def __init__(
        self,
        doctor_model: str = "gpt-4o",
        patient_model: str = "gpt-4o-mini",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        max_turns: int = 20,
        language: str = "zh",
    ):
        self.doctor_model = doctor_model
        self.patient_model = patient_model
        self.api_key = api_key
        self.base_url = base_url
        self.max_turns = max_turns
        self.language = language
        self.results: List[DoctorConversationResult] = []

    def run_single(self, test_case: Dict[str, Any]) -> DoctorConversationResult:
        """Run benchmark for a single test case."""
        from doctor_agent import DoctorAgent

        case_id = test_case.get("Case_ID", "unknown")

        # Create patient agent
        case = MedicalCase.from_test_case(test_case)
        profile = PatientProfile.from_test_case(test_case)
        patient = PatientAgent(
            case=case,
            profile=profile,
            model=self.patient_model,
            api_key=self.api_key,
            base_url=self.base_url,
            language=self.language,
        )

        # Create doctor agent
        doctor = DoctorAgent(
            model=self.doctor_model,
            api_key=self.api_key,
            base_url=self.base_url,
            language=self.language,
            max_turns=self.max_turns,
        )

        history: List[Dict[str, str]] = []

        try:
            # Initial patient complaint (LLM-generated)
            opening = patient.respond()
            history.append({"role": "patient", "content": opening.text})

            terminated_by = "max_turns"
            rounds = 0

            while rounds < self.max_turns:
                # Doctor turn
                result = doctor.inquiry_turn(history)
                rounds += 1

                if result.is_terminated:
                    if result.text:
                        history.append({"role": "doctor", "content": result.text})
                    terminated_by = "doctor"
                    break

                history.append({"role": "doctor", "content": result.text})

                # Patient turn
                patient_response = patient.respond(result.text)
                history.append({"role": "patient", "content": patient_response.text})

            # Diagnosis phase
            diagnosis = doctor.diagnose(history)
            diagnosis_dict = {
                "diagnosis": diagnosis.diagnosis,
                "differential": diagnosis.differential,
                "evidence": diagnosis.evidence,
                "confidence": diagnosis.confidence,
                "parse_success": diagnosis.parse_success,
            }

            return DoctorConversationResult(
                case_id=case_id,
                ground_truth_diagnosis=test_case.get("Ground_Truth_Diagnosis", ""),
                diagnosis_result=diagnosis_dict,
                rounds=rounds,
                terminated_by=terminated_by,
                adversarial_score=patient.calculate_adversarial_score(),
                key_information=case.key_information,
                state_history=patient.state_history,
                conversation=history,
            )

        except Exception as e:
            return DoctorConversationResult(
                case_id=case_id,
                ground_truth_diagnosis=test_case.get("Ground_Truth_Diagnosis", ""),
                diagnosis_result=None,
                rounds=0,
                terminated_by="error",
                key_information=case.key_information,
                conversation=history,
                error=str(e),
            )

    def run(
        self,
        test_cases_path: str,
        limit: Optional[int] = None,
        verbose: bool = True,
    ) -> List[DoctorConversationResult]:
        """Run benchmark on all test cases."""
        test_cases = load_test_cases(test_cases_path)

        if limit:
            test_cases = test_cases[:limit]

        self.results = []
        total = len(test_cases)

        for i, case in enumerate(test_cases):
            if verbose:
                case_id = case.get("Case_ID", i)
                print(f"[{i+1}/{total}] Running case {case_id}...", end=" ")

            start_time = time.time()
            result = self.run_single(case)
            elapsed = time.time() - start_time

            self.results.append(result)

            if verbose:
                diag = (result.diagnosis_result or {}).get("diagnosis", "N/A")
                print(f"{result.terminated_by}, {result.rounds} rounds, "
                      f"diagnosis={diag}, {elapsed:.1f}s")

        return self.results

    def get_metrics(self) -> Dict[str, Any]:
        """Calculate metrics from results"""
        return calculate_doctor_metrics([asdict(r) for r in self.results])

    def save_results(self, output_path: str) -> None:
        """Save results to JSON file."""
        output = {
            "config": {
                "doctor_model": self.doctor_model,
                "patient_model": self.patient_model,
                "max_turns": self.max_turns,
                "language": self.language,
            },
            "metrics": self.get_metrics(),
            "results": [asdict(r) for r in self.results],
        }

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(output, f, ensure_ascii=False, indent=2)

        print(f"Results saved to: {output_path}")


def main():
    """CLI entry point"""
    import argparse

    parser = argparse.ArgumentParser(description="Run medical dialogue benchmark")
    parser.add_argument("--input", "-i", required=True, help="Test cases file (JSON/Excel/CSV)")
    parser.add_argument("--output", "-o", default="benchmark_results.json", help="Output file")
    parser.add_argument("--model", "-m", default="gpt-4o", help="Doctor agent model")
    parser.add_argument("--patient-model", default="gpt-4o-mini", help="Patient agent model")
    parser.add_argument("--limit", "-l", type=int, help="Limit number of cases")
    parser.add_argument("--max-turns", type=int, default=20, help="Max conversation turns")
    args = parser.parse_args()

    runner = DoctorBenchmarkRunner(
        doctor_model=args.model,
        patient_model=args.patient_model,
        max_turns=args.max_turns,
    )

    runner.run(args.input, limit=args.limit)

    print("\n=== Metrics ===")
    metrics = runner.get_metrics()
    for k, v in metrics.items():
        print(f"  {k}: {v}")

    runner.save_results(args.output)


if __name__ == "__main__":
    main()
