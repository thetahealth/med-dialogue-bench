"""
AgentClinic Loader - Load OSCE patient cases from AgentClinic benchmark.

Loads agentclinic_medqa*.jsonl files and flattens to a simple dict per case,
keeping original AgentClinic field names.

Usage:
    python -m case_generator --input-dir /path/to/AgentClinic/
    python -m case_generator --input /path/to/agentclinic_medqa.jsonl
    python -m case_generator --input-dir /path/to/AgentClinic/ --excel
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional


def _flatten(value: Any) -> str:
    """Flatten dict/list to string for Excel compatibility."""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return "; ".join(str(x) for x in value if x)
    if isinstance(value, dict):
        parts = []
        for k, v in value.items():
            flat = _flatten(v)
            if flat:
                parts.append(f"{k}: {flat}")
        return "; ".join(parts)
    return str(value) if value else ""


def load_jsonl(file_path: str) -> List[Dict[str, Any]]:
    """Load a JSONL file."""
    data = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def convert_case(item: Dict[str, Any], case_id: int) -> Dict[str, Any]:
    """Flatten one OSCE case, keeping original field names."""
    osce = item.get("OSCE_Examination", item)
    pa = osce.get("Patient_Actor", {})
    symptoms = pa.get("Symptoms", {})

    return {
        "Case_ID": f"{case_id:03d}",
        "Demographics": pa.get("Demographics", ""),
        "History": pa.get("History", ""),
        "Primary_Symptom": symptoms.get("Primary_Symptom", ""),
        "Secondary_Symptoms": "; ".join(symptoms.get("Secondary_Symptoms", [])),
        "Past_Medical_History": _flatten(pa.get("Past_Medical_History", "")),
        "Medications": _flatten(
            pa.get("Medications", pa.get("Current_Medications", pa.get("Drug_History", "")))
        ),
        "Family_History": _flatten(pa.get("Family_History", "")),
        "Social_History": _flatten(pa.get("Social_History", "")),
        "Review_of_Systems": _flatten(pa.get("Review_of_Systems", "")),
        "Objective_for_Doctor": osce.get("Objective_for_Doctor", ""),
        "Correct_Diagnosis": osce.get("Correct_Diagnosis", ""),
    }


class AgentClinicLoader:
    """Load AgentClinic OSCE data (agentclinic_medqa*.jsonl)."""

    def load(
        self,
        file_path: Optional[str] = None,
        input_dir: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Load and convert OSCE cases.

        Args:
            file_path: Path to a single JSONL file
            input_dir: Directory containing agentclinic_medqa*.jsonl files

        Returns:
            List of flattened case dicts
        """
        files = []
        if file_path:
            files = [Path(file_path)]
        elif input_dir:
            d = Path(input_dir)
            files = sorted(d.glob("agentclinic_medqa*.jsonl"))

        if not files:
            print("No AgentClinic OSCE files found.", file=sys.stderr)
            return []

        cases = []
        case_id = 1

        for fp in files:
            raw_data = load_jsonl(str(fp))
            count = 0
            for item in raw_data:
                case = convert_case(item, case_id)
                if case["Correct_Diagnosis"]:
                    cases.append(case)
                    case_id += 1
                    count += 1
            print(f"  {fp.name}: {len(raw_data)} raw -> {count} loaded")

        print(f"Total: {len(cases)} cases")
        return cases


def main():
    parser = argparse.ArgumentParser(
        description="Load AgentClinic OSCE data for MedDialBench"
    )
    parser.add_argument("--input", type=str, help="Single JSONL file")
    parser.add_argument(
        "--input-dir", type=str, help="Directory with agentclinic_medqa*.jsonl"
    )
    parser.add_argument(
        "--output", type=str, default="cases.json", help="Output JSON file"
    )
    parser.add_argument(
        "--excel", action="store_true", help="Also export to Excel (.xlsx)"
    )
    args = parser.parse_args()

    if not args.input and not args.input_dir:
        print("Specify --input or --input-dir", file=sys.stderr)
        sys.exit(1)

    loader = AgentClinicLoader()
    cases = loader.load(file_path=args.input, input_dir=args.input_dir)

    if not cases:
        print("No cases loaded.", file=sys.stderr)
        sys.exit(1)

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(cases, f, ensure_ascii=False, indent=2)
    print(f"Saved {len(cases)} cases -> {args.output}")

    if args.excel:
        import pandas as pd

        xlsx_path = Path(args.output).with_suffix(".xlsx")
        columns = [
            "Case_ID", "Demographics", "Correct_Diagnosis",
            "Primary_Symptom", "Secondary_Symptoms", "History",
            "Past_Medical_History", "Medications", "Family_History",
            "Social_History", "Review_of_Systems", "Objective_for_Doctor",
        ]
        df = pd.DataFrame(cases)
        cols = [c for c in columns if c in df.columns]
        df = df[cols]

        with pd.ExcelWriter(str(xlsx_path), engine="openpyxl") as writer:
            df.to_excel(writer, index=False, sheet_name="Cases")
            ws = writer.sheets["Cases"]
            for i, col in enumerate(df.columns, 1):
                width = max(len(col), df[col].astype(str).str.len().max())
                ws.column_dimensions[
                    ws.cell(row=1, column=i).column_letter
                ].width = min(width + 2, 60)

        print(f"Saved {len(df)} cases -> {xlsx_path}")


if __name__ == "__main__":
    main()
