"""
Script Generator - Generate case-specific behavioral scripts for APAF dimensions.

Pipeline:
  Step 1: Extract key_information from each case (run_key_info.py)
  Step 2: For each experiment config, generate scripts via one LLM call per case

Input:  a case + 6 dimension levels (e.g. disclosure=proactive, cognition=accurate, ...)
Output: scripts only for non-baseline dimensions in that config

Baselines (no script needed):
  disclosure=responsive, cognition=accurate, logic=consistent,
  attitude=compliant, purpose=symptom_relief, expression=clear/normal
"""

import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from jinja2 import Environment, FileSystemLoader

from shared.clients import DEFAULT_CLIENT, create_client, token_limit_param

# Non-baseline levels that need LLM-generated scripts
DISCLOSURE_LEVELS = ["proactive", "selective", "reluctant", "withholding"]
COGNITION_LEVELS = [
    "partial_understanding", "misconception",
    "web_self_diagnosis", "folk_remedy", "complete_denial",
]
LOGIC_LEVELS = [
    "minor_inconsistency", "occasional_contradiction",
    "frequent_contradiction", "fabricating",
]
ATTITUDE_LEVELS = ["reluctant", "impatient", "evasive", "resistant", "dominant"]
PURPOSE_LEVELS = ["reassurance", "medication_request"]
EXPRESSION_LEVELS = ["vague", "verbose", "tangential", "incoherent"]

DIMENSION_LEVELS = {
    "disclosure": DISCLOSURE_LEVELS,
    "cognition": COGNITION_LEVELS,
    "logic": LOGIC_LEVELS,
    "attitude": ATTITUDE_LEVELS,
    "purpose": PURPOSE_LEVELS,
    "expression": EXPRESSION_LEVELS,
}

ALL_DIMENSIONS = list(DIMENSION_LEVELS.keys())
MAX_RETRIES = 3

BASELINES = {
    "disclosure": ["responsive"],
    "cognition": ["accurate"],
    "logic": ["consistent"],
    "attitude": ["compliant"],
    "purpose": ["symptom_relief"],
    "expression": ["clear", "normal"],
}


def is_baseline(dimension: str, level: str) -> bool:
    """Check if a dimension+level is a baseline (no script needed)."""
    return level in BASELINES.get(dimension, [])


def config_is_all_baseline(config: Dict[str, str]) -> bool:
    """Check if an experiment config is all baselines."""
    return all(is_baseline(dim, level) for dim, level in config.items())


class ScriptGenerator:
    """Generate case-specific behavioral scripts for APAF dimensions.

    Args:
        client_name: LLM client name
        model: Override model name
        language: 'zh' or 'en'
    """

    def __init__(
        self,
        client_name: str = DEFAULT_CLIENT,
        model: Optional[str] = None,
        language: str = "en",
        prompt_version: Optional[str] = None,
    ):
        self.client, self.model = create_client(client_name, model)
        self.language = language

        template_dir = Path(__file__).parent
        jinja_env = Environment(loader=FileSystemLoader(str(template_dir)))
        self.key_info_template = jinja_env.get_template("key_info_prompt.jinja")
        script_template_name = f"script_prompt_{prompt_version}.jinja" if prompt_version else "script_prompt.jinja"
        self.script_template = jinja_env.get_template(script_template_name)

    # --- Step 1: key_information ---

    def extract_key_info(self, case: Dict[str, Any]) -> List[str]:
        """Extract key_information list from a case."""
        prompt = self.key_info_template.render(
            language=self.language,
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
        raw = self._call_llm(prompt, max_tokens=800)
        return self._parse_json_array(raw)

    # --- Step 2: generate scripts for one config ---

    def generate_scripts(
        self,
        case: Dict[str, Any],
        config: Dict[str, str],
    ) -> Dict[str, Any]:
        """Generate behavioral scripts for one case + one experiment config.

        LLM generates scripts for all 6 dimensions. Caller adds "type" field.
        Retries up to MAX_RETRIES times on failure or incomplete output.

        Args:
            case: Case dict with key_information, Correct_Diagnosis, Social_History
            config: 6 dimension→level pairs, e.g.
                    {"disclosure": "withholding", "cognition": "accurate", ...}

        Returns:
            Dict keyed by dimension, e.g.
            {"disclosure": {"hidden": [...], "reveal_condition": "...", "instruction": "..."},
             "cognition": {"instruction": "..."}, ...}
        """
        key_info = case.get("key_information", [])
        prompt = self.script_template.render(
            language=self.language,
            key_information=key_info,
            diagnosis=case.get("Correct_Diagnosis", ""),
            social_history=case.get("Social_History", ""),
            disclosure=config.get("disclosure", "responsive"),
            cognition=config.get("cognition", "accurate"),
            logic=config.get("logic", "consistent"),
            attitude=config.get("attitude", "compliant"),
            purpose=config.get("purpose", "symptom_relief"),
            expression=config.get("expression", "normal"),
        )

        for attempt in range(MAX_RETRIES):
            raw = self._call_llm(prompt, max_tokens=2500)
            if not raw:
                print(
                    f"WARNING: Empty LLM response (attempt {attempt + 1}/{MAX_RETRIES})",
                    file=sys.stderr,
                )
                continue

            result = self._parse_json_object(raw)
            if not result:
                print(
                    f"WARNING: Failed to parse scripts JSON "
                    f"(attempt {attempt + 1}/{MAX_RETRIES})",
                    file=sys.stderr,
                )
                continue

            missing = self._validate_scripts(result)
            if not missing:
                return result

            print(
                f"WARNING: Scripts missing instruction for {missing} "
                f"(attempt {attempt + 1}/{MAX_RETRIES})",
                file=sys.stderr,
            )

        print(
            "ERROR: generate_scripts failed after all retries",
            file=sys.stderr,
        )
        return result if result else {}

    @staticmethod
    def _validate_scripts(result: Dict[str, Any]) -> List[str]:
        """Check that all 6 dimensions have an 'instruction' field.

        Returns list of dimension names missing instruction (empty = valid).
        """
        missing = []
        for dim in ALL_DIMENSIONS:
            entry = result.get(dim)
            if not isinstance(entry, dict) or not entry.get("instruction"):
                missing.append(dim)
        return missing

    # --- LLM helpers ---

    def _call_llm(self, prompt: str, max_tokens: int = 500) -> str:
        """Call LLM and return raw response text (single attempt)."""
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                **token_limit_param(self.model, max_tokens),
            )
            return response.choices[0].message.content or ""
        except Exception as e:
            print(f"LLM error: {e}", file=sys.stderr)
            return ""

    def _parse_json_array(self, raw: str) -> List[str]:
        """Parse JSON array from LLM response."""
        text = self._strip_fences(raw)
        try:
            data = json.loads(text)
            if isinstance(data, list):
                return [str(item) for item in data]
        except json.JSONDecodeError:
            match = re.search(r"\[.*\]", text, re.DOTALL)
            if match:
                try:
                    data = json.loads(match.group(0))
                    if isinstance(data, list):
                        return [str(item) for item in data]
                except json.JSONDecodeError:
                    pass
        return []

    def _parse_json_object(self, raw: str) -> Optional[Dict[str, Any]]:
        """Parse JSON object from LLM response."""
        text = self._strip_fences(raw)
        try:
            data = json.loads(text)
            if isinstance(data, dict):
                return data
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", text, re.DOTALL)
            if match:
                try:
                    data = json.loads(match.group(0))
                    if isinstance(data, dict):
                        return data
                except json.JSONDecodeError:
                    pass
        return None

    @staticmethod
    def _strip_fences(raw: str) -> str:
        """Remove markdown code fences from LLM output."""
        text = raw.strip()
        match = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", text, re.DOTALL)
        if match:
            return match.group(1).strip()
        return text
