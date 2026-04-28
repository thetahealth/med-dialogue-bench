"""
Doctor Agent - Two-Phase Medical Consultation Agent

Phase 1: Inquiry - Multi-turn dialogue to gather patient information
Phase 2: Diagnosis - Analyze dialogue history and provide clinical diagnosis
"""

import json
import logging
import os
import re
from pathlib import Path
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)
from dataclasses import dataclass, field

from openai import OpenAI
from jinja2 import Environment, FileSystemLoader

from shared.clients import token_limit_param, no_thinking_param


END_INQUIRY_MARKER = "[END_INQUIRY]"


@dataclass
class InquiryResult:
    """Result from a single inquiry turn"""
    text: str  # Doctor's question (marker stripped)
    is_terminated: bool  # True if doctor output [END_INQUIRY]
    raw_response: str  # Raw LLM output


@dataclass
class DiagnosisResult:
    """Result from the diagnosis phase"""
    diagnosis: str = ""
    differential: List[str] = field(default_factory=list)
    evidence: List[str] = field(default_factory=list)
    confidence: int = 0
    raw_response: str = ""
    parse_success: bool = False


class DoctorAgent:
    """
    Two-phase doctor agent for medical dialogue benchmark.

    Usage:
        doctor = DoctorAgent(model="gpt-4o")

        # Inquiry phase
        history = [{"role": "patient", "content": "My head hurts."}]
        result = doctor.inquiry_turn(history)
        print(result.text)  # "Where does it hurt?"

        # ... more turns ...

        # Diagnosis phase
        diagnosis = doctor.diagnose(history)
        print(diagnosis.diagnosis)  # "Migraine"
    """

    INQUIRY_PROMPTS = {
        "minimal": "inquiry_prompt.jinja",
        "guided": "inquiry_prompt_guided.jinja",
    }

    def __init__(
        self,
        model: Optional[str] = None,
        client_name: Optional[str] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        language: str = "zh",
        max_turns: int = 20,
        prompt_style: str = "minimal",
    ):
        self.language = language
        self.max_turns = max_turns
        self.prompt_style = prompt_style

        # LLM client — use shared.clients if available, fallback to direct OpenAI
        if client_name or (not api_key and not base_url):
            from shared.clients import create_client, DEFAULT_CLIENT
            client_name = client_name or DEFAULT_CLIENT
            self.client, self.model = create_client(client_name, model)
        else:
            self.client = OpenAI(
                api_key=api_key or os.environ.get("OPENAI_API_KEY"),
                base_url=base_url,
            )
            self.model = model or "gpt-4o"

        # Load Jinja2 templates
        template_dir = Path(__file__).parent
        jinja_env = Environment(loader=FileSystemLoader(str(template_dir)))
        inquiry_file = self.INQUIRY_PROMPTS.get(prompt_style, self.INQUIRY_PROMPTS["minimal"])
        self.inquiry_template = jinja_env.get_template(inquiry_file)
        self.diagnosis_template = jinja_env.get_template("diagnosis_prompt.jinja")

    MAX_RETRIES = 3

    def inquiry_turn(self, history: List[Dict[str, str]]) -> InquiryResult:
        """
        Generate one inquiry turn.

        Args:
            history: Conversation history with roles "doctor" and "patient"

        Returns:
            InquiryResult with the doctor's question or termination signal
        """
        formatted_history = self._format_history(history)

        prompt = self.inquiry_template.render(
            language=self.language,
            dialogue_history=formatted_history,
        )

        messages = [{"role": "user", "content": prompt}]

        for attempt in range(self.MAX_RETRIES):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=0.5,
                    **token_limit_param(self.model, 800),
                    **no_thinking_param(self.model),
                )
            except Exception as e:
                if attempt < self.MAX_RETRIES - 1:
                    logger.warning("API error (attempt %d/%d): %s", attempt + 1, self.MAX_RETRIES, e)
                    continue
                return InquiryResult(text="", is_terminated=False, raw_response=f"API error: {e}")

            choice = response.choices[0]
            raw = choice.message.content or ""
            finish_reason = choice.finish_reason

            if not raw and attempt < self.MAX_RETRIES - 1:
                logger.warning(
                    "Empty response from %s (attempt %d/%d, finish_reason=%s)",
                    self.model, attempt + 1, self.MAX_RETRIES, finish_reason,
                )
                continue

            if not raw:
                logger.warning(
                    "Empty response from %s after %d attempts (finish_reason=%s)",
                    self.model, self.MAX_RETRIES, finish_reason,
                )

            is_terminated = END_INQUIRY_MARKER in raw
            clean_text = raw.replace(END_INQUIRY_MARKER, "").strip()

            return InquiryResult(
                text=clean_text,
                is_terminated=is_terminated,
                raw_response=raw,
            )

        return InquiryResult(text="", is_terminated=False, raw_response="All retries exhausted")

    def diagnose(self, history: List[Dict[str, str]]) -> DiagnosisResult:
        """
        Generate diagnosis from dialogue history.

        Args:
            history: Full conversation history

        Returns:
            DiagnosisResult with diagnosis, differential, evidence, confidence
        """
        formatted_history = self._format_history(history)

        system_prompt = self.diagnosis_template.render(
            language=self.language,
            dialogue_history=formatted_history,
        )

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": system_prompt}],
            temperature=0.3,
            **token_limit_param(self.model, 1000),
            **no_thinking_param(self.model),
        )

        raw = response.choices[0].message.content or ""
        result = self._parse_diagnosis_json(raw)
        result.raw_response = raw
        return result

    def _format_history(self, history: List[Dict[str, str]]) -> str:
        """Format history into readable text for the diagnosis prompt"""
        role_labels = {
            "doctor": "Doctor" if self.language.startswith("en") else "医生",
            "patient": "Patient" if self.language.startswith("en") else "患者",
        }
        lines = []
        for msg in history:
            label = role_labels.get(msg["role"], msg["role"])
            lines.append(f"{label}: {msg['content']}")
        return "\n".join(lines)

    def _parse_diagnosis_json(self, raw: str) -> DiagnosisResult:
        """Parse JSON diagnosis from LLM output, handling markdown fences"""
        result = DiagnosisResult(raw_response=raw)

        # Try extracting JSON from markdown code block first
        json_match = re.search(r'```(?:json)?\s*\n?(.*?)\n?```', raw, re.DOTALL)
        json_str = json_match.group(1).strip() if json_match else raw.strip()

        # Try to find a JSON object if plain parsing fails
        if not json_str.startswith("{"):
            brace_match = re.search(r'\{.*\}', json_str, re.DOTALL)
            if brace_match:
                json_str = brace_match.group(0)

        try:
            data = json.loads(json_str)
            result.diagnosis = str(data.get("diagnosis", ""))
            result.differential = _ensure_list(data.get("differential", []))
            result.evidence = _ensure_list(data.get("evidence", []))
            result.confidence = _ensure_int(data.get("confidence", 0))
            result.parse_success = True
        except (json.JSONDecodeError, ValueError):
            # Fallback: treat entire raw output as diagnosis text
            result.diagnosis = raw.strip()
            result.parse_success = False

        return result


def _ensure_list(val: Any) -> List[str]:
    """Ensure value is a list of strings"""
    if isinstance(val, list):
        return [str(v) for v in val]
    if isinstance(val, str):
        return [val]
    return []


def _ensure_int(val: Any) -> int:
    """Ensure value is an int"""
    try:
        return int(val)
    except (ValueError, TypeError):
        return 0
