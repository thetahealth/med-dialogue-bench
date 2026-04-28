"""
APAF (Adversarial Patient Agent Framework) - Patient Agent

Implements a 6-dimension adversarial patient agent for medical dialogue benchmarking.
The agent maintains internal state (conversation history + dynamic psychological state)
and uses a single LLM call per turn. State updates are self-assessed by the patient
as part of its own reasoning (Inner Voice), not by a separate evaluator.

Dimensions (6 total):
- Fixed traits (unchanged during dialogue): Logic Consistency, Health Cognition, Expression Style
- Dynamic psychological state (evolve based on doctor behavior):
  - Attitude: non-linear switching (e.g., dominant -> compliant if doctor shows expertise)
  - Disclosure: gradual change along trust gradient (withholding <-> proactive)
  - Purpose: non-linear switching based on doctor's treatment approach
"""

import json
import logging
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from jinja2 import Environment, FileSystemLoader
from openai import OpenAI

from shared.clients import token_limit_param

logger = logging.getLogger(__name__)

# === APAF Dimension Definitions ===

ATTITUDE_VALUES = {
    "compliant": 1.0,
    "reluctant": 2.0,
    "impatient": 3.0,
    "evasive": 3.0,
    "resistant": 4.0,
    "dominant": 5.0,
}

LOGIC_VALUES = {
    "consistent": 1.0,
    "minor_inconsistency": 2.0,
    "occasional_contradiction": 3.0,
    "frequent_contradiction": 4.0,
    "fabricating": 5.0,
}

COGNITION_VALUES = {
    "accurate": 1.0,
    "partial_understanding": 2.0,
    "misconception": 3.0,
    "web_self_diagnosis": 3.5,
    "folk_remedy": 4.0,
    "complete_denial": 5.0,
}

EXPRESSION_VALUES = {
    "clear": 1.0,
    "normal": 1.5,
    "vague": 2.5,
    "verbose": 3.0,
    "tangential": 4.0,
    "incoherent": 5.0,
}

PURPOSE_VALUES = {
    "symptom_relief": 1.0,
    "reassurance": 3.0,
    "medication_request": 5.0,
}

DISCLOSURE_VALUES = {
    "proactive": 1.0,
    "responsive": 2.0,
    "selective": 3.0,
    "reluctant": 4.0,
    "withholding": 5.0,
}

APAF_WEIGHTS = {
    "attitude": 0.25,
    "logic": 0.20,
    "cognition": 0.20,
    "expression": 0.15,
    "purpose": 0.10,
    "disclosure": 0.10,
}

# Response parsing patterns
STATE_PATTERN = re.compile(
    r"\[STATE\](.*?)\[/STATE\]", re.DOTALL | re.IGNORECASE
)
REPLY_PATTERN = re.compile(
    r"\[REPLY\](.*?)\[/REPLY\]", re.DOTALL | re.IGNORECASE
)


# === Data Models ===


@dataclass
class PatientProfile:
    """APAF 6-dimension behavioral profile for the patient agent."""

    attitude: str = "compliant"
    logic: str = "consistent"
    cognition: str = "accurate"
    expression: str = "normal"
    purpose: str = "symptom_relief"
    disclosure: str = "responsive"

    def get_attitude_score(self) -> float:
        return ATTITUDE_VALUES.get(self.attitude, 1.0)

    def get_logic_score(self) -> float:
        return LOGIC_VALUES.get(self.logic, 1.0)

    def get_cognition_score(self) -> float:
        return COGNITION_VALUES.get(self.cognition, 1.0)

    def get_expression_score(self) -> float:
        return EXPRESSION_VALUES.get(self.expression, 1.0)

    def get_purpose_score(self) -> float:
        return PURPOSE_VALUES.get(self.purpose, 1.0)

    def get_disclosure_score(self) -> float:
        return DISCLOSURE_VALUES.get(self.disclosure, 1.0)

    def calculate_adversarial_score(self) -> float:
        """Compute weighted APAF adversarial score (1.0-5.0)."""
        return (
            self.get_attitude_score() * APAF_WEIGHTS["attitude"]
            + self.get_logic_score() * APAF_WEIGHTS["logic"]
            + self.get_cognition_score() * APAF_WEIGHTS["cognition"]
            + self.get_expression_score() * APAF_WEIGHTS["expression"]
            + self.get_purpose_score() * APAF_WEIGHTS["purpose"]
            + self.get_disclosure_score() * APAF_WEIGHTS["disclosure"]
        )

    def validate(self) -> None:
        """Validate all dimension values are recognized."""
        checks = [
            ("attitude", self.attitude, ATTITUDE_VALUES),
            ("logic", self.logic, LOGIC_VALUES),
            ("cognition", self.cognition, COGNITION_VALUES),
            ("expression", self.expression, EXPRESSION_VALUES),
            ("purpose", self.purpose, PURPOSE_VALUES),
            ("disclosure", self.disclosure, DISCLOSURE_VALUES),
        ]
        for name, value, valid_values in checks:
            if value not in valid_values:
                raise ValueError(
                    f"Invalid {name}: '{value}'. "
                    f"Must be one of {list(valid_values.keys())}"
                )

    @classmethod
    def from_test_case(cls, data: Dict[str, Any]) -> "PatientProfile":
        """Create profile from test case dict. Uses defaults if APAF fields absent."""
        return cls(
            attitude=data.get("APAF_Attitude", "compliant"),
            logic=data.get("APAF_Logic", "consistent"),
            cognition=data.get("APAF_Cognition", "accurate"),
            expression=data.get("APAF_Expression", "normal"),
            purpose=data.get("APAF_Purpose", "symptom_relief"),
            disclosure=data.get("APAF_Disclosure", "responsive"),
        )


@dataclass
class PatientState:
    """Dynamic psychological state tracked per-turn during dialogue.

    Stores the 3 dynamic APAF dimensions as labels (not numeric levels),
    since transitions are non-linear (especially for attitude and purpose).
    """

    attitude: str = "compliant"
    disclosure: str = "responsive"
    purpose: str = "symptom_relief"

    def snapshot(self) -> Dict[str, Any]:
        """Return a serializable snapshot for logging."""
        return {
            "attitude": self.attitude,
            "disclosure": self.disclosure,
            "purpose": self.purpose,
        }


@dataclass
class MedicalCase:
    """Medical case data (ground truth symptoms, demographics, and medical history)."""

    case_id: str = ""
    profile: Dict[str, str] = field(default_factory=dict)
    ground_truth_diagnosis: str = ""
    key_information: List[str] = field(default_factory=list)
    scripts: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_test_case(cls, data: Dict[str, Any]) -> "MedicalCase":
        """Convert from test case dict (AgentClinic OSCE format)."""
        # Demographics
        profile: Dict[str, str] = {}
        demo = data.get("Demographics", "")
        if demo:
            profile = {"demographics": demo}

        return cls(
            case_id=data.get("Case_ID", ""),
            profile=profile,
            ground_truth_diagnosis=(
                data.get("Correct_Diagnosis", "") or data.get("Ground_Truth_Diagnosis", "")
            ),
            key_information=data.get("key_information", []),
            scripts=data.get("scripts", {}),
        )


@dataclass
class StateUpdate:
    """Self-assessed state change from the patient's [STATE] block.

    Stores label-based state for the 3 dynamic APAF dimensions
    plus a reason field explaining why state did/didn't change.
    """

    attitude: str = ""
    disclosure: str = ""
    purpose: str = ""
    reason: str = ""
    prev_attitude: str = ""
    prev_disclosure: str = ""
    prev_purpose: str = ""
    raw_response: str = ""
    parse_success: bool = True


@dataclass
class PatientResponse:
    """Result from PatientAgent.respond()."""

    text: str = ""
    thought: str = ""
    state: Dict[str, Any] = field(default_factory=dict)
    is_terminated: bool = False
    termination_reason: str = ""
    turn_number: int = 0
    state_update: Optional[StateUpdate] = None
    raw_response: str = ""
    is_empty: bool = False


# === Patient Agent ===


class PatientAgent:
    """
    APAF-based adversarial patient agent for medical dialogue benchmark.

    Stateful agent that maintains conversation history and dynamic state.
    Uses a single LLM call per turn — state updates are self-assessed via
    the STATE block's reason field (dynamic mode: STATE + REPLY, static mode: plain text).

    Args:
        case: Medical case with ground truth symptoms
        profile: APAF 6-dimension behavioral profile
        model: LLM model for patient response generation
        api_key: API key for LLM
        base_url: Base URL for LLM
        language: 'zh' or 'en'
    """

    MAX_TURNS = 20

    def __init__(
        self,
        case: MedicalCase,
        profile: PatientProfile,
        model: Optional[str] = None,
        client_name: Optional[str] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        language: str = "en",
        dynamic_state: bool = False,
        prompt_version: Optional[str] = None,
        generic_scripts: bool = False,
    ):
        self.case = case
        self.profile = profile
        self.language = language
        self.dynamic_state = dynamic_state

        profile.validate()
        if not generic_scripts:
            self._validate_scripts(case, profile)

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

        # Conversation history
        self.messages: List[Dict[str, str]] = []
        self.turn: int = 0

        # Dynamic psychological state (3 dynamic APAF dimensions)
        self.state = PatientState(
            attitude=profile.attitude,
            disclosure=profile.disclosure,
            purpose=profile.purpose,
        )
        self.state_history: List[Dict[str, Any]] = []

        # Jinja2 template
        template_dir = Path(__file__).parent
        jinja_env = Environment(loader=FileSystemLoader(str(template_dir)))
        template_name = f"prompt_{prompt_version}.jinja" if prompt_version else "prompt.jinja"
        self.prompt_template = jinja_env.get_template(template_name)

    @staticmethod
    def _validate_scripts(case: MedicalCase, profile: PatientProfile) -> None:
        """Verify that every dimension has an instruction script.

        Raises ValueError if any dimension is missing its instruction,
        preventing silent fallback to generic defaults.
        """
        scripts = case.scripts
        if not scripts:
            return  # No scripts at all — used in unit tests / manual mode

        dims = {
            "disclosure": profile.disclosure,
            "cognition": profile.cognition,
            "logic": profile.logic,
            "attitude": profile.attitude,
            "purpose": profile.purpose,
            "expression": profile.expression,
        }
        missing = []
        for dim, level in dims.items():
            entry = scripts.get(dim, {})
            if not isinstance(entry, dict) or not entry.get("instruction"):
                missing.append(f"{dim}={level}")

        if missing:
            raise ValueError(
                f"Case {case.case_id}: missing instruction scripts for "
                f"dimensions: {missing}. "
                f"Generate scripts before running simulation."
            )

    def respond(self, doctor_message: str = "") -> PatientResponse:
        """Generate patient response.

        When doctor_message is empty, generates the opening complaint (turn 1).
        Static mode: plain text reply. Dynamic mode: STATE + REPLY.

        Args:
            doctor_message: The doctor's latest utterance (empty for opening)

        Returns:
            PatientResponse with text, thought, state, termination info
        """
        self.turn += 1

        # Add doctor message to history (skip for opening turn)
        if doctor_message:
            self.messages.append({"role": "doctor", "content": doctor_message})

        # Format dialogue history and build full prompt
        dialogue_history = self._format_history()
        prompt = self._build_prompt(dialogue_history)

        raw = ""
        for attempt in range(3):
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.8,
                **token_limit_param(self.model, 800),
            )
            raw = response.choices[0].message.content or ""
            if raw.strip():
                break
            finish = getattr(response.choices[0], "finish_reason", "unknown")
            logger.warning(
                "Empty response from %s (attempt %d/3, finish_reason=%s, turn=%d)",
                self.model, attempt + 1, finish, self.turn,
            )

        # Parse response blocks
        thought, state_update, reply = self._parse_response(raw)

        # Empty after retries → mark as failed, caller should abort
        if not reply.strip():
            logger.error(
                "Patient response empty after 3 retries (model=%s, turn=%d)",
                self.model, self.turn,
            )
            return PatientResponse(
                text="",
                state=self.state.snapshot(),
                turn_number=self.turn,
                raw_response=raw,
                is_empty=True,
            )

        # Apply self-assessed state update (dynamic mode only, skip turn 1)
        if self.dynamic_state and self.turn > 1 and state_update:
            self.state.attitude = state_update.attitude
            self.state.disclosure = state_update.disclosure
            self.state.purpose = state_update.purpose

        self.state_history.append(self.state.snapshot())

        # Append only the visible reply to history
        self.messages.append({"role": "patient", "content": reply})

        # Check termination after response
        should_term, term_reason = self.should_terminate()
        if should_term:
            return PatientResponse(
                text=reply,
                thought=thought,
                state=self.state.snapshot(),
                is_terminated=True,
                termination_reason=term_reason,
                turn_number=self.turn,
                state_update=state_update,
                raw_response=raw,
            )

        return PatientResponse(
            text=reply,
            thought=thought,
            state=self.state.snapshot(),
            is_terminated=False,
            turn_number=self.turn,
            state_update=state_update,
            raw_response=raw,
        )

    def _parse_response(self, raw: str) -> Tuple[str, Optional[StateUpdate], str]:
        """Parse state update and visible reply from patient LLM output.

        Dynamic mode format:
            [STATE]{"attitude": "...", "disclosure": "...", "purpose": "...", "reason": "..."}[/STATE]
            [REPLY]visible response[/REPLY]

        Static mode format:
            Plain text (the reply itself, no tags).

        Returns:
            (reason, state_update, reply) tuple.
            reason comes from the STATE JSON's "reason" field (dynamic mode only).
        """
        # Extract state update (dynamic mode)
        state_update = None
        reason = ""
        state_match = STATE_PATTERN.search(raw)
        if state_match:
            state_update = self._parse_state_json(state_match.group(1).strip())
            if state_update:
                reason = state_update.reason

        # Extract reply
        reply_match = REPLY_PATTERN.search(raw)
        if reply_match:
            reply = reply_match.group(1).strip()
        else:
            # Fallback: remove state block, use rest as reply
            reply = STATE_PATTERN.sub("", raw)
            reply = reply.replace("[REPLY]", "").replace("[/REPLY]", "").strip()

        return reason, state_update, reply

    def _parse_state_json(self, raw: str) -> Optional[StateUpdate]:
        """Parse state update JSON from [STATE] block content.

        Accepts label-based format:
            {"attitude": "reluctant", "disclosure": "selective", "purpose": "reassurance",
             "reason": "doctor was gentle so I relaxed a bit"}
        Validates labels against known values; unknown labels keep previous state.
        """
        json_str = raw.strip()

        # Handle code fences
        json_match = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", json_str, re.DOTALL)
        if json_match:
            json_str = json_match.group(1).strip()

        if not json_str.startswith("{"):
            brace_match = re.search(r"\{.*\}", json_str, re.DOTALL)
            if brace_match:
                json_str = brace_match.group(0)

        prev_att = self.state.attitude
        prev_dis = self.state.disclosure
        prev_pur = self.state.purpose

        try:
            data = json.loads(json_str)

            # Validate labels against known values; keep previous if unknown
            att = data.get("attitude", "")
            new_att = att if att in ATTITUDE_VALUES else prev_att

            dis = data.get("disclosure", "")
            new_dis = dis if dis in DISCLOSURE_VALUES else prev_dis

            pur = data.get("purpose", "")
            new_pur = pur if pur in PURPOSE_VALUES else prev_pur

            return StateUpdate(
                attitude=new_att,
                disclosure=new_dis,
                purpose=new_pur,
                reason=data.get("reason", ""),
                prev_attitude=prev_att,
                prev_disclosure=prev_dis,
                prev_purpose=prev_pur,
                raw_response=raw,
                parse_success=True,
            )
        except (json.JSONDecodeError, ValueError, TypeError):
            return None

    def should_terminate(self) -> Tuple[bool, str]:
        """Check whether the patient should terminate the consultation.

        The patient is a controlled stress-test tool — it never abandons.
        Only terminates when max turns is reached.
        The doctor controls inquiry termination via [END_INQUIRY].

        Returns:
            (should_terminate, reason) tuple
        """
        if self.turn >= self.MAX_TURNS:
            return True, "max_turns_reached"

        return False, ""

    def reset(self) -> None:
        """Reset agent state for a new dialogue."""
        self.messages = []
        self.turn = 0
        self.state = PatientState(
            attitude=self.profile.attitude,
            disclosure=self.profile.disclosure,
            purpose=self.profile.purpose,
        )
        self.state_history = []

    def calculate_adversarial_score(self) -> float:
        """Compute the APAF adversarial score for this profile."""
        return self.profile.calculate_adversarial_score()

    def _format_history(self) -> str:
        """Format message history into readable text for the prompt template."""
        if not self.messages:
            return ""
        role_labels = {
            "doctor": "Doctor" if self.language.startswith("en") else "医生",
            "patient": "Patient" if self.language.startswith("en") else "患者",
        }
        lines = []
        for msg in self.messages:
            label = role_labels.get(msg["role"], msg["role"])
            lines.append(f"{label}: {msg['content']}")
        return "\n".join(lines)

    def _build_prompt(self, dialogue_history: str = "") -> str:
        """Build the full prompt using current state, profile, and dialogue history."""
        # Resolve case-specific scripts (flat structure: scripts[dim] = {type, instruction, ...})
        scripts = self.case.scripts
        disclosure_script = scripts.get("disclosure", {}).get("instruction", "")
        cognition_script = scripts.get("cognition", {}).get("instruction", "")
        logic_script = scripts.get("logic", {}).get("instruction", "")
        attitude_script = scripts.get("attitude", {}).get("instruction", "")
        purpose_script = scripts.get("purpose", {}).get("instruction", "")
        expression_script = scripts.get("expression", {}).get("instruction", "")

        return self.prompt_template.render(
            language=self.language,
            profile=self.case.profile,
            key_information=self.case.key_information,
            dynamic_state=self.dynamic_state,
            # Fixed traits
            logic=self.profile.logic,
            cognition=self.profile.cognition,
            expression=self.profile.expression,
            # Dynamic state (labels)
            current_attitude=self.state.attitude,
            current_disclosure=self.state.disclosure,
            current_purpose=self.state.purpose,
            # Case-specific scripts
            disclosure_script=disclosure_script,
            cognition_script=cognition_script,
            logic_script=logic_script,
            attitude_script=attitude_script,
            purpose_script=purpose_script,
            expression_script=expression_script,
            turn=self.turn,
            dialogue_history=dialogue_history,
        )

