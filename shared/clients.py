"""
Shared LLM client configuration.

Supports: nebula, openrouter
Used by: case_generator, patient_agent, doctor_agent
"""

import os
from pathlib import Path
from typing import Optional

from openai import OpenAI

_ENV_LOADED = False


def _load_env() -> None:
    """Load local.env from project root into os.environ (won't overwrite existing)."""
    global _ENV_LOADED
    if _ENV_LOADED:
        return
    _ENV_LOADED = True

    path = Path(__file__).resolve().parent.parent / ".env"
    if not path.exists():
        return
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key, value = key.strip(), value.strip()
            if key and key not in os.environ:
                os.environ[key] = value


CLIENT_CONFIGS = {
    "openrouter": {
        "base_url": "https://openrouter.ai/api/v1",
        "api_key_env": "OPENROUTER_API_KEY",
        "default_model": "anthropic/claude-3.5-sonnet",
    },
    "nebula": {
        "base_url": "https://llm.ai-nebula.com/v1",
        "api_key_env": "AI_NEBULA_API_KEY",
        "default_model": "qwen3.5-plus",
        "timeout": 120,
    },
}

DEFAULT_CLIENT = "nebula"
CLIENT_NAMES = list(CLIENT_CONFIGS.keys())

# Models that require max_completion_tokens instead of max_tokens
_NEW_API_PREFIXES = ("gpt-5", "o1", "o3", "o4")


def no_thinking_param(model: str) -> dict:
    """Return extra_body parameters to explicitly disable thinking/reasoning mode.

    Different providers use different parameters via nebula:
    - GPT: reasoning_effort='none'
    - Claude/DeepSeek/Qwen: enable_thinking=False
    - Gemini: not disableable via nebula client params (server-side)
    """
    m = model.lower()
    if "gpt" in m or m.startswith("o1") or m.startswith("o3") or m.startswith("o4"):
        return {"extra_body": {"reasoning_effort": "none"}}
    if "gemini" in m:
        # Gemini 3.x thinking cannot be disabled (integral to model architecture).
        # Nebula supports thinking_level=LOW/MEDIUM/HIGH (default HIGH).
        # We keep default (HIGH) — even LOW still produces reasoning_content,
        # so switching level only reduces but never eliminates thinking.
        return {}
    # Claude, DeepSeek, Qwen
    return {"extra_body": {"enable_thinking": False}}


def token_limit_param(model: str, value: int) -> dict:
    """Return the correct max token parameter for a given model.

    OpenAI's newer models (gpt-5.x, o1, o3, o4) require
    'max_completion_tokens'; all others use 'max_tokens'.
    """
    if any(model.startswith(p) for p in _NEW_API_PREFIXES):
        return {"max_completion_tokens": value}
    return {"max_tokens": value}


def create_client(client_name: str, model: Optional[str] = None) -> tuple:
    """Create OpenAI client from client name.

    Args:
        client_name: 'nebula' or 'openrouter'
        model: Override model name (uses client default if None)

    Returns:
        (OpenAI client, resolved model name)

    Raises:
        ValueError: Unknown client name or missing API key
    """
    _load_env()
    cfg = CLIENT_CONFIGS.get(client_name)
    if not cfg:
        raise ValueError(
            f"Unknown client: {client_name}. Choose from: {CLIENT_NAMES}"
        )

    api_key = os.environ.get(cfg["api_key_env"], "")
    if not api_key:
        raise ValueError(
            f"Environment variable {cfg['api_key_env']} not set for client '{client_name}'"
        )

    resolved_model = model or cfg["default_model"]
    timeout = cfg.get("timeout")
    openai_client = OpenAI(
        api_key=api_key, base_url=cfg["base_url"],
        **({"timeout": timeout} if timeout else {}),
    )

    return openai_client, resolved_model
