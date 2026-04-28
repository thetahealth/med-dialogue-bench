"""
Experiment config definitions.

- DEFAULTS: baseline value per dimension
- ALL_LEVELS: all levels per dimension
- GENERATION_BATCHES: 6 packed configs covering all 31 unique dim+level pairs
- EXPERIMENT_CONFIGS: 42 experiment configs
  (1 baseline + 25 single-dim + 8 two-dim + 3 three-dim + 2 four-dim
   + 2 five-dim + 1 six-dim)
"""

from typing import Dict, List

# Default (baseline) value per dimension
DEFAULTS: Dict[str, str] = {
    "disclosure": "responsive",
    "cognition": "accurate",
    "logic": "consistent",
    "attitude": "compliant",
    "purpose": "symptom_relief",
    "expression": "normal",
}

# All levels per dimension (including defaults)
ALL_LEVELS: Dict[str, List[str]] = {
    "disclosure": ["responsive", "proactive", "selective", "reluctant", "withholding"],
    "cognition": [
        "accurate", "partial_understanding", "misconception",
        "web_self_diagnosis", "folk_remedy", "complete_denial",
    ],
    "logic": [
        "consistent", "minor_inconsistency", "occasional_contradiction",
        "frequent_contradiction", "fabricating",
    ],
    "attitude": ["compliant", "reluctant", "impatient", "evasive", "resistant", "dominant"],
    "purpose": ["symptom_relief", "reassurance", "medication_request"],
    "expression": ["clear", "normal", "vague", "verbose", "tangential", "incoherent"],
}

# ---------------------------------------------------------------------------
# Generation batches: 6 packed configs that cover all 31 unique dim+level pairs
# Each call generates scripts for all 6 dims; packing maximizes coverage per call.
#
# Batch 1-4: each covers 1 non-default level per dim (fills all 6 slots)
# Batch 5: remaining non-defaults (cognition, attitude, expression) + defaults for rest
# Batch 6: all defaults (baseline scripts)
# ---------------------------------------------------------------------------

GENERATION_BATCHES: List[Dict[str, str]] = [
    # Batch 1: 6 non-default levels
    {
        "disclosure": "proactive",
        "cognition": "partial_understanding",
        "logic": "minor_inconsistency",
        "attitude": "reluctant",
        "purpose": "reassurance",
        "expression": "clear",
    },
    # Batch 2: 6 non-default levels
    {
        "disclosure": "selective",
        "cognition": "misconception",
        "logic": "occasional_contradiction",
        "attitude": "impatient",
        "purpose": "medication_request",
        "expression": "vague",
    },
    # Batch 3: 5 non-default + 1 default (purpose=symptom_relief)
    {
        "disclosure": "reluctant",
        "cognition": "web_self_diagnosis",
        "logic": "frequent_contradiction",
        "attitude": "evasive",
        "purpose": "symptom_relief",
        "expression": "verbose",
    },
    # Batch 4: 5 non-default + 1 default (purpose=symptom_relief)
    {
        "disclosure": "withholding",
        "cognition": "folk_remedy",
        "logic": "fabricating",
        "attitude": "resistant",
        "purpose": "symptom_relief",
        "expression": "tangential",
    },
    # Batch 5: 3 non-default + 3 defaults
    {
        "disclosure": "responsive",
        "cognition": "complete_denial",
        "logic": "consistent",
        "attitude": "dominant",
        "purpose": "symptom_relief",
        "expression": "incoherent",
    },
    # Batch 6: all defaults (baseline)
    {
        "disclosure": "responsive",
        "cognition": "accurate",
        "logic": "consistent",
        "attitude": "compliant",
        "purpose": "symptom_relief",
        "expression": "normal",
    },
]


# ---------------------------------------------------------------------------
# 38 Experiment Configs
# ---------------------------------------------------------------------------

def _make_config(**overrides) -> Dict[str, str]:
    """Create a config from DEFAULTS with specified overrides."""
    config = dict(DEFAULTS)
    config.update(overrides)
    return config


def _build_experiment_configs() -> List[Dict[str, str]]:
    configs = []

    # 1. Baseline (1)
    configs.append(dict(DEFAULTS))

    # 2. Single-dimension (25): one dim at non-default, rest at default
    for dim, levels in ALL_LEVELS.items():
        for level in levels:
            if level != DEFAULTS[dim]:
                configs.append(_make_config(**{dim: level}))

    # 3. Two-dimension (8)
    # --- Original 4 (exploratory) ---
    configs.append(_make_config(attitude="evasive", cognition="misconception"))
    configs.append(_make_config(attitude="resistant", logic="frequent_contradiction"))
    configs.append(_make_config(disclosure="withholding", expression="incoherent"))
    configs.append(_make_config(cognition="misconception", logic="frequent_contradiction"))
    # --- Hypothesis-driven 4 (based on single-dim findings) ---
    configs.append(_make_config(logic="fabricating", disclosure="withholding"))        # cross-pathway: pollution + deficit
    configs.append(_make_config(logic="fabricating", expression="incoherent"))         # friction paradox vs pollution
    configs.append(_make_config(disclosure="withholding", attitude="dominant"))        # same-pathway: deficit + deficit
    configs.append(_make_config(cognition="complete_denial", logic="fabricating"))     # same-pathway: double pollution
    # --- New combos (C5 already above as withholding+incoherent) ---
    configs.append(_make_config(cognition="complete_denial", disclosure="withholding"))  # C6: cross-pathway pollution + deficit
    configs.append(_make_config(disclosure="reluctant", logic="fabricating"))            # C7: mild deficit + pollution

    # 4. Three-dimension (3)
    configs.append(_make_config(attitude="evasive", logic="frequent_contradiction", cognition="misconception"))
    configs.append(_make_config(disclosure="withholding", expression="incoherent", logic="frequent_contradiction"))
    configs.append(_make_config(attitude="evasive", disclosure="withholding", cognition="misconception"))

    # 5. Four-dimension (2)
    configs.append(_make_config(attitude="evasive", logic="frequent_contradiction", cognition="misconception", expression="incoherent"))
    configs.append(_make_config(disclosure="withholding", logic="fabricating", cognition="web_self_diagnosis", purpose="medication_request"))

    # 6. Five-dimension (2)
    configs.append(_make_config(attitude="evasive", disclosure="withholding", cognition="misconception", logic="frequent_contradiction", expression="incoherent"))
    configs.append(_make_config(attitude="resistant", disclosure="withholding", cognition="web_self_diagnosis", logic="fabricating", purpose="medication_request"))

    # 7. Six-dimension (1)
    configs.append(_make_config(attitude="evasive", disclosure="withholding", cognition="misconception", logic="frequent_contradiction", expression="incoherent", purpose="medication_request"))

    return configs


EXPERIMENT_CONFIGS: List[Dict[str, str]] = _build_experiment_configs()


def config_label(config: Dict[str, str]) -> str:
    """Human-readable label showing only non-default dimensions."""
    parts = []
    for dim in ("disclosure", "cognition", "logic", "attitude", "purpose", "expression"):
        level = config[dim]
        if level != DEFAULTS[dim]:
            parts.append(f"{dim}={level}")
    return "+".join(parts) if parts else "baseline"


def get_unique_dim_levels() -> List[tuple]:
    """Return all 31 unique (dimension, level) pairs needed across all experiments."""
    pairs = set()
    for config in EXPERIMENT_CONFIGS:
        for dim, level in config.items():
            pairs.add((dim, level))
    return sorted(pairs)
