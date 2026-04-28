#!/usr/bin/env python3
"""Statistical significance tests for MedDialBench results.

1. McNemar test: paired comparison of accuracy between configs (per model)
2. Bootstrap CI: confidence intervals for all metrics
"""

import json
from pathlib import Path
from collections import defaultdict

import numpy as np
from scipy import stats

RESULTS_DIR = Path("results_non_dynamic")
CASES_FILE = Path("data/cases_filtered_85.json")

MODELS = [
    ("gpt-5.4", "GPT-5.4"),
    ("claude-opus-4-6", "Claude Opus 4.6"),
    ("deepseek-v3.2", "DeepSeek V3.2"),
    ("gemini-3.1-pro", "Gemini 3.1 Pro"),
    ("qwen3.5-plus", "Qwen 3.5 Plus"),
]

CONFIGS = [
    "baseline",
    "expression=vague",
    "expression=incoherent",
    "cognition=partial_understanding",
    "cognition=complete_denial",
    "logic=occasional_contradiction",
    "logic=fabricating",
    "disclosure=reluctant",
    "disclosure=withholding",
    "attitude=impatient",
    "attitude=dominant",
]

# McNemar comparisons: (config_a, config_b, description)
MCNEMAR_PAIRS = [
    # Each extreme config vs baseline
    ("baseline", "expression=incoherent", "baseline vs incoherent"),
    ("baseline", "cognition=complete_denial", "baseline vs complete_denial"),
    ("baseline", "logic=fabricating", "baseline vs fabricating"),
    ("baseline", "disclosure=withholding", "baseline vs withholding"),
    ("baseline", "attitude=dominant", "baseline vs dominant"),
    # Cross-dimension: fabricating vs vague (info veracity >> expression quality)
    ("logic=fabricating", "expression=vague", "fabricating vs vague"),
    ("logic=fabricating", "expression=incoherent", "fabricating vs incoherent"),
    # Info pollution vs info deficit extremes
    ("logic=fabricating", "disclosure=withholding", "fabricating vs withholding"),
]


def load_case_ids() -> list[str]:
    with open(CASES_FILE) as f:
        return [c["Case_ID"] for c in json.load(f)]


def load_eval(model_dir: str, config: str, case_id: str) -> dict | None:
    p = RESULTS_DIR / model_dir / config / f"eval_{case_id}.json"
    if p.exists():
        with open(p) as f:
            return json.load(f)
    return None


def load_per_case_accuracy(model_dir: str, config: str, case_ids: list[str]) -> dict[str, int]:
    """Returns {case_id: 1/0} for accuracy."""
    result = {}
    for cid in case_ids:
        ev = load_eval(model_dir, config, cid)
        if ev:
            result[cid] = 1 if ev.get("semantic_accuracy") else 0
    return result


def load_per_case_metrics(model_dir: str, config: str, case_ids: list[str]) -> list[dict]:
    """Returns list of per-case metric dicts."""
    results = []
    for cid in case_ids:
        ev = load_eval(model_dir, config, cid)
        if ev:
            rounds = ev.get("rounds", 0)
            coverage = ev.get("coverage", 0)
            results.append({
                "case_id": cid,
                "accuracy": 1 if ev.get("semantic_accuracy") else 0,
                "misled": 1 if ev.get("misled") else 0,
                "coverage": coverage,
                "rounds": rounds,
                "efficiency": coverage / rounds if rounds > 0 else 0,
                "doc_term": 1 if ev.get("terminated_by") == "doctor" else 0,
            })
    return results


# ===================== McNemar Test =====================

def mcnemar_test(acc_a: dict[str, int], acc_b: dict[str, int]) -> dict:
    """McNemar test on paired accuracy data.

    Returns contingency table counts and p-value.
    acc_a, acc_b: {case_id: 1/0}
    """
    common_ids = sorted(set(acc_a.keys()) & set(acc_b.keys()))
    n = len(common_ids)

    # a: both correct, b: only A correct, c: only B correct, d: both wrong
    a = b = c = d = 0
    for cid in common_ids:
        va, vb = acc_a[cid], acc_b[cid]
        if va == 1 and vb == 1:
            a += 1
        elif va == 1 and vb == 0:
            b += 1
        elif va == 0 and vb == 1:
            c += 1
        else:
            d += 1

    # McNemar's test
    if b + c == 0:
        p_value = 1.0
    elif (b + c) < 25:
        # exact binomial test (more appropriate for small samples)
        p_value = stats.binomtest(b, b + c, 0.5).pvalue
    else:
        # chi-square approximation with continuity correction
        chi2 = (abs(b - c) - 1) ** 2 / (b + c)
        p_value = 1 - stats.chi2.cdf(chi2, df=1)

    return {
        "n": n,
        "both_correct": a,
        "only_A_correct": b,
        "only_B_correct": c,
        "both_wrong": d,
        "p_value": p_value,
    }


# ===================== Bootstrap CI =====================

def bootstrap_ci(values: np.ndarray, stat_fn=np.mean, n_boot: int = 10000,
                 ci: float = 0.95, seed: int = 42) -> tuple[float, float, float]:
    """Returns (point_estimate, ci_low, ci_high)."""
    rng = np.random.RandomState(seed)
    point = float(stat_fn(values))
    boot_stats = np.array([
        stat_fn(rng.choice(values, size=len(values), replace=True))
        for _ in range(n_boot)
    ])
    alpha = (1 - ci) / 2
    lo = float(np.percentile(boot_stats, alpha * 100))
    hi = float(np.percentile(boot_stats, (1 - alpha) * 100))
    return point, lo, hi


# ===================== Main =====================

def run_mcnemar(case_ids: list[str]):
    print("=" * 80)
    print("McNEMAR TEST RESULTS (Accuracy: config_A vs config_B)")
    print("=" * 80)

    for model_dir, model_name in MODELS:
        print(f"\n### {model_name}")
        print(f"{'Comparison':<35} {'Acc_A':>6} {'Acc_B':>6} {'Δ':>7} "
              f"{'b(A>B)':>6} {'c(B>A)':>6} {'p-value':>10} {'sig':>5}")
        print("-" * 90)

        for cfg_a, cfg_b, desc in MCNEMAR_PAIRS:
            acc_a = load_per_case_accuracy(model_dir, cfg_a, case_ids)
            acc_b = load_per_case_accuracy(model_dir, cfg_b, case_ids)
            result = mcnemar_test(acc_a, acc_b)

            n = result["n"]
            acc_a_pct = (result["both_correct"] + result["only_A_correct"]) / n * 100
            acc_b_pct = (result["both_correct"] + result["only_B_correct"]) / n * 100
            delta = acc_a_pct - acc_b_pct
            p = result["p_value"]

            sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"

            print(f"{desc:<35} {acc_a_pct:>5.1f}% {acc_b_pct:>5.1f}% {delta:>+6.1f}pp "
                  f"{result['only_A_correct']:>5}  {result['only_B_correct']:>5}  "
                  f"{p:>9.4f}  {sig:>4}")


def run_bootstrap(case_ids: list[str]):
    print("\n" + "=" * 80)
    print("BOOTSTRAP 95% CONFIDENCE INTERVALS")
    print("=" * 80)

    metrics_keys = [
        ("accuracy", "Accuracy (%)", 100, ".1f"),
        ("coverage", "Coverage", 1, ".3f"),
        ("misled", "Misled Rate (%)", 100, ".1f"),
        ("efficiency", "Efficiency", 1, ".3f"),
        ("rounds", "Avg. Turns", 1, ".1f"),
    ]

    for model_dir, model_name in MODELS:
        print(f"\n### {model_name}")
        print(f"{'Config':<30} ", end="")
        for _, label, _, _ in metrics_keys:
            print(f"{label:>30}", end="")
        print()
        print("-" * 180)

        for config in CONFIGS:
            cases = load_per_case_metrics(model_dir, config, case_ids)
            if not cases:
                continue

            print(f"{config:<30} ", end="")
            for key, _, scale, fmt in metrics_keys:
                values = np.array([c[key] for c in cases])
                point, lo, hi = bootstrap_ci(values)
                point *= scale
                lo *= scale
                hi *= scale
                print(f"  {point:{fmt}} [{lo:{fmt}}, {hi:{fmt}}]", end="")
            print()


def run_cross_model_fabricating(case_ids: list[str]):
    """Check if fabricating is #1 killer across all models."""
    print("\n" + "=" * 80)
    print("FABRICATING vs ALL OTHER EXTREME CONFIGS (per model)")
    print("=" * 80)

    extreme_configs = [
        ("expression=incoherent", "Expression"),
        ("cognition=complete_denial", "Cognition"),
        ("logic=fabricating", "Logic"),
        ("disclosure=withholding", "Disclosure"),
        ("attitude=dominant", "Attitude"),
    ]

    for model_dir, model_name in MODELS:
        print(f"\n### {model_name}")
        fab_acc = load_per_case_accuracy(model_dir, "logic=fabricating", case_ids)
        bl_acc = load_per_case_accuracy(model_dir, "baseline", case_ids)

        for config, dim in extreme_configs:
            if config == "logic=fabricating":
                continue
            other_acc = load_per_case_accuracy(model_dir, config, case_ids)
            result = mcnemar_test(fab_acc, other_acc)
            n = result["n"]
            fab_pct = (result["both_correct"] + result["only_A_correct"]) / n * 100
            oth_pct = (result["both_correct"] + result["only_B_correct"]) / n * 100
            p = result["p_value"]
            sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
            print(f"  fabricating ({fab_pct:.1f}%) vs {dim} extreme ({oth_pct:.1f}%): "
                  f"Δ={fab_pct - oth_pct:+.1f}pp, p={p:.4f} {sig}")


def main():
    case_ids = load_case_ids()
    print(f"Using {len(case_ids)} cases from {CASES_FILE}\n")

    run_mcnemar(case_ids)
    run_cross_model_fabricating(case_ids)
    run_bootstrap(case_ids)


if __name__ == "__main__":
    main()
