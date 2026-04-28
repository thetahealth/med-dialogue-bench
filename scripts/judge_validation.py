#!/usr/bin/env python3
"""Judge validation: re-judge 200 cases with gemini-3-pro-preview, excluding pilot 20.

Combined with pilot's 20 cases → 220 total for agreement analysis.

Usage:
    PYTHONPATH=. python scripts/judge_validation.py --phase sample
    PYTHONPATH=. python scripts/judge_validation.py --phase rejudge
    PYTHONPATH=. python scripts/judge_validation.py --phase analyze
    PYTHONPATH=. python scripts/judge_validation.py   # all phases
"""

import argparse
import json
import random
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np

from benchmark.metrics import _llm_judge_diagnosis_and_misled
from shared.clients import create_client

RESULTS_DIR = Path("results_non_dynamic")
MODELS = ["gpt-5.4", "claude-opus-4-6", "deepseek-v3.2", "gemini-3.1-pro", "qwen3.5-plus"]
OUTPUT_DIR = RESULTS_DIR / "judge_validation"
PILOT_DIR = RESULTS_DIR / "judge_validation_pilot"

SECOND_JUDGE = "gemini-3-pro-preview"
CLIENT_NAME = "nebula"

N_CORRECT = 100
N_WRONG_NOT_MISLED = 50
N_WRONG_MISLED = 50
SEED = 42
CONCURRENCY = 5


# ===================== Phase 1: Sample =====================

def collect_all_evals():
    all_cases = []
    for model_dir in MODELS:
        model_path = RESULTS_DIR / model_dir
        if not model_path.exists():
            continue
        for config_dir in sorted(model_path.iterdir()):
            if not config_dir.is_dir():
                continue
            for eval_file in sorted(config_dir.glob("eval_*.json")):
                dial_file = eval_file.with_name(eval_file.name.replace("eval_", "dial_"))
                if not dial_file.exists():
                    continue
                with open(eval_file) as f:
                    ev = json.load(f)
                all_cases.append({
                    "model": model_dir,
                    "config": config_dir.name,
                    "case_id": ev.get("case_id", ""),
                    "eval_path": str(eval_file),
                    "dial_path": str(dial_file),
                    "original_correct": ev.get("semantic_accuracy"),
                    "original_misled": ev.get("misled"),
                })
    return all_cases


def load_pilot_keys():
    """Load pilot manifest to get keys of already-tested cases."""
    manifest_path = PILOT_DIR / "pilot_manifest.json"
    if not manifest_path.exists():
        return set()
    with open(manifest_path) as f:
        pilot = json.load(f)
    return {(c["model"], c["config"], c["case_id"]) for c in pilot}


def phase_sample():
    print("Collecting all eval files...")
    all_cases = collect_all_evals()
    print(f"Total evals: {len(all_cases)}")

    # Exclude pilot cases
    pilot_keys = load_pilot_keys()
    print(f"Excluding {len(pilot_keys)} pilot cases")
    pool = [c for c in all_cases if (c["model"], c["config"], c["case_id"]) not in pilot_keys]
    print(f"Remaining pool: {len(pool)}")

    # Stratify
    correct = [c for c in pool if c["original_correct"] is True]
    wrong_not_misled = [c for c in pool if c["original_correct"] is False and c["original_misled"] is not True]
    wrong_misled = [c for c in pool if c["original_correct"] is False and c["original_misled"] is True]
    print(f"Pool strata: correct={len(correct)}, wrong_not_misled={len(wrong_not_misled)}, wrong_misled={len(wrong_misled)}")

    rng = random.Random(SEED)
    sample = (rng.sample(correct, min(N_CORRECT, len(correct)))
              + rng.sample(wrong_not_misled, min(N_WRONG_NOT_MISLED, len(wrong_not_misled)))
              + rng.sample(wrong_misled, min(N_WRONG_MISLED, len(wrong_misled))))
    rng.shuffle(sample)
    print(f"Sampled: {len(sample)} cases (target: {N_CORRECT}+{N_WRONG_NOT_MISLED}+{N_WRONG_MISLED}={N_CORRECT+N_WRONG_NOT_MISLED+N_WRONG_MISLED})")

    # Verify model distribution
    from collections import Counter
    model_dist = Counter(c["model"] for c in sample)
    print(f"Model distribution: {dict(model_dist)}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_DIR / "sample_manifest.json", "w") as f:
        json.dump(sample, f, indent=2)
    print(f"Saved manifest to {OUTPUT_DIR / 'sample_manifest.json'}")


# ===================== Phase 2: Re-judge =====================

def rejudge_one(case, client, model):
    with open(case["dial_path"]) as f:
        dial = json.load(f)

    result = _llm_judge_diagnosis_and_misled(
        ground_truth=dial["ground_truth_diagnosis"],
        predicted=dial.get("diagnosis", ""),
        conversation=dial["dialogue"],
        client=client,
        model=model,
        key_information=dial.get("key_information", []),
    )
    return {
        **case,
        "second_judge": model,
        "second_correct": result["correct"],
        "second_misled": result["misled"],
    }


def result_path(case):
    return OUTPUT_DIR / f"rejudge_{case['model']}_{case['config']}_{case['case_id']}.json"


def phase_rejudge():
    manifest_path = OUTPUT_DIR / "sample_manifest.json"
    if not manifest_path.exists():
        print("ERROR: Run --phase sample first")
        return

    with open(manifest_path) as f:
        sample = json.load(f)

    # Check which are already done (resume support)
    todo = []
    done = 0
    for case in sample:
        if result_path(case).exists():
            done += 1
        else:
            todo.append(case)
    print(f"Total: {len(sample)}, already done: {done}, remaining: {len(todo)}")

    if not todo:
        print("All cases already re-judged.")
        return

    client, model = create_client(CLIENT_NAME, SECOND_JUDGE)
    completed = done
    errors = 0

    def process(case):
        r = rejudge_one(case, client, model)
        with open(result_path(case), "w") as f:
            json.dump(r, f, indent=2)
        return r

    with ThreadPoolExecutor(max_workers=CONCURRENCY) as executor:
        futures = {executor.submit(process, c): c for c in todo}
        for future in as_completed(futures):
            case = futures[future]
            completed += 1
            try:
                r = future.result()
                match = "OK" if r["original_correct"] == r["second_correct"] else "DISAGREE"
                print(f"  [{completed}/{len(sample)}] {case['model']}/{case['config']}/case_{case['case_id']} "
                      f"→ correct: {r['second_correct']} ({match})")
            except Exception as e:
                errors += 1
                print(f"  [{completed}/{len(sample)}] {case['model']}/{case['config']}/case_{case['case_id']} "
                      f"→ ERROR: {e}")

    print(f"\nDone. Errors: {errors}/{len(todo)}")


# ===================== Phase 3: Analyze =====================

def cohen_kappa(y1, y2):
    """Compute Cohen's kappa for two lists of binary labels."""
    n = len(y1)
    if n == 0:
        return float("nan")
    # Observed agreement
    po = sum(1 for a, b in zip(y1, y2) if a == b) / n
    # Expected agreement
    p1_pos = sum(y1) / n
    p2_pos = sum(y2) / n
    pe = p1_pos * p2_pos + (1 - p1_pos) * (1 - p2_pos)
    if pe == 1.0:
        return 1.0
    return (po - pe) / (1 - pe)


def bootstrap_kappa_ci(y1, y2, n_boot=10000, ci=0.95, seed=42):
    """Bootstrap 95% CI for Cohen's kappa."""
    rng = np.random.RandomState(seed)
    y1, y2 = np.array(y1), np.array(y2)
    n = len(y1)
    kappas = []
    for _ in range(n_boot):
        idx = rng.choice(n, size=n, replace=True)
        k = cohen_kappa(y1[idx].tolist(), y2[idx].tolist())
        if not np.isnan(k):
            kappas.append(k)
    alpha = (1 - ci) / 2
    lo = float(np.percentile(kappas, alpha * 100))
    hi = float(np.percentile(kappas, (1 - alpha) * 100))
    return lo, hi


def load_all_results():
    """Load rejudge results from both pilot and main validation."""
    results = []

    # Pilot results
    pilot_path = PILOT_DIR / "pilot_results.json"
    if pilot_path.exists():
        with open(pilot_path) as f:
            pilot = json.load(f)
        for r in pilot:
            r["source"] = "pilot"
        results.extend(pilot)
        print(f"Loaded {len(pilot)} pilot results")

    # Main results
    main_count = 0
    for f in sorted(OUTPUT_DIR.glob("rejudge_*.json")):
        with open(f) as fh:
            r = json.load(fh)
        r["source"] = "main"
        results.append(r)
        main_count += 1
    print(f"Loaded {main_count} main results")

    print(f"Total: {len(results)} results")
    return results


def phase_analyze():
    results = load_all_results()
    if not results:
        print("ERROR: No results found. Run --phase rejudge first.")
        return

    lines = []  # for saving report

    def pr(s=""):
        print(s)
        lines.append(s)

    pr(f"# Judge Validation Report")
    pr(f"")
    pr(f"**Primary judge:** qwen3-max")
    pr(f"**Second judge:** {SECOND_JUDGE}")
    pr(f"**Total cases:** {len(results)} (pilot: {sum(1 for r in results if r.get('source')=='pilot')}, "
       f"main: {sum(1 for r in results if r.get('source')=='main')})")
    pr()

    # --- 1. semantic_accuracy ---
    pr("## 1. Diagnostic Accuracy Agreement (semantic_accuracy)")
    pr()

    y1_correct = [1 if r["original_correct"] else 0 for r in results]
    y2_correct = [1 if r["second_correct"] else 0 for r in results]
    agree = sum(1 for a, b in zip(y1_correct, y2_correct) if a == b)
    kappa = cohen_kappa(y1_correct, y2_correct)
    lo, hi = bootstrap_kappa_ci(y1_correct, y2_correct)

    pr(f"- Agreement: {agree}/{len(results)} ({agree/len(results)*100:.1f}%)")
    pr(f"- Cohen's kappa: {kappa:.3f} [95% CI: {lo:.3f}, {hi:.3f}]")
    pr()

    # Confusion matrix
    tt = sum(1 for r in results if r["original_correct"] and r["second_correct"])
    tf = sum(1 for r in results if r["original_correct"] and not r["second_correct"])
    ft = sum(1 for r in results if not r["original_correct"] and r["second_correct"])
    ff = sum(1 for r in results if not r["original_correct"] and not r["second_correct"])
    pr("| | Second=Correct | Second=Incorrect |")
    pr("|---|---|---|")
    pr(f"| **Original=Correct** | {tt} | {tf} |")
    pr(f"| **Original=Incorrect** | {ft} | {ff} |")
    pr()

    # --- 2. misled ---
    both_wrong = [r for r in results if not r["original_correct"] and not r["second_correct"]]
    pr(f"## 2. Misled Detection Agreement (N={len(both_wrong)} where both say incorrect)")
    pr()

    if both_wrong:
        y1_misled = [1 if r["original_misled"] else 0 for r in both_wrong]
        y2_misled = [1 if r["second_misled"] else 0 for r in both_wrong]
        agree_m = sum(1 for a, b in zip(y1_misled, y2_misled) if a == b)
        kappa_m = cohen_kappa(y1_misled, y2_misled)
        lo_m, hi_m = bootstrap_kappa_ci(y1_misled, y2_misled)

        pr(f"- Agreement: {agree_m}/{len(both_wrong)} ({agree_m/len(both_wrong)*100:.1f}%)")
        pr(f"- Cohen's kappa: {kappa_m:.3f} [95% CI: {lo_m:.3f}, {hi_m:.3f}]")
        pr()

        mt = sum(1 for r in both_wrong if r["original_misled"] and r["second_misled"])
        mf = sum(1 for r in both_wrong if r["original_misled"] and not r["second_misled"])
        fm = sum(1 for r in both_wrong if not r["original_misled"] and r["second_misled"])
        fn = sum(1 for r in both_wrong if not r["original_misled"] and not r["second_misled"])
        pr("| | Second=Misled | Second=Not Misled |")
        pr("|---|---|---|")
        pr(f"| **Original=Misled** | {mt} | {mf} |")
        pr(f"| **Original=Not Misled** | {fm} | {fn} |")
        pr()

    # --- 3. Breakdown by model ---
    pr("## 3. Agreement by Doctor Model")
    pr()
    pr("| Model | N | Correct Agree | Correct Kappa |")
    pr("|---|---|---|---|")
    from collections import Counter
    for model in MODELS:
        subset = [r for r in results if r["model"] == model]
        if not subset:
            continue
        y1 = [1 if r["original_correct"] else 0 for r in subset]
        y2 = [1 if r["second_correct"] else 0 for r in subset]
        ag = sum(1 for a, b in zip(y1, y2) if a == b)
        k = cohen_kappa(y1, y2)
        pr(f"| {model} | {len(subset)} | {ag}/{len(subset)} ({ag/len(subset)*100:.1f}%) | {k:.3f} |")
    pr()

    # --- 4. Breakdown by config tier ---
    pr("## 4. Agreement by Config Tier")
    pr()

    def classify_tier(config):
        if config == "baseline":
            return "baseline"
        if "+" in config:
            return "multi-dim"
        # Check if extreme
        extremes = {"expression=incoherent", "cognition=complete_denial", "logic=fabricating",
                     "disclosure=withholding", "attitude=dominant"}
        if config in extremes:
            return "extreme"
        return "mild"

    pr("| Tier | N | Correct Agree | Correct Kappa |")
    pr("|---|---|---|---|")
    tiers = {}
    for r in results:
        t = classify_tier(r["config"])
        tiers.setdefault(t, []).append(r)
    for tier_name in ["baseline", "mild", "extreme", "multi-dim"]:
        subset = tiers.get(tier_name, [])
        if not subset:
            continue
        y1 = [1 if r["original_correct"] else 0 for r in subset]
        y2 = [1 if r["second_correct"] else 0 for r in subset]
        ag = sum(1 for a, b in zip(y1, y2) if a == b)
        k = cohen_kappa(y1, y2)
        pr(f"| {tier_name} | {len(subset)} | {ag}/{len(subset)} ({ag/len(subset)*100:.1f}%) | {k:.3f} |")
    pr()

    # --- 5. All disagreements ---
    disagree_correct = [r for r in results if r["original_correct"] != r["second_correct"]]
    pr(f"## 5. Disagreements on Correct ({len(disagree_correct)} cases)")
    pr()
    if disagree_correct:
        pr("| Model | Config | Case | Original | Second |")
        pr("|---|---|---|---|---|")
        for r in sorted(disagree_correct, key=lambda x: (x["model"], x["config"], x["case_id"])):
            pr(f"| {r['model']} | {r['config']} | {r['case_id']} | {r['original_correct']} | {r['second_correct']} |")
        pr()

    misled_disagree = [r for r in both_wrong if r["original_misled"] != r["second_misled"]]
    pr(f"## 6. Disagreements on Misled ({len(misled_disagree)} cases)")
    pr()
    if misled_disagree:
        pr("| Model | Config | Case | Original Misled | Second Misled |")
        pr("|---|---|---|---|---|")
        for r in sorted(misled_disagree, key=lambda x: (x["model"], x["config"], x["case_id"])):
            pr(f"| {r['model']} | {r['config']} | {r['case_id']} | {r['original_misled']} | {r['second_misled']} |")
        pr()

    # Save report
    report_path = OUTPUT_DIR / "agreement_report.md"
    with open(report_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\nReport saved to {report_path}")


# ===================== Main =====================

def main():
    parser = argparse.ArgumentParser(description="Judge validation (200 cases + 20 pilot)")
    parser.add_argument("--phase", choices=["sample", "rejudge", "analyze"],
                        help="Run specific phase (default: all)")
    args = parser.parse_args()

    if args.phase == "sample" or args.phase is None:
        phase_sample()
    if args.phase == "rejudge" or args.phase is None:
        phase_rejudge()
    if args.phase == "analyze" or args.phase is None:
        phase_analyze()


if __name__ == "__main__":
    main()
