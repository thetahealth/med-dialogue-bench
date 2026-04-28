#!/usr/bin/env python3
"""Fig 3: Interaction Index bar chart — 4 combos, per-model II with bootstrap 95% CI."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif'],
    'font.size': 9,
    'figure.dpi': 300,
})

RESULTS_DIR = Path("results_non_dynamic")
CASES_FILE = Path("data/cases_filtered_85.json")

MODELS = [
    ("gpt-5.4", "GPT-5.4"),
    ("claude-opus-4-6", "Claude Opus 4.6"),
    ("deepseek-v3.2", "DeepSeek V3.2"),
    ("gemini-3.1-pro", "Gemini 3.1 Pro"),
    ("qwen3.5-plus", "Qwen 3.5 Plus"),
]

COMBOS = [
    ("C1: fabricating\n+ withholding", "disclosure=withholding+logic=fabricating",
     "disclosure=withholding", "logic=fabricating"),
    ("C2: fabricating\n+ incoherent", "logic=fabricating+expression=incoherent",
     "logic=fabricating", "expression=incoherent"),
    ("C3: fabricating\n+ denial", "cognition=complete_denial+logic=fabricating",
     "cognition=complete_denial", "logic=fabricating"),
    ("C4: withholding\n+ dominant", "disclosure=withholding+attitude=dominant",
     "disclosure=withholding", "attitude=dominant"),
    ("C5: withholding\n+ incoherent", "disclosure=withholding+expression=incoherent",
     "disclosure=withholding", "expression=incoherent"),
    ("C6: withholding\n+ denial", "disclosure=withholding+cognition=complete_denial",
     "cognition=complete_denial", "disclosure=withholding"),
]

COMBO_COLORS = ['#D62728', '#FF7F0E', '#1F77B4', '#9467BD']


def load_case_ids():
    with open(CASES_FILE) as f:
        return [c["Case_ID"] for c in json.load(f)]


def load_accuracy(model_dir, config, case_ids):
    """Return {case_id: 0/1} for accuracy."""
    result = {}
    for cid in case_ids:
        p = RESULTS_DIR / model_dir / config / f"eval_{cid}.json"
        if p.exists():
            with open(p) as f:
                e = json.load(f)
            result[cid] = 1 if e.get("semantic_accuracy") else 0
    return result


def bootstrap_ii(bl, a, b, c, n_boot=10000, seed=42):
    """Bootstrap Interaction Index and return (point, lo, hi)."""
    rng = np.random.RandomState(seed)
    n = len(bl)
    indices = []
    for _ in range(n_boot):
        idx = rng.choice(n, size=n, replace=True)
        bl_m = bl[idx].mean()
        a_m = a[idx].mean()
        b_m = b[idx].mean()
        c_m = c[idx].mean()
        expected = a_m * b_m / bl_m if bl_m > 0 else 0
        ii = c_m / expected if expected > 0 else 0
        indices.append(ii)
    indices = np.array(indices)
    bl_m = bl.mean()
    expected = a.mean() * b.mean() / bl_m if bl_m > 0 else 0
    point = c.mean() / expected if expected > 0 else 0
    return point, np.percentile(indices, 2.5), np.percentile(indices, 97.5)


def main():
    case_ids = load_case_ids()
    print(f"Using {len(case_ids)} cases")

    n_combos = len(COMBOS)
    n_models = len(MODELS)

    # Compute II per model per combo
    ii_data = []  # list of (combo_label, [(point, lo, hi) per model], mean_point)
    for combo_label, combo_cfg, dim_a, dim_b in COMBOS:
        model_iis = []
        for model_dir, _ in MODELS:
            bl = load_accuracy(model_dir, "baseline", case_ids)
            a = load_accuracy(model_dir, dim_a, case_ids)
            b = load_accuracy(model_dir, dim_b, case_ids)
            c = load_accuracy(model_dir, combo_cfg, case_ids)
            common = sorted(set(bl) & set(a) & set(b) & set(c))
            bl_arr = np.array([bl[cid] for cid in common])
            a_arr = np.array([a[cid] for cid in common])
            b_arr = np.array([b[cid] for cid in common])
            c_arr = np.array([c[cid] for cid in common])
            point, lo, hi = bootstrap_ii(bl_arr, a_arr, b_arr, c_arr)
            model_iis.append((point, lo, hi))
        mean_point = np.mean([m[0] for m in model_iis])
        ii_data.append((combo_label, model_iis, mean_point))

    # ── Plot ──
    fig, ax = plt.subplots(figsize=(7.0, 2.8))

    bar_width = 0.13
    x = np.arange(n_combos)
    # Add gap between fabricating group (0,1,2) and non-fabricating group (3,4,5)
    x = np.array([0, 1, 2, 3.3, 4.3, 5.3])
    model_colors = ['#4C72B0', '#55A868', '#C44E52', '#8172B2', '#CCB974']

    for mi, (model_dir, model_name) in enumerate(MODELS):
        offset = (mi - (n_models - 1) / 2) * bar_width
        points = [ii_data[ci][1][mi][0] for ci in range(n_combos)]
        los = [ii_data[ci][1][mi][1] for ci in range(n_combos)]
        his = [ii_data[ci][1][mi][2] for ci in range(n_combos)]
        yerr_lo = [p - l for p, l in zip(points, los)]
        yerr_hi = [h - p for p, h in zip(points, his)]

        ax.bar(x + offset, points, bar_width * 0.9, label=model_name,
               color=model_colors[mi], alpha=0.85)

    # Reference line at II = 1.0
    ax.axhline(y=1.0, color='#999999', linestyle='--', linewidth=0.8, zorder=0)
    ax.text(5.8, 1.10, 'O/E = 1.0\n(additive)', fontsize=7,
            color='#999999', ha='right', va='bottom')

    ax.set_xticks(x)
    ax.set_xticklabels([d[0] for d in ii_data], fontsize=8)
    ax.set_ylabel('O/E Ratio', fontsize=10)
    ax.set_ylim(0.45, 1.28)

    # Group labels at top
    ax.text(1.0, 1.25, 'fabricating-involving', ha='center', fontsize=7.5,
            fontweight='bold', color='#D62728')
    ax.text(4.3, 1.25, 'non-fabricating', ha='center', fontsize=7.5,
            fontweight='bold', color='#1F77B4')
    ax.legend(fontsize=7.5, ncol=5, loc='upper center', bbox_to_anchor=(0.5, 1.18),
              frameon=False, columnspacing=1.0)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    out_pdf = Path("prepare_paper/fig4_interaction_index.pdf")
    out_png = Path("prepare_paper/fig4_interaction_index.png")
    plt.savefig(out_pdf, format='pdf', bbox_inches='tight')
    plt.savefig(out_png, format='png', dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved {out_pdf} and {out_png}")

    # Print values for reference
    for combo_label, model_iis, mean_val in ii_data:
        vals = " | ".join(f"{m[0]:.2f} [{m[1]:.2f},{m[2]:.2f}]" for m in model_iis)
        print(f"  {combo_label}: {vals} | mean={mean_val:.2f}")


if __name__ == "__main__":
    main()
