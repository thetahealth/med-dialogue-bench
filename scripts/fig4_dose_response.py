#!/usr/bin/env python3
"""Fig 4: Dose-response curves — Cognition (smooth) vs Logic (threshold), 5 models."""

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

MODEL_COLORS = ['#4C72B0', '#55A868', '#C44E52', '#8172B2', '#CCB974']
MODEL_MARKERS = ['o', 's', '^', 'D', 'v']

DIMS = {
    "Cognition (Pollution)": [
        "baseline",
        "cognition=partial_understanding",
        "cognition=complete_denial",
    ],
    "Logic (Pollution)": [
        "baseline",
        "logic=occasional_contradiction",
        "logic=fabricating",
    ],
}

TIER_LABELS = ["Baseline", "Moderate", "Extreme"]


def load_case_ids():
    with open(CASES_FILE) as f:
        return [c["Case_ID"] for c in json.load(f)]


def compute_accuracy(model_dir, config, case_ids):
    evals = []
    for cid in case_ids:
        p = RESULTS_DIR / model_dir / config / f"eval_{cid}.json"
        if p.exists():
            with open(p) as f:
                evals.append(json.load(f))
    if not evals:
        return float('nan')
    return sum(1 for e in evals if e.get("semantic_accuracy")) / len(evals) * 100


def main():
    case_ids = load_case_ids()
    print(f"Using {len(case_ids)} cases")

    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.4), sharey=True)

    x = np.arange(3)

    for pi, (dim_label, configs) in enumerate(DIMS.items()):
        ax = axes[pi]

        for mi, (model_dir, model_name) in enumerate(MODELS):
            accs = [compute_accuracy(model_dir, cfg, case_ids) for cfg in configs]
            ax.plot(x, accs, marker=MODEL_MARKERS[mi], color=MODEL_COLORS[mi],
                    linewidth=1.5, markersize=5, label=model_name, alpha=0.85)

        ax.set_xticks(x)
        ax.set_xticklabels(TIER_LABELS, fontsize=9)
        ax.set_title(dim_label, fontsize=10, fontweight='bold', pad=6)
        ax.set_xlim(-0.15, 2.15)
        ax.grid(axis='y', alpha=0.3, linewidth=0.5)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        if pi == 0:
            ax.set_ylabel('Accuracy (%)', fontsize=10)

    # Annotate patterns — position in whitespace to avoid overlapping lines
    axes[0].annotate('smooth\ndegradation', xy=(1.6, 90), fontsize=8,
                     color='#666666', fontstyle='italic', ha='center')
    axes[1].annotate('threshold\neffect', xy=(1.6, 90), fontsize=8,
                     color='#D62728', fontstyle='italic', ha='center')

    axes[0].set_ylim(46, 96)

    # Shared legend
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, fontsize=8, ncol=5,
               loc='upper center', bbox_to_anchor=(0.5, 1.06),
               frameon=False, columnspacing=1.5)

    plt.tight_layout()

    out_pdf = Path("prepare_paper/fig3_dose_response.pdf")
    out_png = Path("prepare_paper/fig3_dose_response.png")
    plt.savefig(out_pdf, format='pdf', bbox_inches='tight')
    plt.savefig(out_png, format='png', dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved {out_pdf} and {out_png}")


if __name__ == "__main__":
    main()
