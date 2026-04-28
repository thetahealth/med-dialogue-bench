#!/usr/bin/env python3
"""Fig 2: Accuracy + Coverage heatmaps — 2 rows × 5 models horizontal, single-column layout."""

import json
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

# NeurIPS-friendly style
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif'],
    'font.size': 8,
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

DIMENSIONS = ["Expression", "Cognition", "Logic", "Disclosure", "Attitude"]
TIER_LABELS = ["B", "M", "E"]  # compact: Baseline/Moderate/Extreme

DIM_CONFIGS = {
    "Expression": ["baseline", "expression=vague", "expression=incoherent"],
    "Cognition": ["baseline", "cognition=partial_understanding", "cognition=complete_denial"],
    "Logic": ["baseline", "logic=occasional_contradiction", "logic=fabricating"],
    "Disclosure": ["baseline", "disclosure=reluctant", "disclosure=withholding"],
    "Attitude": ["baseline", "attitude=impatient", "attitude=dominant"],
}

PANELS = [
    ("accuracy", "Accuracy (%)", "RdYlGn", ".1f", 45, 95),
    ("coverage", "Coverage (0\u20131)", "RdYlGn", ".3f", 0.55, 0.85),
]


def load_case_ids() -> list[str]:
    with open(CASES_FILE) as f:
        cases = json.load(f)
    return [c["Case_ID"] for c in cases]


def compute_metrics(model_dir: str, config: str, case_ids: list[str]) -> dict:
    evals = []
    for cid in case_ids:
        p = RESULTS_DIR / model_dir / config / f"eval_{cid}.json"
        if p.exists():
            with open(p) as f:
                evals.append(json.load(f))
    n = len(evals)
    if n == 0:
        return {}
    accuracy = sum(1 for e in evals if e.get("semantic_accuracy")) / n * 100
    coverage = sum(e.get("coverage", 0) for e in evals) / n
    return {"accuracy": accuracy, "coverage": coverage}


def build_all_data(case_ids):
    all_data = {}
    for model_dir, _ in MODELS:
        matrices = {k: np.full((5, 3), np.nan) for k, *_ in PANELS}
        for di, dim in enumerate(DIMENSIONS):
            for ti, config in enumerate(DIM_CONFIGS[dim]):
                m = compute_metrics(model_dir, config, case_ids)
                for key, *_ in PANELS:
                    if key in m:
                        matrices[key][di, ti] = m[key]
        all_data[model_dir] = matrices
    return all_data


def main():
    case_ids = load_case_ids()
    print(f"Using {len(case_ids)} cases")
    all_data = build_all_data(case_ids)

    n_models = len(MODELS)
    n_panels = len(PANELS)

    fig = plt.figure(figsize=(7.0, 2.8))
    # 2 rows (metrics) × 6 cols (5 models + 1 colorbar)
    gs = gridspec.GridSpec(n_panels, n_models + 1, figure=fig,
                           width_ratios=[1] * n_models + [0.06],
                           wspace=0.12, hspace=0.18)

    for pi, (key, label, cmap_name, fmt, vmin, vmax) in enumerate(PANELS):
        ims = []
        for mi, (model_dir, display_name) in enumerate(MODELS):
            ax = fig.add_subplot(gs[pi, mi])
            matrix = all_data[model_dir][key]

            im = ax.imshow(matrix, cmap=cmap_name, aspect="auto", vmin=vmin, vmax=vmax)
            ims.append(im)

            # Y-axis: dimension labels (only leftmost model)
            ax.set_yticks(range(len(DIMENSIONS)))
            if mi == 0:
                ax.set_yticklabels(DIMENSIONS, fontsize=7)
            else:
                ax.set_yticklabels([])

            # X-axis: tier labels (only bottom row)
            ax.set_xticks(range(3))
            if pi == n_panels - 1:
                ax.set_xticklabels(TIER_LABELS, fontsize=7)
            else:
                ax.set_xticklabels([])

            # Model name on top (only first row)
            if pi == 0:
                ax.set_title(display_name, fontsize=8, fontweight='bold', pad=4)

            # Row label on left (only leftmost model)
            if mi == 0:
                ax.set_ylabel(label, fontsize=7.5, fontweight='bold', labelpad=4)

            # Cell annotations
            for i in range(5):
                for j in range(3):
                    val = matrix[i, j]
                    if np.isnan(val):
                        continue
                    norm_val = (val - vmin) / (vmax - vmin) if vmax > vmin else 0.5
                    txt_color = 'white' if norm_val < 0.2 or norm_val > 0.88 else 'black'
                    ax.text(j, i, f"{val:{fmt}}", ha='center', va='center',
                            fontsize=6.5, color=txt_color)

            ax.tick_params(length=0)  # hide tick marks

        # Colorbar on the right
        cax = fig.add_subplot(gs[pi, n_models])
        cbar = fig.colorbar(ims[-1], cax=cax)
        cbar.ax.tick_params(labelsize=6.5)

    # Legend for B/M/E at bottom
    fig.text(0.5, 0.005, 'B = Baseline    M = Moderate    E = Extreme',
             ha='center', fontsize=8, fontweight='bold', color='#555555')

    out_pdf = Path("prepare_paper/fig2_accuracy_coverage.pdf")
    out_png = Path("prepare_paper/fig2_accuracy_coverage.png")
    plt.savefig(out_pdf, format='pdf', bbox_inches='tight')
    plt.savefig(out_png, format='png', dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved {out_pdf} and {out_png}")


if __name__ == "__main__":
    main()
