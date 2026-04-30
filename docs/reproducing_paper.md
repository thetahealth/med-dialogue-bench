# Reproducing Paper Results

Every figure and statistic in the paper can be reproduced from the
pre-computed dialogues hosted on Hugging Face. **No API keys required.**

## 1. Set up environment

```bash
pip install -e .
pip install huggingface_hub
```

## 2. Download the dialogues dataset

The Hugging Face dataset contains the full directory tree
(`claude-opus-4-6/`, `qwen3.5-plus/`, `deepseek-v3.2/`, `gemini-3.1-pro/`,
`gpt-5.4/`, `judge_validation/`, `adherence/`, plus `cases_filtered_85.json`).
Download it into `results_non_dynamic/` at the repository root:

```bash
hf download anon-meddial-2026/meddialbench \
    --repo-type=dataset \
    --local-dir=results_non_dynamic
```

The five analysis scripts under `scripts/` read directly from this
`results_non_dynamic/` tree, so no further setup is required.

Croissant metadata for the dataset is at
`https://huggingface.co/api/datasets/anon-meddial-2026/meddialbench/croissant`.

## 3. Regenerate results

| Command | Paper artifact |
|---------|----------------|
| `python scripts/statistical_tests.py`      | Table 3 (accuracy per model x config), Table 4 (McNemar + power) |
| `python scripts/fig2_accuracy_coverage.py` | Figure 2 (accuracy x coverage heatmap) |
| `python scripts/fig3_interaction_index.py` | Figure 4 (O/E ratio bar chart for interactions) |
| `python scripts/fig4_dose_response.py`     | Figure 3 (mild -> extreme dose-response curves) |
| `python scripts/judge_validation.py`       | Appendix B: Cohen's kappa = 0.882 (Qwen3-Max vs Gemini 3 Pro), 220-case sub-sample |

Each script writes outputs next to itself (`*.pdf`, `*.png`) and prints
the headline numbers that appear in the paper text.

## Expected numbers to verify against

- Overall super-additive rate across the 4 interaction configs: 32.1%
- Super-additive rate for pollution-involving combinations: 35-44%
- Super-additive rate for deficit-only combination (C3): 10.4%
- Pollution-vs-deficit accuracy drop ratio: 1.7x - 3.4x
- Worst-case accuracy drop range: 38.8 - 54.1 percentage points
- Judge agreement (semantic accuracy): kappa = 0.882

## Running fresh evaluations

If you want to evaluate a new doctor model against the benchmark, see the
quickstart (`examples/quickstart.py`) and `benchmark/runner.py`.
