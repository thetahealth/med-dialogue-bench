# Reproducing Paper Results

Every figure and statistic in the paper can be reproduced from the
pre-computed dialogues on HuggingFace. **No API keys required.**

## 1. Set up environment

```bash
pip install -e .
```

## 2. Download the dialogues dataset

```python
from datasets import load_dataset
ds = load_dataset("anon-meddial/meddial-dialogues")
```

**Current state of the scripts:** the five analysis scripts under
`scripts/` read from a local directory tree (`results_non_dynamic/`) that
mirrors how dialogues were written during the paper runs. Public users have
two options:

1. Use the HuggingFace dataset directly by replacing the disk I/O in each
   script with `load_dataset(...)`. A reshape helper lives at
   `scripts_internal/build_hf_dialogues.py` and documents the field mapping.
2. Rebuild a local `results_non_dynamic/` tree from the HuggingFace dataset:
   each row becomes `results_non_dynamic/<doctor_model>/<config>/dial_<case>.json`
   (turns) and `eval_<case>.json` (judge verdicts + metric fields).

A future revision of this repo will ship the scripts with native HF support
so this reshape step is not required.

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
