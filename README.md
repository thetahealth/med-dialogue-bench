# MedDialBench

A controlled factorial benchmark for evaluating LLM diagnostic robustness under parametric adversarial patient behaviors.

- **Paper:** under review, NeurIPS 2026
- **Code:** this repository
- **Cases dataset:** https://huggingface.co/datasets/anon-meddial/meddial-cases
- **Dialogues dataset:** https://huggingface.co/datasets/anon-meddial/meddial-dialogues

## What's in this repo

- `doctor_agent/` — two-phase doctor agent (inquiry then diagnosis), stateless
- `patient_agent/` — stateful adversarial patient; 6 behavioral dimensions (3 static, 3 dynamic)
- `case_generator/` — MedQA-derived case extraction and colloquialization pipeline
- `benchmark/` — experiment runner, metrics, judge prompts
- `scripts/` — analysis scripts that regenerate every figure and statistic in the paper
- `examples/quickstart.py` — end-to-end demo (one case, one config, one doctor model)
- `data/cases_filtered_85.json` — the 85 clinical vignettes used in the paper
- `docs/` — data schema, reproduction guide, ethics statement

## Quick start

```bash
git clone <repo-url> && cd med-dialogue-bench
pip install -e .
cp .env.example .env      # fill in your API keys
python examples/quickstart.py
```

The quickstart runs one case × one behavioral config × one doctor model (~1 minute, ~$0.01).

## Reproducing the paper

Every figure and statistic can be reproduced **without running any model** by downloading the pre-computed dialogues from HuggingFace:

```bash
pip install datasets
python scripts/statistical_tests.py        # paper Table 3, Table 4
python scripts/fig2_accuracy_coverage.py   # paper Figure 2
python scripts/fig3_interaction_index.py   # paper Figure 4 (O/E ratio)
python scripts/fig4_dose_response.py       # paper Figure 3
python scripts/judge_validation.py         # paper kappa = 0.882
```

See `docs/reproducing_paper.md` for a step-by-step walkthrough.

## Running your own evaluation

To re-run the full evaluation (requires API keys for all models):

```bash
python -m benchmark.runner \
    --cases data/cases_filtered_85.json \
    --config baseline \
    --doctor-model gpt-5.4 \
    --patient-model claude-opus-4-5-20251101 \
    --judge-model qwen3-max \
    --out results/my_run
```

Cost estimate for a full 85-case × 17-config × 5-model re-run: approximately USD $600–$900 depending on provider pricing at time of run.

## Citation

See `CITATION.bib`.

## License

- Code: MIT (see `LICENSE`)
- Data: CC BY 4.0 (see `LICENSE-data`)
- MedQA-derived cases remain subject to the upstream MedQA MIT License.

## Ethics and limitations

These cases are fictional research artifacts and must not be used for clinical decision-making. See `docs/ethics.md`.
