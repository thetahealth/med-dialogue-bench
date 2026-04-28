# Dataset Schemas

Both datasets are distributed via HuggingFace Datasets. The on-disk JSON in
this repo (`data/cases_filtered_85.json`) has the same content as the cases
dataset below but uses the legacy CamelCase column names.

## meddial-cases (85 rows)

One row per clinical vignette.

| Column                 | Type         | Description                                                     |
|------------------------|--------------|-----------------------------------------------------------------|
| `case_id`              | string       | e.g., `"001"`                                                   |
| `demographics`         | string       | Free text, e.g., `"35-year-old female"`                         |
| `history`              | string       | Free-text clinical history of present illness                   |
| `primary_symptom`      | string       | Chief complaint                                                 |
| `secondary_symptoms`   | string       | Semicolon-separated associated symptoms                         |
| `past_medical_history` | string       | Prior diagnoses, surgeries, etc.                                |
| `medications`          | string       | Current medications                                             |
| `family_history`       | string       | Relevant family history                                         |
| `social_history`       | string       | Smoking, alcohol, occupation                                    |
| `review_of_systems`    | string       | System-by-system review                                         |
| `objective_for_doctor` | string       | Diagnostic task description shown to the doctor agent           |
| `correct_diagnosis`    | string       | Ground-truth diagnosis                                          |
| `key_information`      | list[string] | Atomic facts used by the coverage metric                        |

## meddial-dialogues

One row per (case × config × doctor_model) dialogue.

| Column               | Type           | Description                                                              |
|----------------------|----------------|--------------------------------------------------------------------------|
| `dialogue_id`        | string         | `<case_id>__<config>__<doctor_model>`                                    |
| `case_id`            | string         | Join key to `meddial-cases`                                              |
| `config`             | string         | e.g. `"baseline"`, `"logic=fabricating"`, `"disclosure=withholding+attitude=dominant"` |
| `config_dimensions`  | dict[str,str]  | Parsed dimension -> level map                                            |
| `doctor_model`       | string         | e.g. `"gpt-5.4"`                                                         |
| `patient_model`      | string         | `"claude-opus-4-5-20251101"` (fixed across paper)                        |
| `turns`              | list[dict]     | Ordered turns; each has `role` (`"doctor"` or `"patient"`) and `content`. Patient turns additionally have `thought` (may be empty) and `state` (dimension labels at that turn). |
| `final_diagnosis`    | string         | Doctor's final diagnosis utterance                                       |
| `rounds`             | int            | Number of inquiry rounds                                                 |
| `terminated_by`      | string         | `"doctor"` or `"max_turns"`                                              |
| `judge_correct`      | bool           | Primary judge (Qwen3-Max): does the final diagnosis match ground truth?  |
| `judge_misled`       | bool or null   | Primary judge: was the error driven by the adversarial behavior? `null` when the diagnosis is correct. |
| `coverage`           | float          | Fraction of `key_information` items surfaced in the dialogue             |
| `efficiency`         | float          | Information elicited per turn                                            |
| `disclosure_turns`   | list[int]      | Turn index (1-based) at which each `key_information` item was disclosed; `0` if never |
| `covered_count`      | int            | `sum(1 for t in disclosure_turns if t > 0)`                              |
| `total_count`        | int            | `len(key_information)`                                                   |
| `optimal_stop_turn`  | int            | Earliest turn at which all elicited info was available                   |
| `wasted_turns`       | int            | Turns after `optimal_stop_turn`                                          |

**Known limitation.** The primary judge's free-text reasoning was not
persisted in the original run; only the boolean verdicts are available. A
future revision of this dataset may include rationales from a re-run.

**Secondary judge.** The secondary judge (Gemini 3 Pro) was applied only to a
220-dialogue sub-sample used to compute inter-judge agreement
(kappa = 0.882). That sub-sample and its secondary verdicts live in the
paper's supplementary materials rather than in the main dialogues dataset.
