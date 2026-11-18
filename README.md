# FebSynth Code

Generation and validation code, with pretrained checkpoints, accompanying **A clinically constrained synthetic dataset of febrile infants for hospitalization and ICU risk prediction**.

## Installation

Use Python 3.10 for the validated checkpoint environment. From this folder:

```bash
python -m pip install -r requirements-models.txt
```

`requirements-models.txt` pins the checkpoint-loading dependencies and includes the general runtime requirements. `models/manifest.json` records the versions used for loading and sampling checks. For GPU execution, install the matching PyTorch build for your CUDA environment.

## Input And Configuration

The generation pipeline accepts a preprocessed CSV with the Chinese source-column names defined in `shared_generation.py`. The supplied `constraints.json` and `grouping.json` use the same names. The generation and evaluation entry points use `translation.json` and `output_localization.py` to label newly generated tables, summaries and figures. `--output_language en` controls output labelling, not input-column conversion.

The published analytical tables contain 21 continuous fields, Sex, 22 symptoms and two outcomes (46 fields). Diagnosis ICD Code is used for stratification and is not an output feature. Missing diagnosis values enter Other. Symptom missingness remains distinct from absence. The source table may also contain Platelet Distribution Width, which is not selected for the 46-field table.

Pretrained checkpoints are included in `models/`; private clinical inputs are not. The English synthetic tables in the separate data record are research outputs, not replacements for the restricted real training and held-out inputs required to reproduce real-reference validation.

Paths beginning `/path/to/` are examples to replace for your environment. Relative paths to bundled modules and configurations are portable. Run the commands below from this folder.

## Main Pipeline

```bash
python pipeline.py --model copula --input /path/to/real_data.csv --output_root /path/to/output_en/evaluate_results --constraints constraints.json --grouping grouping.json --test_size 0.2 --random_state 42 --output_language en
```

Supported generators are `copula`, `ctgan`, `tvae` and `tabddpm`. The pipeline splits the real table, fits on training records, generates a matched table by default, and evaluates it against the appropriate real training and held-out references. The study split contains 18,604 training and 4,651 held-out visits.

Use `--n_samples 50000` for the extended size, `--cpu` for CPU execution, and `--model_path /path/to/models/copula` to choose a model directory. Without an override, saved models use `models/<model>/` relative to the code folder. Add `--load_model` to reuse the bundled checkpoints with the corresponding study input and configuration. Without it, the pipeline trains models. Keep model directories specific to the training input and configuration, and use a separate `--model_path` when training on new data.

For the complete English experiment sequence:

```bash
INPUT_CSV=/path/to/real_data.csv OUTPUT_ROOT=/path/to/output_en bash run_scientific_data_experiments.sh
```

The script runs the four generators, Copula ablations, sensitivity analysis and the extension, with model reuse enabled for the main comparison. The complete ablation setting copies the main Copula run, not an independent replicate. Sensitivity analysis fits its own generators for the specified training subsets. Invoke the script with `bash`.

## Pretrained Checkpoints

`models/` contains 105 `model_best.pkl` checkpoints: 26 strata for each of the four generators, plus one ungrouped Copula checkpoint. Directory names are preserved for the existing grouped-model loader. `model_last.pkl` and standalone training-history files are not included because inference uses the best checkpoint. Each file's SHA-256 and preparation details are listed in `models/manifest.json`.

The CTGAN checkpoints retain inference parameters, fitted transformations and category probabilities, but omit the per-category training-row index cache. Released tensor states are CPU-portable. Checkpoints are for generation, not exact continuation of the original training process.

A single grouped checkpoint generates its stratum's modelled columns; the grouping pipeline restores fixed grouping fields and combines strata. Reproducing the full study sampling allocation and real-reference evaluation still requires the corresponding restricted input and configuration.

Pickle files can execute code when loaded. Load only checkpoints from a trusted source and verify their manifest hashes.

## Other Entry Points

```bash
python main_plus.py --list_models
python main_plus.py --model copula --input /path/to/real_data.csv --output /path/to/synthetic_data.csv --constraints constraints.json --grouping grouping.json
python nsample_sensitivity.py --model copula --input /path/to/real_data.csv --output_root /path/to/output_en/nsample_sensitivity --constraints constraints.json --grouping grouping.json --output_language en
python evaluate.py --real /path/to/test_real.csv --train_real /path/to/train_real.csv --synthetic /path/to/synthetic_data.csv --output_dir /path/to/evaluation_output --output_language en
```

Use consistently named columns in all inputs to standalone evaluation. For the documented real-reference protocol, supply distinct training and held-out tables explicitly. See [evaluation and data roles](docs/evaluation.md).

## Layout

- `generators/`, `config/`: model implementations and settings.
- `constraints/`, `grouping/`: clinical rules and stratified generation.
- `evaluation/`: fidelity, utility and empirical privacy evaluation.
- `utils/`: input validation, column preparation and split/outcome management.
- `models/`: pretrained best checkpoints and their manifest.
- Root Python files: command-line entry points, shared generation and output localization.
- Root JSON files: executable configurations and terminology mapping.

## Licence

Original code, supplied model checkpoints and code documentation: MIT. See [LICENSE.txt](LICENSE.txt). Third-party dependencies retain their own licences. The separately supplied data record uses CC BY 4.0, with its two Python configuration snapshots under MIT.
