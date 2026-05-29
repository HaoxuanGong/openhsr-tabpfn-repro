# OpenHSR TabPFN Reproducibility Code

This repository contains public-data-only code for reproducing OpenHSR-based
Health Star Rating (HSR) experiments. It is designed to accompany a manuscript
without releasing or depending on any proprietary product records.

The code supports two analyses:

1. **OpenHSR-only TabPFN-3 evaluation**
   - Repeated random 50-product held-out splits.
   - Nested greedy feature selection using only OpenHSR fields.
   - TabPFN-3 regressor evaluated on the held-out OpenHSR products.

2. **OpenHSR baseline split-sensitivity audit**
   - Reimplementation of the executable classical-model section of the OpenHSR
     notebook.
   - Repeated random 80/20 splits to quantify variability from the small
     OpenHSR test set.

No proprietary dataset paths, features, labels, or product records are required.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

On Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## TabPFN-3 Model Access

This repository does not distribute TabPFN-3 model weights or any model-access
token. Users must obtain their own TabPFN-3 access from Prior Labs and comply
with the TABPFN-3 licence.

For local interactive use, the `tabpfn` package can open a browser on first use
so the user can log in, accept the licence, and cache the authentication token
locally. For headless servers or notebooks, create a Prior Labs API key after
accepting the licence, then set it as an environment variable before running the
TabPFN script:

```bash
export TABPFN_TOKEN="your_prior_labs_token"
```

On Windows PowerShell:

```powershell
$env:TABPFN_TOKEN = "your_prior_labs_token"
```

For offline or restricted-compute environments, manually download the model
weights through the Prior Labs interface and set:

```bash
export TABPFN_MODEL_CACHE_DIR=/path/to/tabpfn-weights
```

Do not commit `TABPFN_TOKEN`, downloaded model weights, or local cache
directories to this repository.

## Data

The experiments use the public OpenHSR dataset:

> N'kam Suguem, F., & Lafargue, V. (2025). Open Health Star Rating (OpenHSR)
> (Version v1) [Data set]. Zenodo. https://doi.org/10.5281/zenodo.17469191

Download the dataset into `data/OpenHSR.csv`:

```bash
python scripts/download_openhsr.py --output data/OpenHSR.csv
```

If the download URL changes, manually download `OpenHSR.csv` from the OpenHSR
Zenodo or GitHub repository and place it at `data/OpenHSR.csv`.

## Run OpenHSR-only TabPFN-3

```bash
python scripts/run_tabpfn_openhsr_random50_greedy.py --openhsr-data data/OpenHSR.csv --output-dir results/openhsr_tabpfn_random50_greedy --n-splits 200 --test-size 50 --validation-size 50 --selection-metric raw_mse --device auto
```

Key outputs:

- `results/openhsr_tabpfn_random50_greedy/summary.csv`
- `results/openhsr_tabpfn_random50_greedy/metrics_by_split.csv`
- `results/openhsr_tabpfn_random50_greedy/selected_feature_sets_by_split.csv`
- `results/openhsr_tabpfn_random50_greedy/predictions_all_splits.csv`
- `results/openhsr_tabpfn_random50_greedy/paper_table_rows.tex`

For a quick smoke test:

```bash
python scripts/run_tabpfn_openhsr_random50_greedy.py --openhsr-data data/OpenHSR.csv --output-dir results/smoke_tabpfn --n-splits 2 --test-size 50 --validation-size 50 --max-greedy-steps 2 --device cpu --tabpfn-n-estimators 2
```

## Run OpenHSR Baseline Audit

```bash
python scripts/audit_openhsr_baselines_random_splits.py --openhsr-csv data/OpenHSR.csv --output-dir results/openhsr_baseline_random_split_audit --n-splits 200
```

To also retrain the released MLP architecture:

```bash
python scripts/audit_openhsr_baselines_random_splits.py --openhsr-csv data/OpenHSR.csv --output-dir results/openhsr_baseline_random_split_audit_mlp --n-splits 200 --include-mlp
```

Key outputs:

- `results/openhsr_baseline_random_split_audit/metrics_by_split.csv`
- `results/openhsr_baseline_random_split_audit/summary_by_model.csv`
- `results/openhsr_baseline_random_split_audit/reported_seed_metrics.csv`
- `results/openhsr_baseline_random_split_audit/reported_table_comparison.csv`
- `results/openhsr_baseline_random_split_audit/openhsr_random_split_audit.pdf`

## Reproducibility Notes

- The TabPFN script performs feature selection inside each outer split. Held-out
  test labels are never used for feature selection, text encoding, or context.
- The baseline audit follows the executable preprocessing pattern in the
  OpenHSR notebook: one-hot encode `category`, select numeric and boolean
  columns, scale train/test using a train-fitted `StandardScaler`, then fit each
  model on the training fold.
- The upstream notebook variable named `XGBR` is implemented as scikit-learn
  `GradientBoostingRegressor`, not the external `xgboost` package.

## Licences and Attribution

This repository contains only code and documentation for OpenHSR-based
reproducibility. OpenHSR data should be cited and used according to the OpenHSR
dataset licence. TabPFN-3 should be used according to the TabPFN licence.
