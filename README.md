# OpenHSR TabPFN Reproducibility Code

This repository contains code and released outputs for Health Star Rating (HSR)
prediction experiments on tabular food product data. It includes OpenHSR-only
experiments, an OpenHSR split-sensitivity audit, and a compact Open Food Facts
(OFF) prediction table keyed by product code.

The repository is organised around three workflows:

1. **OpenHSR-only TabPFN-3 evaluation**
   - Repeated random 50-product held-out splits.
   - Nested greedy feature selection using only OpenHSR fields.
   - TabPFN-3 regression on the held-out OpenHSR products.

2. **OpenHSR baseline split-sensitivity audit**
   - Reimplementation of the executable classical-model section of the OpenHSR
     notebook.
   - Repeated random 80/20 splits to quantify how much the small OpenHSR test
     set affects reported performance.

3. **Open Food Facts HSR prediction table**
   - A compact table of model-derived HSR predictions for OFF products.
   - Each row is keyed by OFF product `code`, so product metadata can be joined
     from the official OFF export.
   - Release files and checksums are in
     `release/openfoodfacts_hsr_assignment_public/`.

The OpenHSR workflows can be rerun with public data. The OFF prediction table is
provided as a released output; the labelled context data used for the manuscript
run are not part of this repository.

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

TabPFN-3 model weights and access tokens are not included. To rerun TabPFN-based
analyses, install the `tabpfn` package, obtain access from Prior Labs, and follow
their licence terms.

For local interactive use, the package can open a browser on first use so you
can log in and accept the licence. On headless servers, create a Prior Labs API
key and set it before running TabPFN scripts:

```bash
export TABPFN_TOKEN="your_prior_labs_token"
```

On Windows PowerShell:

```powershell
$env:TABPFN_TOKEN = "your_prior_labs_token"
```

Do not commit access tokens, downloaded model weights, or local cache directories.

## Data

The OpenHSR experiments use the public OpenHSR dataset:

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

## Open Food Facts HSR Predictions

The compact OFF prediction file is tracked in this repository:

```text
release/openfoodfacts_hsr_assignment_public/openfoodfacts_hsr_assignments_public.csv.gz
```

It contains 4,532,767 rows. The join key is Open Food Facts product `code`.
The table contains predicted HSR values from two context settings, mapped
nutrient counts, and agreement fields comparing the two predictions. Product
names, brands, categories, and nutrient values can be joined from the official
Open Food Facts export when needed.

To rebuild the compact file from the full local/cloud prediction outputs:

```bash
python scripts/build_public_off_prediction_release.py
```

The file checksum is recorded in
`release/openfoodfacts_hsr_assignment_public/SHA256SUMS.txt`.

## Reproducibility Notes

- The TabPFN OpenHSR script performs feature selection inside each outer split.
  Held-out test labels are never used for feature selection, text encoding, or
  model context.
- The baseline audit follows the executable preprocessing pattern in the OpenHSR
  notebook: one-hot encode `category`, select numeric and boolean columns, scale
  train/test using a train-fitted `StandardScaler`, then fit each model on the
  training fold.
- The upstream notebook variable named `XGBR` is implemented as scikit-learn
  `GradientBoostingRegressor`, not the external `xgboost` package.
- OFF `assigned_hsr` values are model-derived predictions, not official Open
  Food Facts labels and not direct HSR-calculator outputs.

## Licences and Attribution

OpenHSR data should be cited and used according to the OpenHSR dataset licence.
Open Food Facts product data should be accessed from Open Food Facts and reused
under its database licence and attribution requirements. TabPFN-3 should be used
according to the TabPFN licence.
