# ============================================================
# Experiment: Download OpenHSR public dataset
# Paper Section / Research Question:
#   Supports the OpenHSR-only reproducibility analyses and public-data release.
#
# Purpose:
#   Download the public OpenHSR CSV into this release repository so that all
#   analyses can be reproduced without using any non-public product records.
#
# Dataset(s):
#   - Source files:
#       OpenHSR.csv from the public OpenHSR GitHub repository.
#   - Retailer(s):
#       Public OpenHSR sources as provided by the OpenHSR authors.
#   - Inclusion criteria:
#       All rows present in the downloaded OpenHSR CSV.
#   - Exclusion criteria:
#       None in this downloader.
#   - Target variable:
#       HSR, as distributed by OpenHSR.
#   - Unit convention:
#       Uses the unit conventions already encoded in OpenHSR.
#
# HSR Assumptions:
#   - HSR algorithm version:
#       Not recalculated by this script.
#   - Category mapping:
#       Not modified by this script.
#   - Treatment of ambiguous categories:
#       Not modified by this script.
#   - Treatment of missing nutrition fields:
#       Not modified by this script.
#   - Treatment of ineligible products:
#       Not modified by this script.
#
# Model / Method:
#   - Model type:
#       None.
#   - Feature set:
#       None.
#   - Preprocessing:
#       None.
#   - Train/validation/test split:
#       None.
#   - Random seed(s):
#       None.
#
# Hyperparameters:
#   None.
#
# Evaluation:
#   - Metrics:
#       None.
#   - Error tolerance:
#       None.
#   - Subgroup analyses:
#       None.
#   - Robustness tests:
#       None.
#
# Hardware Used:
#   - CPU:
#       Any machine capable of downloading and storing the CSV.
#   - GPU:
#       None.
#   - RAM:
#       Minimal.
#   - Storage:
#       Requires space for the OpenHSR CSV.
#   - OS:
#       Cross-platform Python.
#
# Compute Required to Reproduce:
#   - Expected wall-clock time:
#       Seconds to minutes, depending on network speed.
#   - Number of runs:
#       One.
#   - Peak RAM:
#       Minimal.
#   - Peak GPU memory:
#       N/A.
#   - Required disk space:
#       Size of the downloaded OpenHSR CSV.
#   - Parallelism:
#       None.
#
# Software Environment:
#   - Python/R version:
#       Python 3.10+ recommended.
#   - Key packages:
#       Python standard library only.
#   - Environment file:
#       requirements.txt.
#
# Outputs:
#   - Tables:
#       data/OpenHSR.csv.
#   - Figures:
#       None.
#   - Models:
#       None.
#   - Logs:
#       Console output only.
#
# Reproducibility Notes:
#   - The public URL may change if the OpenHSR repository is reorganized.
#
# Limitations:
#   - This script does not validate the scientific contents of OpenHSR.
# ============================================================
from __future__ import annotations

import argparse
import urllib.error
import urllib.request
from pathlib import Path


URLS = [
    "https://raw.githubusercontent.com/ValentinLafargue/HealthStarDataset/main/Data/OpenHSR.csv",
    "https://raw.githubusercontent.com/ValentinLafargue/HealthStarDataset/master/Data/OpenHSR.csv",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download the public OpenHSR.csv file.")
    parser.add_argument("--output", type=Path, default=Path("data/OpenHSR.csv"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    last_error: Exception | None = None
    for url in URLS:
        try:
            print(f"Downloading {url}")
            urllib.request.urlretrieve(url, args.output)
            print(f"Wrote {args.output}")
            return
        except (urllib.error.URLError, urllib.error.HTTPError, OSError) as exc:
            last_error = exc
            print(f"Failed: {exc}")
    raise SystemExit(
        "Could not download OpenHSR.csv automatically. Download it manually from "
        "the OpenHSR GitHub or Zenodo page and place it at data/OpenHSR.csv. "
        f"Last error: {last_error}"
    )


if __name__ == "__main__":
    main()
