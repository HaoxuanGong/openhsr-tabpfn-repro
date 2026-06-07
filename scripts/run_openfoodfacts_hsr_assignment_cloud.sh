#!/usr/bin/env bash
# ============================================================
# Experiment: Open Food Facts HSR assignment cloud runner
# Paper Section / Research Question:
#   Generates large-scale model-derived HSR assignments for Open Food Facts
#   using all proprietary HSR-labelled context rows, and all proprietary plus
#   all OpenHSR labelled context rows.
#
# Purpose:
#   This runner installs the required Python environment when requested and
#   executes the full OFF assignment script on a cloud GPU server. It is designed
#   for a high-memory CUDA server, including Nvidia Blackwell systems with large
#   VRAM.
#
# Dataset(s):
#   - Source files:
#       DATA_CSV: proprietary/local HSR-labelled context data
#       OPENHSR_CSV: OpenHSR labelled context data
#       OFF_DATA: Open Food Facts product export
#   - Inclusion criteria:
#       Products readable by the OFF assignment script after field mapping.
#   - Exclusion criteria:
#       Controlled by script flags such as REQUIRE_PRODUCT_NAME and
#       MIN_NONMISSING_NUTRIENTS.
#   - Target variable:
#       Model-derived Health Star Rating pseudo-label.
#
# HSR Assumptions:
#   - HSR algorithm version:
#       Learned from labelled context rows; this runner does not implement the
#       rule calculator directly.
#   - Treatment of missing nutrition fields:
#       Controlled by the assignment script feature mapping and quality labels.
#
# Model / Method:
#   - Model type:
#       TabPFN regressor with optional multi-seed aggregation.
#   - Feature set:
#       Nutrient fields, mapped categorical/text fields, and optional text-SVD
#       features as configured below.
#   - Random seed(s):
#       SEEDS environment variable, default 42,101,202,303,404.
#
# Hardware Used:
#   - GPU:
#       CUDA-capable GPU recommended; developed for large-memory Nvidia server.
#
# Software Environment:
#   - Environment file:
#       requirements-model-benchmark.txt
#
# Outputs:
#   - Tables:
#       OUTPUT_DIR/predictions/*.csv.gz and summary CSV files.
#
# Reproducibility Notes:
#   This public repository does not ship proprietary labelled data or TabPFN
#   weights/tokens. The full assignment script may also be supplied from the
#   private manuscript workspace or cloud bundle. Run from repository/bundle root:
#     INSTALL_DEPS=1 OFF_DATA=/path/to/openfoodfacts-products.csv.gz bash scripts/run_openfoodfacts_hsr_assignment_cloud.sh
#
# Limitations:
#   assigned_hsr values are model-derived estimates and should be described as
#   pseudo-labels unless separately validated/calculated.
# ============================================================

set -euo pipefail

if [[ -z "${PYTHON_BIN:-}" ]]; then
  if command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN="$(command -v python3)"
  elif command -v python >/dev/null 2>&1; then
    PYTHON_BIN="$(command -v python)"
  else
    echo "Could not find python3 or python on PATH."
    echo "Install Python, activate your environment, or rerun with PYTHON_BIN=/path/to/python."
    exit 127
  fi
fi

VENV_DIR="${VENV_DIR:-.venv}"
USE_VENV="${USE_VENV:-auto}"

if [[ "${INSTALL_DEPS:-0}" == "1" && -z "${VIRTUAL_ENV:-}" && "${USE_VENV}" != "0" && "${USE_VENV}" != "false" && "${USE_VENV}" != "False" ]]; then
  if [[ ! -x "${VENV_DIR}/bin/python" ]]; then
    echo "Creating local Python virtual environment at ${VENV_DIR}"
    if ! "${PYTHON_BIN}" -m venv "${VENV_DIR}"; then
      echo "Could not create a virtual environment."
      echo "On Debian/Ubuntu, install venv support first, for example:"
      echo "  sudo apt install python3-venv"
      echo "or ask the cluster admin to provide a Python module/conda environment."
      exit 1
    fi
  fi
  PYTHON_BIN="${VENV_DIR}/bin/python"
fi

if [[ "${INSTALL_DEPS:-0}" == "1" ]]; then
  "${PYTHON_BIN}" -m pip install -r requirements-model-benchmark.txt
fi

export TABPFN_NO_BROWSER="${TABPFN_NO_BROWSER:-1}"

ASSIGN_SCRIPT="${ASSIGN_SCRIPT:-scripts/assign_openfoodfacts_hsr_with_full_contexts.py}"
DATA_CSV="${DATA_CSV:-data/WW_Readable_Parent_2025-11_hsr_clean_model_features.csv}"
OPENHSR_CSV="${OPENHSR_CSV:-data/OpenHSR.csv}"
OFF_DATA="${OFF_DATA:-data/OpenFoodFacts_clean_sample.csv}"
OFF_DOWNLOAD_URL="${OFF_DOWNLOAD_URL:-https://static.openfoodfacts.org/data/en.openfoodfacts.org.products.csv.gz}"
OFF_USER_AGENT="${OFF_USER_AGENT:-HSR-Prediction/1.0 research-bulk-download}"
DOWNLOAD_OFF="${DOWNLOAD_OFF:-auto}"

if [[ ! -f "${ASSIGN_SCRIPT}" ]]; then
  echo "Missing assignment script: ${ASSIGN_SCRIPT}"
  echo "This public repo intentionally does not include private labelled data."
  echo "If you are reproducing the manuscript run, copy the full assignment script and helper modules from the private workspace/cloud bundle, or set ASSIGN_SCRIPT=/path/to/script."
  exit 1
fi

if [[ ! -f "${DATA_CSV}" ]]; then
  echo "Missing DATA_CSV: ${DATA_CSV}"
  exit 1
fi

if [[ ! -f "${OPENHSR_CSV}" ]]; then
  echo "Missing OPENHSR_CSV: ${OPENHSR_CSV}"
  exit 1
fi

if [[ ! -f "${OFF_DATA}" ]]; then
  if [[ "${DOWNLOAD_OFF}" == "0" || "${DOWNLOAD_OFF}" == "false" || "${DOWNLOAD_OFF}" == "False" ]]; then
    echo "Missing OFF_DATA: ${OFF_DATA}"
    echo "Set OFF_DATA to the full Open Food Facts CSV/TSV/CSV.GZ export path, or allow download with DOWNLOAD_OFF=auto."
    exit 1
  fi
  echo "OFF_DATA not found: ${OFF_DATA}"
  echo "Downloading Open Food Facts products export from: ${OFF_DOWNLOAD_URL}"
  mkdir -p "$(dirname "${OFF_DATA}")"
  OFF_PARTIAL="${OFF_DATA}.part"
  if command -v curl >/dev/null 2>&1; then
    curl -L --fail --retry 5 --retry-delay 10 --connect-timeout 30 \
      -A "${OFF_USER_AGENT}" \
      -o "${OFF_PARTIAL}" \
      "${OFF_DOWNLOAD_URL}"
  elif command -v wget >/dev/null 2>&1; then
    wget --tries=5 --waitretry=10 \
      --user-agent="${OFF_USER_AGENT}" \
      -O "${OFF_PARTIAL}" \
      "${OFF_DOWNLOAD_URL}"
  else
    echo "Could not find curl or wget to download Open Food Facts."
    echo "Install curl/wget, manually place the export at ${OFF_DATA}, or set OFF_DATA to an existing file."
    exit 127
  fi
  mv "${OFF_PARTIAL}" "${OFF_DATA}"
  echo "Downloaded Open Food Facts export to: ${OFF_DATA}"
fi

COMMAND=(
  "${PYTHON_BIN}" "${ASSIGN_SCRIPT}"
  --our-data "${DATA_CSV}"
  --openhsr-data "${OPENHSR_CSV}"
  --off-data "${OFF_DATA}"
  --output-dir "${OUTPUT_DIR:-results/openfoodfacts_hsr_assignment_full_contexts}"
  --contexts "${CONTEXTS:-our_data_context,our_data_plus_openhsr_context}"
  --target-column "${TARGET_COLUMN:-hsr}"
  --seeds "${SEEDS:-42,101,202,303,404}"
  --device "${DEVICE:-cuda}"
  --tabpfn-model-path "${TABPFN_MODEL_PATH:-auto}"
  --tabpfn-n-estimators "${TABPFN_N_ESTIMATORS:-8}"
  --tabpfn-n-preprocessing-jobs "${TABPFN_N_PREPROCESSING_JOBS:-1}"
  --prediction-batch-size "${PREDICTION_BATCH_SIZE:-4096}"
  --text-svd-components "${TEXT_SVD_COMPONENTS:-16}"
  --text-max-features "${TEXT_MAX_FEATURES:-2500}"
  --min-nonmissing-nutrients "${MIN_NONMISSING_NUTRIENTS:-0}"
  --low-feature-count-threshold "${LOW_FEATURE_COUNT_THRESHOLD:-4}"
  --max-off-rows "${MAX_OFF_ROWS:-0}"
  --off-read-chunksize "${OFF_READ_CHUNKSIZE:-50000}"
  --off-on-bad-lines "${OFF_ON_BAD_LINES:-skip}"
  --off-parser-engine "${OFF_PARSER_ENGINE:-c}"
  --output-compression "${OUTPUT_COMPRESSION:-gzip}"
  --aggregation-chunksize "${AGGREGATION_CHUNKSIZE:-100000}"
  --assignment-mode "${ASSIGNMENT_MODE:-rounded_mode}"
  --max-prediction-sd "${MAX_PREDICTION_SD:-0.20}"
  --min-rounded-agreement "${MIN_ROUNDED_AGREEMENT:-0.80}"
)

if [[ "${NUMERIC_ONLY:-0}" == "1" ]]; then
  COMMAND+=(--numeric-only)
fi

if [[ "${REQUIRE_PRODUCT_NAME:-0}" == "1" ]]; then
  COMMAND+=(--require-product-name)
fi

if [[ "${KEEP_SEED_PREDICTIONS:-0}" == "1" ]]; then
  COMMAND+=(--keep-seed-predictions)
fi

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  COMMAND+=(--dry-run)
fi

printf 'Running command:'
printf ' %q' "${COMMAND[@]}"
printf '\n'

"${COMMAND[@]}"
