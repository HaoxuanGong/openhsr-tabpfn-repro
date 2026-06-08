#!/usr/bin/env bash
# ============================================================
# Experiment: Upload public Open Food Facts HSR assignment release
# Paper Section / Research Question:
#   Makes the compact Open Food Facts HSR pseudo-label outputs publicly
#   accessible without requiring readers to rerun the assignment experiment.
#
# Purpose:
#   Uploads the compact public OFF HSR assignment CSV.GZ and metadata files as
#   GitHub release assets. This keeps the repository clone lightweight while
#   giving readers direct access to the prediction results.
#
# Dataset(s):
#   - Source files:
#       release/openfoodfacts_hsr_assignment_public/
#       openfoodfacts_hsr_assignments_public_seed42.csv.gz
#       release/openfoodfacts_hsr_assignment_public/metadata.json
#       release/openfoodfacts_hsr_assignment_public/SHA256SUMS.txt
#       release/openfoodfacts_hsr_assignment_public/README.md
#   - Retailer(s):
#       Open Food Facts product identifiers; private/proprietary context data
#       are not uploaded.
#   - Inclusion criteria:
#       Rows included in the compact public release file.
#   - Exclusion criteria:
#       Not modified by this upload script.
#   - Target variable:
#       Model-derived Health Star Rating pseudo-labels.
#   - Unit convention:
#       Not applicable to this upload script.
#
# HSR Assumptions:
#   - HSR algorithm version:
#       Inherited from the upstream OFF assignment experiment.
#   - Category mapping:
#       Inherited from the upstream OFF assignment experiment.
#   - Treatment of ambiguous categories:
#       Inherited from the upstream OFF assignment experiment.
#   - Treatment of missing nutrition fields:
#       Captured in mapped_nutrient_count fields.
#   - Treatment of ineligible products:
#       Values are pseudo-labels unless independently validated.
#
# Model / Method:
#   - Model type:
#       No model is fitted here.
#   - Feature set:
#       No model features are used by this upload script.
#   - Preprocessing:
#       None.
#   - Train/validation/test split:
#       Not applicable.
#   - Random seed(s):
#       Inherited from the compact release file, normally seed 42.
#
# Hyperparameters:
#   - GitHub repository:
#       GITHUB_REPO, default HaoxuanGong/openhsr-tabpfn-repro.
#   - Release tag:
#       RELEASE_TAG, default off-hsr-assignments-seed42.
#   - Release directory:
#       RELEASE_DIR, default release/openfoodfacts_hsr_assignment_public.
#
# Evaluation:
#   - Metrics:
#       None. The script checks required file presence before upload.
#   - Error tolerance:
#       Not applicable.
#   - Subgroup analyses:
#       Not applicable.
#   - Robustness tests:
#       Not applicable.
#
# Hardware Used:
#   - CPU:
#       Any.
#   - GPU:
#       None.
#   - RAM:
#       Minimal.
#   - Storage:
#       Requires the compact release files.
#   - OS:
#       Linux/macOS bash, or Windows Git Bash/WSL.
#
# Compute Required to Reproduce:
#   - Expected wall-clock time:
#       Minutes, depending on network.
#   - Number of runs:
#       One.
#   - Peak RAM:
#       Minimal.
#   - Peak GPU memory:
#       N/A.
#   - Required disk space:
#       Existing release files plus no substantial temporary storage.
#   - Parallelism:
#       Single process.
#
# Software Environment:
#   - Python/R version:
#       Not required.
#   - Key packages:
#       GitHub CLI gh.
#   - Environment file:
#       None.
#
# Outputs:
#   - Tables:
#       GitHub release asset uploads.
#   - Figures:
#       None.
#   - Models:
#       None.
#   - Logs:
#       Terminal output from gh.
#
# Reproducibility Notes:
#   Authenticate before running with either `gh auth login` or a GH_TOKEN with
#   permission to create releases and upload assets for the target repository.
#
# Limitations:
#   This script does not validate the scientific correctness of the predictions;
#   it only publishes already-generated compact outputs.
# ============================================================

set -euo pipefail

GITHUB_REPO="${GITHUB_REPO:-HaoxuanGong/openhsr-tabpfn-repro}"
RELEASE_TAG="${RELEASE_TAG:-off-hsr-assignments-seed42}"
RELEASE_TITLE="${RELEASE_TITLE:-Open Food Facts HSR predictions, seed 42}"
RELEASE_DIR="${RELEASE_DIR:-release/openfoodfacts_hsr_assignment_public}"
PREDICTION_FILE="${PREDICTION_FILE:-${RELEASE_DIR}/openfoodfacts_hsr_assignments_public_seed42.csv.gz}"

if ! command -v gh >/dev/null 2>&1; then
  echo "Missing GitHub CLI: gh"
  echo "Install gh and authenticate with gh auth login, or set GH_TOKEN."
  exit 127
fi

required_files=(
  "${PREDICTION_FILE}"
  "${RELEASE_DIR}/README.md"
  "${RELEASE_DIR}/metadata.json"
  "${RELEASE_DIR}/SHA256SUMS.txt"
)

for file in "${required_files[@]}"; do
  if [[ ! -f "${file}" ]]; then
    echo "Missing required release file: ${file}"
    exit 1
  fi
done

if ! gh release view "${RELEASE_TAG}" --repo "${GITHUB_REPO}" >/dev/null 2>&1; then
  gh release create "${RELEASE_TAG}" \
    --repo "${GITHUB_REPO}" \
    --title "${RELEASE_TITLE}" \
    --notes-file "${RELEASE_DIR}/README.md"
else
  gh release edit "${RELEASE_TAG}" \
    --repo "${GITHUB_REPO}" \
    --title "${RELEASE_TITLE}" \
    --notes-file "${RELEASE_DIR}/README.md"
fi

gh release upload "${RELEASE_TAG}" \
  --repo "${GITHUB_REPO}" \
  --clobber \
  "${PREDICTION_FILE}" \
  "${RELEASE_DIR}/README.md" \
  "${RELEASE_DIR}/metadata.json" \
  "${RELEASE_DIR}/SHA256SUMS.txt"

echo "Uploaded public OFF HSR assignment release assets to ${GITHUB_REPO}:${RELEASE_TAG}"
