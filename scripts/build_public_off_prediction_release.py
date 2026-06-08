# ============================================================
# Experiment: Public Open Food Facts HSR assignment release builder
# Paper Section / Research Question:
#   Supports publication of large-scale model-derived Health Star Rating
#   assignments for Open Food Facts without requiring readers to rerun TabPFN.
#
# Purpose:
#   Converts the full OFF prediction outputs into a compact public artifact that
#   contains product identifiers, assigned HSR pseudo-labels, mapped nutrient
#   counts, context-comparison fields, checksums, and release documentation.
#
# Dataset(s):
#   - Source files:
#       cloud_run/openfoodfacts_hsr_assignment_bundle/results/
#       openfoodfacts_hsr_assignment_full_contexts/predictions/
#       openfoodfacts_hsr_assignments_our_data_context.csv.gz
#       cloud_run/openfoodfacts_hsr_assignment_bundle/results/
#       openfoodfacts_hsr_assignment_full_contexts/predictions/
#       openfoodfacts_hsr_assignments_our_data_plus_openhsr_context.csv.gz
#   - Retailer(s):
#       Open Food Facts for products; proprietary/private and OpenHSR rows are
#       used only as labelled context in the upstream prediction experiment.
#   - Inclusion criteria:
#       Rows present in both final OFF assignment files.
#   - Exclusion criteria:
#       None in this release builder; rows are inherited from the upstream OFF
#       assignment outputs.
#   - Target variable:
#       Model-derived HSR pseudo-label, rounded to valid 0.5-star increments.
#   - Unit convention:
#       OFF nutrient fields used upstream are per 100 g or per 100 mL according
#       to OFF conventions and upstream mapping.
#
# HSR Assumptions:
#   - HSR algorithm version:
#       Learned from the upstream labelled contexts; this builder does not run
#       the HSR rule calculator.
#   - Category mapping:
#       Inherited from the upstream OFF assignment script.
#   - Treatment of ambiguous categories:
#       Inherited from the upstream OFF assignment script and not reprocessed.
#   - Treatment of missing nutrition fields:
#       Inherited from upstream predictions; this builder preserves mapped
#       nutrient counts but does not publish upstream row flags.
#   - Treatment of ineligible products:
#       Inherited from upstream predictions; assigned values should be described
#       as pseudo-labels unless independently verified.
#
# Model / Method:
#   - Model type:
#       No model is fitted here. The script post-processes final TabPFN-based
#       OFF assignment outputs.
#   - Feature set:
#       Release fields only: OFF row id, OFF product code, assigned HSRs,
#       mapped nutrient counts, and context-agreement fields.
#   - Preprocessing:
#       Chunked CSV reading, row-order validation, and compact CSV.GZ writing.
#   - Train/validation/test split:
#       Not applicable; this is a release-formatting script.
#   - Random seed(s):
#       Inherited from upstream OFF assignment outputs, typically seed 42 for
#       the current public release.
#
# Hyperparameters:
#   - chunksize: default 200000 rows.
#   - compression: gzip output.
#   - context labels:
#       our_data_context and our_data_plus_openhsr_context.
#
# Evaluation:
#   - Metrics:
#       Row counts, missing-code counts, HSR distributions, context exact
#       agreement, and mean absolute context difference.
#   - Error tolerance:
#       Not applicable in this post-processing script.
#   - Subgroup analyses:
#       None.
#   - Robustness tests:
#       Row alignment is validated between the two context outputs.
#
# Hardware Used:
#   - CPU:
#       Any modern CPU.
#   - GPU:
#       None.
#   - RAM:
#       Designed for chunked processing; expected peak below 4 GB.
#   - Storage:
#       Requires both full prediction CSV.GZ inputs and enough space for the
#       compact public output.
#   - OS:
#       Windows, macOS, or Linux with Python.
#
# Compute Required to Reproduce:
#   - Expected wall-clock time:
#       Minutes for roughly 4.5 million OFF rows.
#   - Number of runs:
#       One.
#   - Peak RAM:
#       Estimated below 4 GB with default chunksize.
#   - Peak GPU memory:
#       N/A.
#   - Required disk space:
#       Source predictions plus generated compact release output.
#   - Parallelism:
#       Single process.
#
# Software Environment:
#   - Python/R version:
#       Python 3.10+ recommended.
#   - Key packages:
#       pandas.
#   - Environment file:
#       requirements-model-benchmark.txt.
#
# Outputs:
#   - Tables:
#       release/openfoodfacts_hsr_assignment_public/
#       openfoodfacts_hsr_assignments_public_seed42.csv.gz
#   - Figures:
#       None.
#   - Models:
#       None.
#   - Logs:
#       release/openfoodfacts_hsr_assignment_public/metadata.json
#       release/openfoodfacts_hsr_assignment_public/SHA256SUMS.txt
#       release/openfoodfacts_hsr_assignment_public/README.md
#
# Reproducibility Notes:
#   The compact release intentionally omits product names, brands, categories,
#   and nutrient values. Users can join by OFF product code against the official
#   Open Food Facts export if they need product metadata.
#
# Limitations:
#   The assigned HSR values are model-derived pseudo-labels, not official OFF
#   labels or direct HSR-calculator outputs.
# ============================================================

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from itertools import zip_longest
from pathlib import Path
from typing import Iterable

import pandas as pd


DEFAULT_RESULTS_DIR = Path(
    "cloud_run/openfoodfacts_hsr_assignment_bundle/results/"
    "openfoodfacts_hsr_assignment_full_contexts"
)
DEFAULT_OUR_CONTEXT = (
    DEFAULT_RESULTS_DIR
    / "predictions/openfoodfacts_hsr_assignments_our_data_context.csv.gz"
)
DEFAULT_PLUS_OPENHSR_CONTEXT = (
    DEFAULT_RESULTS_DIR
    / "predictions/"
    "openfoodfacts_hsr_assignments_our_data_plus_openhsr_context.csv.gz"
)
DEFAULT_OUTPUT_DIR = Path("release/openfoodfacts_hsr_assignment_public")
DEFAULT_OUTPUT_NAME = "openfoodfacts_hsr_assignments_public_seed42.csv.gz"

USECOLS = [
    "off_source_row_id",
    "code",
    "mapped_nutrient_count",
    "assigned_hsr",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a compact public release artifact from full Open Food Facts "
            "HSR assignment outputs."
        )
    )
    parser.add_argument("--our-context", type=Path, default=DEFAULT_OUR_CONTEXT)
    parser.add_argument(
        "--our-plus-openhsr-context",
        type=Path,
        default=DEFAULT_PLUS_OPENHSR_CONTEXT,
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--output-name", default=DEFAULT_OUTPUT_NAME)
    parser.add_argument("--chunksize", type=int, default=200_000)
    parser.add_argument(
        "--skip-source-hashes",
        action="store_true",
        help="Skip SHA256 hashes of large source prediction files.",
    )
    return parser.parse_args()


def sha256_file(path: Path, block_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(block_size), b""):
            digest.update(block)
    return digest.hexdigest()


def compact_counts(counter: Counter) -> dict[str, int]:
    return {str(key): int(counter[key]) for key in sorted(counter, key=str)}


def update_counter(counter: Counter, values: Iterable[object]) -> None:
    counter.update("__missing__" if pd.isna(value) else str(value) for value in values)


def validate_inputs(paths: Iterable[Path]) -> None:
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing input file(s): " + ", ".join(missing))


def build_release(args: argparse.Namespace) -> dict[str, object]:
    validate_inputs([args.our_context, args.our_plus_openhsr_context])
    args.output_dir.mkdir(parents=True, exist_ok=True)

    output_csv = args.output_dir / args.output_name
    metadata_json = args.output_dir / "metadata.json"
    readme_path = args.output_dir / "README.md"
    checksums_path = args.output_dir / "SHA256SUMS.txt"

    dtype = {
        "off_source_row_id": "Int64",
        "code": "string",
        "mapped_nutrient_count": "Int64",
        "assigned_hsr": "Float64",
    }

    our_reader = pd.read_csv(
        args.our_context,
        usecols=USECOLS,
        dtype=dtype,
        chunksize=args.chunksize,
    )
    plus_reader = pd.read_csv(
        args.our_plus_openhsr_context,
        usecols=USECOLS,
        dtype=dtype,
        chunksize=args.chunksize,
    )

    hsr_our = Counter()
    hsr_plus = Counter()
    rows = 0
    missing_code = 0
    context_same = 0
    context_abs_diff_sum = 0.0

    with gzip.open(output_csv, "wt", encoding="utf-8", newline="") as output_handle:
        for chunk_index, pair in enumerate(zip_longest(our_reader, plus_reader), start=1):
            our_chunk, plus_chunk = pair
            if our_chunk is None or plus_chunk is None:
                raise ValueError("Context files have different numbers of chunks.")
            if len(our_chunk) != len(plus_chunk):
                raise ValueError(
                    "Context files have different chunk lengths at chunk "
                    f"{chunk_index}: {len(our_chunk)} vs {len(plus_chunk)}."
                )

            our_ids = our_chunk["off_source_row_id"].reset_index(drop=True)
            plus_ids = plus_chunk["off_source_row_id"].reset_index(drop=True)
            if not our_ids.equals(plus_ids):
                raise ValueError(f"off_source_row_id mismatch at chunk {chunk_index}.")

            our_codes = our_chunk["code"].fillna("").reset_index(drop=True)
            plus_codes = plus_chunk["code"].fillna("").reset_index(drop=True)
            if not our_codes.equals(plus_codes):
                raise ValueError(f"OFF product code mismatch at chunk {chunk_index}.")

            assigned_our = our_chunk["assigned_hsr"].reset_index(drop=True)
            assigned_plus = plus_chunk["assigned_hsr"].reset_index(drop=True)
            abs_diff = (assigned_our - assigned_plus).abs()
            same = assigned_our.eq(assigned_plus)

            out = pd.DataFrame(
                {
                    "off_source_row_id": our_ids,
                    "code": our_codes.replace("", pd.NA),
                    "assigned_hsr_our_data_context": assigned_our,
                    "mapped_nutrient_count_our_data_context": our_chunk[
                        "mapped_nutrient_count"
                    ].reset_index(drop=True),
                    "assigned_hsr_our_data_plus_openhsr_context": assigned_plus,
                    "mapped_nutrient_count_our_data_plus_openhsr_context": plus_chunk[
                        "mapped_nutrient_count"
                    ].reset_index(drop=True),
                    "assigned_hsr_context_abs_diff": abs_diff,
                    "assigned_hsr_context_same": same,
                }
            )

            out.to_csv(output_handle, index=False, header=chunk_index == 1)

            rows += len(out)
            missing_code += int(out["code"].isna().sum())
            context_same += int(same.sum())
            context_abs_diff_sum += float(abs_diff.sum())
            update_counter(hsr_our, out["assigned_hsr_our_data_context"])
            update_counter(hsr_plus, out["assigned_hsr_our_data_plus_openhsr_context"])

    output_hash = sha256_file(output_csv)
    source_hashes = {}
    if not args.skip_source_hashes:
        source_hashes = {
            str(args.our_context): sha256_file(args.our_context),
            str(args.our_plus_openhsr_context): sha256_file(
                args.our_plus_openhsr_context
            ),
        }

    metadata = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_files": {
            "our_data_context": str(args.our_context),
            "our_data_plus_openhsr_context": str(args.our_plus_openhsr_context),
        },
        "source_sha256": source_hashes,
        "output_files": {
            "public_predictions": str(output_csv),
        },
        "output_sha256": {
            output_csv.name: output_hash,
        },
        "row_count": rows,
        "missing_code_count": missing_code,
        "context_exact_agreement_count": context_same,
        "context_exact_agreement_fraction": context_same / rows if rows else None,
        "context_mean_absolute_difference": (
            context_abs_diff_sum / rows if rows else None
        ),
        "assigned_hsr_counts": {
            "our_data_context": compact_counts(hsr_our),
            "our_data_plus_openhsr_context": compact_counts(hsr_plus),
        },
        "release_columns": [
            "off_source_row_id",
            "code",
            "assigned_hsr_our_data_context",
            "mapped_nutrient_count_our_data_context",
            "assigned_hsr_our_data_plus_openhsr_context",
            "mapped_nutrient_count_our_data_plus_openhsr_context",
            "assigned_hsr_context_abs_diff",
            "assigned_hsr_context_same",
        ],
        "interpretation": (
            "assigned_hsr columns are model-derived HSR pseudo-labels, not "
            "official Open Food Facts labels or direct HSR-calculator outputs."
        ),
    }
    metadata_json.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    checksums_path.write_text(
        "\n".join(
            [
                f"{output_hash}  {output_csv.name}",
                f"{sha256_file(metadata_json)}  {metadata_json.name}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    readme = f"""# Open Food Facts HSR Prediction Public Release

This folder documents the compact public output for the Open Food Facts HSR
prediction experiment. The main CSV.GZ may be tracked in this folder or attached
as a GitHub release asset, depending on repository-size policy. The full
internal prediction files include product names, brands, categories, and
nutrient fields; those fields are intentionally omitted from this public
release. Users can join by `code` against the official Open Food Facts export.

## Main File

- `{output_csv.name}`

Rows: `{rows:,}`
SHA256: `{output_hash}`

The assigned HSR columns are model-derived pseudo-labels, not official Open Food
Facts labels and not direct HSR-calculator outputs.

## Columns

- `off_source_row_id`: row index from the OFF export used in the assignment run.
- `code`: Open Food Facts product code/barcode.
- `assigned_hsr_our_data_context`: HSR assigned using the private labelled
  context data.
- `assigned_hsr_our_data_plus_openhsr_context`: HSR assigned using private
  labelled context data plus OpenHSR.
- `mapped_nutrient_count_*`: number of mapped nutrient fields available to the
  upstream prediction model.
- `assigned_hsr_context_abs_diff`: absolute difference between the two context
  assignments.
- `assigned_hsr_context_same`: whether the two context assignments are exactly
  equal.

## Summary

- Context exact agreement: `{context_same / rows:.3f}`
- Mean absolute context difference: `{context_abs_diff_sum / rows:.3f}`
- Missing product code rows: `{missing_code:,}`

## Licence And Attribution Notes

Open Food Facts product data should be accessed from Open Food Facts and reused
under its database licence and attribution requirements. This release provides
model-derived HSR pseudo-labels keyed by OFF product code rather than a full
redistribution of product metadata or nutrition data.
"""
    readme_path.write_text(readme, encoding="utf-8")

    return metadata


def main() -> None:
    args = parse_args()
    metadata = build_release(args)
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
