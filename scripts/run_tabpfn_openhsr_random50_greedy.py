# ============================================================
# Experiment: OpenHSR-only TabPFN-3 random-50 greedy evaluation
# Paper Section / Research Question:
#   Tests TabPFN-3 on OpenHSR when no external non-public product records are
#   used. Supports the OpenHSR reproducibility and portability analyses.
#
# Purpose:
#   Repeatedly split OpenHSR into labelled context, validation, and a held-out
#   test set of exactly 50 products. For each split, perform nested forward
#   greedy feature selection on the OpenHSR-only validation set and evaluate
#   TabPFN-3 on the held-out OpenHSR test products.
#
# Dataset(s):
#   - Source files:
#       data/OpenHSR.csv, downloaded from the public OpenHSR repository.
#   - Retailer(s):
#       Public OpenHSR sources as provided by the OpenHSR authors.
#   - Inclusion criteria:
#       Rows with a valid numeric HSR label.
#   - Exclusion criteria:
#       Rows without a valid HSR label.
#   - Target variable:
#       HSR.
#   - Unit convention:
#       Uses OpenHSR nutrient fields as distributed; no unit conversion is
#       performed by this script.
#
# HSR Assumptions:
#   - HSR algorithm version:
#       HSR labels are taken from OpenHSR and are not recalculated.
#   - Category mapping:
#       Uses OpenHSR category/product-type fields as candidate predictors.
#   - Treatment of ambiguous categories:
#       Retained as categorical values when present.
#   - Treatment of missing nutrition fields:
#       Numeric missing values are retained for TabPFN-3; categorical/text
#       missing values are encoded with explicit missing sentinels.
#   - Treatment of ineligible products:
#       Not independently reassessed; follows OpenHSR inclusion.
#
# Model / Method:
#   - Model type:
#       TabPFN-3 regressor.
#   - Feature set:
#       OpenHSR product name, ingredients, allergen/size text where available,
#       product type/category/source metadata, and nutrient fields available in
#       OpenHSR. The final set is selected separately inside each split.
#   - Preprocessing:
#       Numeric values are parsed as numeric; categorical values use missing
#       sentinels; text fields are encoded by TF-IDF unigrams/bigrams followed
#       by truncated SVD for mixed tabular input.
#   - Train/validation/test split:
#       Repeated random splits. Each split holds out exactly 50 products for
#       testing and 50 products for validation by default; remaining rows form
#       the TabPFN labelled context.
#   - Random seed(s):
#       Defaults to 0..n_splits-1 unless --split-seeds is supplied.
#
# Hyperparameters:
#   --n-splits 200; --test-size 50; --validation-size 50;
#   --selection-metric raw_mse; --min-improvement 0.0;
#   --max-greedy-steps 8; --tabpfn-n-estimators 8;
#   --text-max-features 2500; --text-svd-components 16.
#
# Evaluation:
#   - Metrics:
#       Raw MSE, raw MAE, RMSE, rounded exact half-star agreement, and rounded
#       within-0.5-star agreement.
#   - Error tolerance:
#       Exact rounded HSR and within one half-star step.
#   - Subgroup analyses:
#       None.
#   - Robustness tests:
#       Split-sensitivity over repeated random test sets.
#
# Hardware Used:
#   - CPU:
#       Cross-platform CPU for preprocessing.
#   - GPU:
#       CUDA GPU recommended for TabPFN-3, or CPU for slower execution.
#   - RAM:
#       Depends on selected TabPFN backend and number of estimators.
#   - Storage:
#       Requires OpenHSR CSV and result CSV files.
#   - OS:
#       Cross-platform Python; developed on Windows and Ubuntu-style paths.
#
# Compute Required to Reproduce:
#   - Expected wall-clock time:
#       Depends on GPU and TabPFN installation; 200 splits may take hours.
#   - Number of runs:
#       One run containing n_splits repeated random splits.
#   - Peak RAM:
#       Dataset is small; TabPFN runtime dominates.
#   - Peak GPU memory:
#       Depends on TabPFN backend and n_estimators.
#   - Required disk space:
#       Less than 1 GB for outputs.
#   - Parallelism:
#       Controlled by TabPFN and scikit-learn internals.
#
# Software Environment:
#   - Python/R version:
#       Python 3.10+ recommended.
#   - Key packages:
#       numpy, pandas, scikit-learn, tabpfn, torch.
#   - Environment file:
#       requirements.txt.
#
# Outputs:
#   - Tables:
#       metrics_by_split.csv, summary.csv, paper_table_rows.tex.
#   - Figures:
#       None.
#   - Models:
#       None; TabPFN-3 is used at inference time.
#   - Logs:
#       Console output and JSON configuration files.
#
# Reproducibility Notes:
#   - TabPFN model availability and version should be reported with results.
#   - Feature selection is nested inside each random split to avoid using test
#     labels during selection.
#
# Limitations:
#   - OpenHSR is small; random-split variability should be interpreted together
#     with mean and standard deviation across splits.
# ============================================================
from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from openhsr_repro import common


FEATURE_SET_NAME = "openhsr_only_greedy_random50"
ALL_FEATURES_NAME = "openhsr_only_all_features_random50"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="OpenHSR-only TabPFN-3 random-50 evaluation with nested greedy feature selection.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--openhsr-data", type=Path, default=Path("data/OpenHSR.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/openhsr_tabpfn_random50_greedy"))
    parser.add_argument("--target-column", default="HSR")
    parser.add_argument("--n-splits", type=int, default=200)
    parser.add_argument("--test-size", type=int, default=50)
    parser.add_argument("--validation-size", type=int, default=50)
    parser.add_argument("--split-seeds", default="")
    parser.add_argument("--stratify", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--selection-metric",
        choices=["raw_mse", "raw_mae", "rounded_exact_accuracy", "rounded_within_0_5_accuracy"],
        default="raw_mse",
    )
    parser.add_argument("--min-improvement", type=float, default=0.0)
    parser.add_argument("--max-greedy-steps", type=int, default=8)
    parser.add_argument("--evaluate-all-features-reference", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--tabpfn-model-path", default="auto")
    parser.add_argument("--tabpfn-n-estimators", type=int, default=8)
    parser.add_argument("--tabpfn-n-preprocessing-jobs", type=int, default=1)
    parser.add_argument("--prediction-batch-size", type=int, default=0)
    parser.add_argument("--text-svd-components", type=int, default=16)
    parser.add_argument("--text-max-features", type=int, default=2500)
    parser.add_argument("--include-date-collected", action="store_true")
    return parser.parse_args()


def parse_seeds(value: str, n_splits: int) -> list[int]:
    if value.strip():
        seeds = [int(part.strip()) for part in value.split(",") if part.strip()]
    else:
        seeds = list(range(n_splits))
    if len(seeds) < n_splits:
        raise ValueError(f"Requested {n_splits} splits but only {len(seeds)} seeds were supplied.")
    return seeds[:n_splits]


def clone_args(args: argparse.Namespace, seed: int) -> argparse.Namespace:
    cloned = copy.copy(args)
    cloned.random_seed = seed
    return cloned


def split_rows(
    df: pd.DataFrame,
    rows: np.ndarray,
    *,
    test_size: int,
    seed: int,
    target_column: str,
    stratify: bool,
) -> tuple[np.ndarray, np.ndarray, bool, str]:
    y = pd.to_numeric(df.loc[rows, target_column], errors="coerce").to_numpy(dtype=float)
    labels = common.round_to_half_star(y)
    note = ""
    if stratify:
        try:
            left, right = train_test_split(
                rows,
                test_size=test_size,
                random_state=seed,
                shuffle=True,
                stratify=labels,
            )
            return np.asarray(left, dtype=int), np.asarray(right, dtype=int), True, note
        except ValueError as exc:
            note = f"Stratified split failed; unstratified split used: {exc}"
    left, right = train_test_split(rows, test_size=test_size, random_state=seed, shuffle=True, stratify=None)
    return np.asarray(left, dtype=int), np.asarray(right, dtype=int), False, note


def metric_is_lower_better(metric: str) -> bool:
    return metric in {"raw_mse", "raw_mae"}


def metric_improved(candidate: float, current: float, metric: str, min_improvement: float) -> bool:
    if metric_is_lower_better(metric):
        return candidate < current - min_improvement
    return candidate > current + min_improvement


def run_tabpfn_batched(mixed: common.MixedFrames, y_train: np.ndarray, args: argparse.Namespace) -> np.ndarray:
    from tabpfn import TabPFNRegressor

    categorical_indices = [mixed.X_train.columns.get_loc(c) for c in mixed.categorical_names]
    reg = TabPFNRegressor(
        n_estimators=args.tabpfn_n_estimators,
        categorical_features_indices=categorical_indices,
        model_path=args.tabpfn_model_path,
        device=args.device,
        random_state=args.random_seed,
        n_preprocessing_jobs=args.tabpfn_n_preprocessing_jobs,
    )
    reg.fit(mixed.X_train, y_train)

    batch_size = int(args.prediction_batch_size or 0)
    if batch_size <= 0 or len(mixed.X_test) <= batch_size:
        return np.asarray(reg.predict(mixed.X_test), dtype=float)

    chunks = []
    for start in range(0, len(mixed.X_test), batch_size):
        end = min(start + batch_size, len(mixed.X_test))
        chunks.append(np.asarray(reg.predict(mixed.X_test.iloc[start:end]), dtype=float))
    return np.concatenate(chunks, axis=0)


def evaluate_units(
    df: pd.DataFrame,
    train_rows: np.ndarray,
    eval_rows: np.ndarray,
    selected_units: list[str],
    feature_units: dict[str, dict[str, Any]],
    args: argparse.Namespace,
    *,
    feature_set: str,
    split_index: int,
    split_seed: int,
    phase: str,
    extra: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], pd.DataFrame]:
    eval_args = clone_args(args, split_seed)
    y_train = pd.to_numeric(df.loc[train_rows, args.target_column], errors="coerce").to_numpy(dtype=float)
    y_eval = pd.to_numeric(df.loc[eval_rows, args.target_column], errors="coerce").to_numpy(dtype=float)
    mixed = common.build_mixed_frames(
        df,
        train_rows,
        eval_rows,
        selected_units,
        feature_units,
        text_svd_components=args.text_svd_components,
        text_max_features=args.text_max_features,
        random_seed=split_seed,
    )
    start = time.time()
    pred = run_tabpfn_batched(mixed, y_train, eval_args)
    elapsed = time.time() - start

    row = {
        "feature_set": feature_set,
        "model": "tabpfn_v3_regressor",
        "status": "completed",
        "split_index": int(split_index),
        "split_seed": int(split_seed),
        "seed": int(split_seed),
        "phase": phase,
        "train_rows": int(len(train_rows)),
        "test_rows": int(len(eval_rows)),
        "feature_count": int(len(mixed.feature_names)),
        "fit_predict_seconds": round(elapsed, 3),
        "selected_units": ",".join(selected_units),
        "text_source_columns": ",".join(mixed.text_report.get("source_columns", [])),
        "text_svd_components": int(mixed.text_report.get("svd_components", 0)),
    }
    row.update(common.regressor_metrics(y_eval, pred))
    if extra:
        row.update(extra)

    predictions = common.predictions_frame(eval_rows, y_eval, pred)
    predictions["split_index"] = int(split_index)
    predictions["split_seed"] = int(split_seed)
    predictions["feature_set"] = feature_set
    predictions["phase"] = phase
    predictions["product_name"] = df.loc[eval_rows, "product_name"].to_numpy() if "product_name" in df.columns else ""
    predictions["source_row_id"] = df.loc[eval_rows, "source_row_id"].to_numpy()
    return row, predictions


def run_greedy_selection(
    df: pd.DataFrame,
    greedy_train_rows: np.ndarray,
    greedy_val_rows: np.ndarray,
    feature_units: dict[str, dict[str, Any]],
    args: argparse.Namespace,
    *,
    split_index: int,
    split_seed: int,
) -> tuple[list[str], list[dict[str, Any]], list[dict[str, Any]]]:
    selected_units: list[str] = []
    remaining = list(feature_units)
    candidate_rows: list[dict[str, Any]] = []
    path_rows: list[dict[str, Any]] = []
    current_best = np.inf if metric_is_lower_better(args.selection_metric) else -np.inf
    max_steps = len(remaining) if args.max_greedy_steps < 0 else min(args.max_greedy_steps, len(remaining))

    for step in range(1, max_steps + 1):
        step_results: list[dict[str, Any]] = []
        for unit in remaining:
            trial_units = selected_units + [unit]
            try:
                row, _ = evaluate_units(
                    df,
                    greedy_train_rows,
                    greedy_val_rows,
                    trial_units,
                    feature_units,
                    args,
                    feature_set=f"{FEATURE_SET_NAME}_greedy_validation",
                    split_index=split_index,
                    split_seed=split_seed,
                    phase="greedy_validation",
                    extra={"greedy_step": step, "candidate_unit": unit, "trial_units": ",".join(trial_units)},
                )
                row["selection_error"] = ""
            except Exception as exc:
                row = {
                    "feature_set": f"{FEATURE_SET_NAME}_greedy_validation",
                    "model": "tabpfn_v3_regressor",
                    "status": "failed",
                    "split_index": int(split_index),
                    "split_seed": int(split_seed),
                    "phase": "greedy_validation",
                    "greedy_step": step,
                    "candidate_unit": unit,
                    "trial_units": ",".join(trial_units),
                    "selected_units": ",".join(selected_units),
                    "selection_error": f"{type(exc).__name__}: {exc}",
                }
            candidate_rows.append(row)
            if row.get("status") == "completed":
                step_results.append(row)

        if not step_results:
            break
        best_row = min(step_results, key=lambda item: float(item[args.selection_metric])) if metric_is_lower_better(args.selection_metric) else max(step_results, key=lambda item: float(item[args.selection_metric]))
        best_value = float(best_row[args.selection_metric])
        if not metric_improved(best_value, current_best, args.selection_metric, args.min_improvement):
            break
        accepted = str(best_row["candidate_unit"])
        selected_units.append(accepted)
        remaining.remove(accepted)
        current_best = best_value
        path_rows.append(
            {
                "split_index": int(split_index),
                "split_seed": int(split_seed),
                "greedy_step": step,
                "accepted_unit": accepted,
                "selection_metric": args.selection_metric,
                "selection_metric_value": best_value,
                "selected_units": ",".join(selected_units),
                "remaining_units": ",".join(remaining),
            }
        )
        print(f"[greedy split={split_index} step={step}] accepted={accepted} {args.selection_metric}={best_value:.6f}", flush=True)
        if not remaining:
            break
    return selected_units, candidate_rows, path_rows


def summarize_metrics(metrics: pd.DataFrame) -> pd.DataFrame:
    metric_cols = [
        "raw_mae",
        "raw_mse",
        "raw_rmse",
        "raw_r2",
        "rounded_exact_accuracy",
        "rounded_mae",
        "rounded_rmse",
        "rounded_within_0_5_accuracy",
        "rounded_within_1_0_accuracy",
        "fit_predict_seconds",
    ]
    rows: list[dict[str, Any]] = []
    for (feature_set, model), group in metrics.groupby(["feature_set", "model"], dropna=False):
        row: dict[str, Any] = {
            "feature_set": feature_set,
            "model": model,
            "n_splits": int(len(group)),
            "train_rows": int(group["train_rows"].iloc[0]),
            "test_rows": int(group["test_rows"].iloc[0]),
            "feature_count_mean": float(pd.to_numeric(group["feature_count"], errors="coerce").mean()),
            "feature_count_sd": float(pd.to_numeric(group["feature_count"], errors="coerce").std(ddof=1)),
        }
        for column in metric_cols:
            values = pd.to_numeric(group[column], errors="coerce")
            row[f"{column}_mean"] = float(values.mean()) if values.notna().any() else np.nan
            row[f"{column}_sd"] = float(values.std(ddof=1)) if values.notna().sum() > 1 else 0.0
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["feature_set", "model"]).reset_index(drop=True)


def format_mean_sd(mean: float, sd: float, scale: float = 1.0, digits: int = 3) -> str:
    if pd.isna(mean):
        return "--"
    return f"{mean * scale:.{digits}f} $\\pm$ {sd * scale:.{digits}f}"


def write_outputs(summary: pd.DataFrame, selected_by_split: pd.DataFrame, output_dir: Path, setup: dict[str, Any]) -> None:
    table_rows = []
    lines = ["# OpenHSR-only random-50 greedy TabPFN evaluation", ""]
    for _, row in summary.iterrows():
        label = str(row["feature_set"]).replace("_", " ")
        lines.append(f"## {label}")
        lines.append(f"- Splits: {int(row['n_splits'])}")
        lines.append(f"- Context rows per split: {int(row['train_rows'])}")
        lines.append(f"- Test rows per split: {int(row['test_rows'])}")
        lines.append(f"- Mean feature count: {row['feature_count_mean']:.1f} +/- {row['feature_count_sd']:.1f}")
        lines.append(f"- MSE: {row['raw_mse_mean']:.3f} +/- {row['raw_mse_sd']:.3f}")
        lines.append(f"- MAE: {row['raw_mae_mean']:.3f} +/- {row['raw_mae_sd']:.3f}")
        lines.append(f"- Exact agreement: {row['rounded_exact_accuracy_mean'] * 100:.1f}% +/- {row['rounded_exact_accuracy_sd'] * 100:.1f}%")
        lines.append(f"- Within 0.5-star agreement: {row['rounded_within_0_5_accuracy_mean'] * 100:.1f}% +/- {row['rounded_within_0_5_accuracy_sd'] * 100:.1f}%")
        lines.append("")
        table_rows.append(
            f"{label} & OpenHSR random splits (context \\(n={int(row['train_rows'])}\\), test \\(n={int(row['test_rows'])}\\)) & "
            f"{format_mean_sd(row['raw_mse_mean'], row['raw_mse_sd'], digits=3)} & "
            f"{format_mean_sd(row['rounded_exact_accuracy_mean'], row['rounded_exact_accuracy_sd'], scale=100, digits=1)} & "
            f"{format_mean_sd(row['rounded_within_0_5_accuracy_mean'], row['rounded_within_0_5_accuracy_sd'], scale=100, digits=1)} \\\\"
        )

    if not selected_by_split.empty:
        counts = Counter()
        for units in selected_by_split["selected_units"].fillna(""):
            for unit in str(units).split(","):
                unit = unit.strip()
                if unit:
                    counts[unit] += 1
        lines.append("## Selected feature frequency")
        for unit, count in counts.most_common():
            lines.append(f"- {unit}: {count}/{len(selected_by_split)} splits")
        lines.append("")

    lines.append("## Setup")
    for key, value in setup.items():
        if key != "split_seeds":
            lines.append(f"- {key}: {value}")
    (output_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (output_dir / "paper_table_rows.tex").write_text("\n".join(table_rows) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    started = time.time()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    df = common.load_openhsr(args.openhsr_data, args.target_column)
    feature_units = common.build_openhsr_feature_units(df, include_date_collected=args.include_date_collected)
    split_seeds = parse_seeds(args.split_seeds, args.n_splits)

    if len(df) <= args.test_size:
        raise ValueError(f"OpenHSR has {len(df)} labelled rows, not enough for test_size={args.test_size}.")
    if len(df) - args.test_size <= args.validation_size:
        raise ValueError(f"Context pool has {len(df) - args.test_size} rows, not enough for validation_size={args.validation_size}.")
    if not feature_units:
        raise ValueError("No usable OpenHSR feature units were found.")

    common.write_json(args.output_dir / "run_config.json", {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()})
    common.write_json(args.output_dir / "environment.json", common.environment_report())
    common.write_json(args.output_dir / "feature_space.json", {"candidate_units": feature_units, "target_column": args.target_column})

    metrics_rows: list[dict[str, Any]] = []
    candidate_rows: list[dict[str, Any]] = []
    path_rows: list[dict[str, Any]] = []
    selected_rows: list[dict[str, Any]] = []
    prediction_frames: list[pd.DataFrame] = []
    assignment_rows: list[dict[str, Any]] = []

    all_rows = np.arange(len(df), dtype=int)
    all_feature_units = list(feature_units)

    for split_index, split_seed in enumerate(split_seeds):
        context_pool_rows, test_rows, outer_stratified, outer_note = split_rows(
            df,
            all_rows,
            test_size=args.test_size,
            seed=split_seed,
            target_column=args.target_column,
            stratify=args.stratify,
        )
        greedy_train_rows, greedy_val_rows, inner_stratified, inner_note = split_rows(
            df,
            context_pool_rows,
            test_size=args.validation_size,
            seed=split_seed + 100000,
            target_column=args.target_column,
            stratify=args.stratify,
        )
        selected_units, split_candidate_rows, split_path_rows = run_greedy_selection(
            df,
            greedy_train_rows,
            greedy_val_rows,
            feature_units,
            args,
            split_index=split_index,
            split_seed=split_seed,
        )
        candidate_rows.extend(split_candidate_rows)
        path_rows.extend(split_path_rows)
        selected_rows.append(
            {
                "split_index": int(split_index),
                "split_seed": int(split_seed),
                "selected_units": ",".join(selected_units),
                "n_selected_units": int(len(selected_units)),
                "outer_stratified": bool(outer_stratified),
                "inner_stratified": bool(inner_stratified),
                "outer_split_note": outer_note,
                "inner_split_note": inner_note,
            }
        )

        final_row, final_predictions = evaluate_units(
            df,
            context_pool_rows,
            test_rows,
            selected_units,
            feature_units,
            args,
            feature_set=FEATURE_SET_NAME,
            split_index=split_index,
            split_seed=split_seed,
            phase="final_test",
            extra={"n_selected_units": int(len(selected_units)), "outer_stratified": bool(outer_stratified), "inner_stratified": bool(inner_stratified)},
        )
        metrics_rows.append(final_row)
        prediction_frames.append(final_predictions)

        if args.evaluate_all_features_reference:
            all_row, all_predictions = evaluate_units(
                df,
                context_pool_rows,
                test_rows,
                all_feature_units,
                feature_units,
                args,
                feature_set=ALL_FEATURES_NAME,
                split_index=split_index,
                split_seed=split_seed,
                phase="final_test_all_features",
                extra={"n_selected_units": int(len(all_feature_units)), "outer_stratified": bool(outer_stratified), "inner_stratified": bool(inner_stratified)},
            )
            metrics_rows.append(all_row)
            prediction_frames.append(all_predictions)

        for local_index in context_pool_rows:
            assignment_rows.append({"split_index": int(split_index), "split_seed": int(split_seed), "source_row_id": int(df.loc[local_index, "source_row_id"]), "split_role": "context"})
        for local_index in test_rows:
            assignment_rows.append({"split_index": int(split_index), "split_seed": int(split_seed), "source_row_id": int(df.loc[local_index, "source_row_id"]), "split_role": "test"})

        print(f"[split {split_index + 1}/{len(split_seeds)}] seed={split_seed} selected={','.join(selected_units) if selected_units else 'none'} mse={final_row['raw_mse']:.3f} exact={final_row['rounded_exact_accuracy']:.1%}", flush=True)

    metrics = pd.DataFrame(metrics_rows)
    metrics.to_csv(args.output_dir / "metrics_by_split.csv", index=False)
    pd.DataFrame(candidate_rows).to_csv(args.output_dir / "validation_candidate_metrics.csv", index=False)
    pd.DataFrame(path_rows).to_csv(args.output_dir / "greedy_selection_path.csv", index=False)
    selected_by_split = pd.DataFrame(selected_rows)
    selected_by_split.to_csv(args.output_dir / "selected_feature_sets_by_split.csv", index=False)
    pd.DataFrame(assignment_rows).to_csv(args.output_dir / "split_assignments.csv", index=False)
    pd.concat(prediction_frames, ignore_index=True).to_csv(args.output_dir / "predictions_all_splits.csv", index=False)

    summary = summarize_metrics(metrics)
    summary.to_csv(args.output_dir / "summary.csv", index=False)
    setup = {
        "openhsr_rows": int(len(df)),
        "n_splits": int(args.n_splits),
        "test_size": int(args.test_size),
        "context_rows_per_split": int(len(df) - args.test_size),
        "validation_size_within_context": int(args.validation_size),
        "greedy_train_rows_within_context": int(len(df) - args.test_size - args.validation_size),
        "stratify_requested": bool(args.stratify),
        "outer_stratified_splits_used": int(selected_by_split["outer_stratified"].sum()),
        "inner_stratified_splits_used": int(selected_by_split["inner_stratified"].sum()),
        "candidate_unit_count": int(len(feature_units)),
        "candidate_units": list(feature_units),
        "selection_metric": args.selection_metric,
        "min_improvement": float(args.min_improvement),
        "max_greedy_steps": int(args.max_greedy_steps),
        "split_seeds": split_seeds,
        "elapsed_seconds": round(time.time() - started, 3),
    }
    common.write_json(args.output_dir / "setup_summary.json", setup)
    write_outputs(summary, selected_by_split, args.output_dir, setup)

    print(f"Wrote OpenHSR-only greedy outputs to: {args.output_dir}", flush=True)
    print(f"Elapsed seconds: {time.time() - started:.1f}", flush=True)


if __name__ == "__main__":
    main()
