# ============================================================
# Experiment: OpenHSR baseline random-split audit
# Paper Section / Research Question:
#   Audits how sensitive the reported OpenHSR baseline metrics are to the
#   particular 80/20 train-test split used in the public benchmark.
#
# Purpose:
#   Re-run OpenHSR-style classical regression baselines over many random 80/20
#   splits and compare the resulting distributions with the values reported in
#   the OpenHSR benchmark table.
#
# Dataset(s):
#   - Source files:
#       data/OpenHSR.csv, downloaded from the public OpenHSR repository.
#   - Retailer(s):
#       Public OpenHSR sources as provided by the OpenHSR authors.
#   - Inclusion criteria:
#       Rows with numeric HSR labels and the OpenHSR numeric predictor columns
#       used by the released benchmark.
#   - Exclusion criteria:
#       Rows with missing target values.
#   - Target variable:
#       HSR.
#   - Unit convention:
#       Uses OpenHSR fields as distributed; no unit conversion is performed.
#
# HSR Assumptions:
#   - HSR algorithm version:
#       HSR labels are taken from OpenHSR and are not recalculated.
#   - Category mapping:
#       No category mapping is performed.
#   - Treatment of ambiguous categories:
#       Not used by the classical numeric baseline feature set.
#   - Treatment of missing nutrition fields:
#       Median imputation inside the scikit-learn pipeline.
#   - Treatment of ineligible products:
#       Not independently reassessed; follows OpenHSR inclusion.
#
# Model / Method:
#   - Model type:
#       Linear regression, decision tree, K-nearest neighbors, random forest,
#       scikit-learn gradient boosting, support vector regression, and optional
#       multilayer perceptron.
#   - Feature set:
#       OpenHSR numeric nutrient and size fields used by the public benchmark.
#   - Preprocessing:
#       Median imputation and standard scaling where appropriate.
#   - Train/validation/test split:
#       Repeated random 80/20 splits.
#   - Random seed(s):
#       Defaults to 0..n_splits-1 plus the reported seed 42.
#
# Hyperparameters:
#   Classical scikit-learn defaults are used unless matching the OpenHSR-style
#   released code requires a fixed setting. Optional MLP defaults:
#   --neural-epochs 7, --neural-batch-size 32, --neural-lr 0.01.
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
#       Split-sensitivity over repeated random 80/20 splits.
#
# Hardware Used:
#   - CPU:
#       Any modern CPU.
#   - GPU:
#       None for classical models; optional MLP can use CPU.
#   - RAM:
#       Minimal for OpenHSR.
#   - Storage:
#       Requires OpenHSR CSV and result files.
#   - OS:
#       Cross-platform Python.
#
# Compute Required to Reproduce:
#   - Expected wall-clock time:
#       Minutes for classical models over 200 splits; longer with --include-mlp.
#   - Number of runs:
#       One run containing n_splits repeated random splits.
#   - Peak RAM:
#       Low.
#   - Peak GPU memory:
#       N/A.
#   - Required disk space:
#       Less than 1 GB for outputs.
#   - Parallelism:
#       scikit-learn model internals only.
#
# Software Environment:
#   - Python/R version:
#       Python 3.10+ recommended.
#   - Key packages:
#       numpy, pandas, scikit-learn, matplotlib; torch/schedulefree only for
#       optional MLP retraining.
#   - Environment file:
#       requirements.txt.
#
# Outputs:
#   - Tables:
#       metrics_by_split.csv, summary_by_model.csv,
#       reported_vs_random_splits.csv, paper_table_rows.tex.
#   - Figures:
#       seed42_position_exact_accuracy.pdf/.png unless --no-figure is set.
#   - Models:
#       None.
#   - Logs:
#       Console output and JSON configuration files.
#
# Reproducibility Notes:
#   - The gradient-boosting row is named explicitly as scikit-learn gradient
#     boosting to avoid implying that XGBoost is used.
#
# Limitations:
#   - This audit evaluates split sensitivity, not all possible modelling
#     choices or hyperparameter searches.
# ============================================================
from __future__ import annotations

import argparse
import json
import math
import platform
import subprocess
import sys
import time
from collections import OrderedDict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from sklearn import ensemble, linear_model, metrics, model_selection, neighbors, pipeline, preprocessing, svm, tree


REPORTED_TABLE = {
    "Linear regression": {"mse": 0.82, "exact": 0.42, "within_0_5": 0.76},
    "Decision tree": {"mse": 0.55, "exact": 0.34, "within_0_5": 0.76},
    "K-nearest neighbor": {"mse": 0.40, "exact": 0.42, "within_0_5": 0.86},
    "Random forest": {"mse": 0.39, "exact": 0.32, "within_0_5": 0.84},
    "Gradient boosting": {"mse": 0.35, "exact": 0.54, "within_0_5": 0.92},
    "Support vector machine": {"mse": 0.28, "exact": 0.62, "within_0_5": 0.86},
    "Multilayer perceptron": {"mse": 0.67, "exact": 0.40, "within_0_5": 0.78},
}

MODEL_ORDER = list(REPORTED_TABLE)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit OpenHSR baseline performance over repeated random 80/20 splits.")
    parser.add_argument("--openhsr-csv", type=Path, default=Path("data/OpenHSR.csv"))
    parser.add_argument("--upstream-repo", type=Path, default=Path("."))
    parser.add_argument("--output-dir", type=Path, default=Path("results/openhsr_baseline_random_split_audit"))
    parser.add_argument("--n-splits", type=int, default=200)
    parser.add_argument("--split-seeds", type=str, default=None)
    parser.add_argument("--reported-seed", type=int, default=42)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--model-rng-seed", type=int, default=0)
    parser.add_argument("--no-figure", action="store_true")
    parser.add_argument("--include-mlp", action="store_true")
    parser.add_argument("--neural-epochs", type=int, default=7)
    parser.add_argument("--neural-batch-size", type=int, default=32)
    parser.add_argument("--neural-lr", type=float, default=0.01)
    parser.add_argument("--neural-model-seed", type=int, default=123)
    parser.add_argument("--neural-validation-seed", type=int, default=123)
    return parser.parse_args()


def get_git_commit(repo: Path) -> str | None:
    if not repo.exists():
        return None
    try:
        return subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True).strip()
    except Exception:
        return None


def package_version(module_name: str) -> str | None:
    try:
        module = __import__(module_name)
        return getattr(module, "__version__", None)
    except Exception:
        return None


def resolve_split_seeds(args: argparse.Namespace) -> list[int]:
    if args.split_seeds:
        seeds = [int(x.strip()) for x in args.split_seeds.split(",") if x.strip()]
    else:
        seeds = list(range(args.n_splits))
    if args.reported_seed not in seeds:
        seeds.append(args.reported_seed)
    return sorted(set(seeds))


def load_upstream_features(csv_path: Path) -> tuple[pd.DataFrame, pd.Series, list[str], pd.DataFrame]:
    if not csv_path.exists():
        raise FileNotFoundError(f"OpenHSR CSV not found: {csv_path}. Run scripts/download_openhsr.py or pass --openhsr-csv.")

    raw = pd.read_csv(csv_path)
    raw = raw.loc[raw["HSR"].notna()].reset_index(drop=True)
    encoded = pd.get_dummies(raw, columns=["category"])
    selected_cols = list(encoded.select_dtypes(include=["float64", "int64", "bool"]).columns)
    if "HSR" not in selected_cols:
        raise ValueError("HSR target was not found among selected numeric columns.")

    X = encoded[selected_cols].drop(columns=["HSR"])
    y = encoded["HSR"].astype(float)
    missing_counts = X.isna().sum()
    if int(missing_counts.sum()) > 0:
        missing = missing_counts[missing_counts > 0].to_dict()
        raise ValueError(f"Selected features contain missing values but the upstream notebook does not impute them: {missing}")
    return X, y, list(X.columns), raw


def clip_hsr(predictions: np.ndarray) -> np.ndarray:
    pred = np.asarray(predictions, dtype=float).copy()
    pred[pred < 0.5] = 0.5
    pred[pred > 5.0] = 5.0
    return pred


def half_star(predictions: np.ndarray) -> np.ndarray:
    return np.round(clip_hsr(predictions) * 2.0) / 2.0


def evaluate_predictions(y_true: np.ndarray, predictions: np.ndarray) -> dict[str, float]:
    clipped = clip_hsr(predictions)
    rounded = half_star(clipped)
    return {
        "mse": float(metrics.mean_squared_error(y_true, clipped)),
        "mae": float(metrics.mean_absolute_error(y_true, clipped)),
        "rmse": float(math.sqrt(metrics.mean_squared_error(y_true, clipped))),
        "exact": float(np.mean(np.abs(rounded - y_true) == 0.0)),
        "within_0_5": float(np.mean(np.abs(rounded - y_true) <= 0.5)),
    }


def fit_predict_models(
    X_train_scaled: np.ndarray,
    X_test_scaled: np.ndarray,
    y_train: np.ndarray,
    model_rng_seed: int,
    split_seed: int,
) -> OrderedDict[str, np.ndarray]:
    predictions: OrderedDict[str, np.ndarray] = OrderedDict()
    np.random.seed(model_rng_seed + split_seed)

    lr = linear_model.LinearRegression()
    lr.fit(X_train_scaled, y_train)
    predictions["Linear regression"] = lr.predict(X_test_scaled)

    dtr = tree.DecisionTreeRegressor(random_state=0, max_depth=4, min_samples_split=10, max_features=8)
    dtr.fit(X_train_scaled, y_train)
    predictions["Decision tree"] = dtr.predict(X_test_scaled)

    knn_grid = model_selection.GridSearchCV(neighbors.KNeighborsRegressor(), {"n_neighbors": [3, 5, 7, 10, 20]})
    knn_grid.fit(X_train_scaled, y_train)
    knn = neighbors.KNeighborsRegressor(n_neighbors=knn_grid.best_params_["n_neighbors"])
    knn.fit(X_train_scaled, y_train)
    predictions["K-nearest neighbor"] = knn.predict(X_test_scaled)

    rfr = ensemble.RandomForestRegressor(max_depth=5, min_samples_split=5, max_features="log2", random_state=0)
    rfr.fit(X_train_scaled, y_train)
    predictions["Random forest"] = rfr.predict(X_test_scaled)

    gbr = pipeline.make_pipeline(
        preprocessing.StandardScaler(),
        ensemble.GradientBoostingRegressor(
            n_estimators=500,
            max_depth=4,
            min_samples_split=5,
            learning_rate=0.01,
            loss="squared_error",
        ),
    )
    gbr.fit(X_train_scaled, y_train)
    predictions["Gradient boosting"] = gbr.predict(X_test_scaled)

    svr = pipeline.make_pipeline(preprocessing.StandardScaler(), svm.SVR(C=100, epsilon=0.01, kernel="rbf"))
    svr.fit(X_train_scaled, y_train)
    predictions["Support vector machine"] = svr.predict(X_test_scaled)

    return predictions


def train_openhsr_mlp(
    X_train_scaled: np.ndarray,
    X_test_scaled: np.ndarray,
    y_train: np.ndarray,
    neural_epochs: int,
    neural_batch_size: int,
    neural_lr: float,
    neural_model_seed: int,
    neural_validation_seed: int,
) -> np.ndarray:
    try:
        import schedulefree
        import torch
        from torch import nn
    except ImportError as exc:
        raise ImportError("MLP retraining requires torch and schedulefree. Install requirements or omit --include-mlp.") from exc

    class Network(nn.Module):
        def __init__(self, init_column: int, n_nodes: int = 32, n_loop: int = 3) -> None:
            super().__init__()
            layers: list[nn.Module] = [nn.BatchNorm1d(init_column), nn.Linear(init_column, n_nodes), nn.ReLU()]
            for _ in range(n_loop):
                layers.extend([nn.BatchNorm1d(n_nodes), nn.Linear(n_nodes, n_nodes), nn.ReLU()])
            layers.append(nn.Linear(n_nodes, 1))
            self.seq = nn.Sequential(*layers)

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            return self.seq(x)

    torch.manual_seed(neural_model_seed)
    indices = np.arange(len(X_train_scaled))
    rng = np.random.RandomState(neural_validation_seed)
    rng.shuffle(indices)
    train_size = int(0.8 * len(X_train_scaled))
    train_idx = indices[:train_size]
    valid_idx = indices[train_size:]

    X_train_tensor = torch.from_numpy(X_train_scaled[train_idx].astype(np.float32))
    y_train_tensor = torch.from_numpy(y_train[train_idx].astype(np.float32))
    X_valid_tensor = torch.from_numpy(X_train_scaled[valid_idx].astype(np.float32))
    X_test_tensor = torch.from_numpy(X_test_scaled.astype(np.float32))

    model = Network(X_train_tensor.shape[1])
    optimizer = schedulefree.AdamWScheduleFree(model.parameters(), lr=neural_lr)
    n_train_batches = max(1, math.ceil(len(X_train_tensor) / neural_batch_size))

    for _ in range(neural_epochs):
        model.train()
        optimizer.train()
        for batch_count in range(n_train_batches):
            start = batch_count * neural_batch_size
            end = min((batch_count + 1) * neural_batch_size, len(X_train_tensor))
            optimizer.zero_grad()
            output = model(X_train_tensor[start:end].float()).squeeze()
            loss = ((y_train_tensor[start:end].float() - output) ** 2).mean()
            loss.backward()
            optimizer.step()
        model.eval()
        optimizer.eval()
        with torch.no_grad():
            _ = model(X_valid_tensor.float())

    model.eval()
    with torch.no_grad():
        return model(X_test_tensor).squeeze().detach().cpu().numpy()


def run_split(seed: int, X: pd.DataFrame, y: pd.Series, args: argparse.Namespace) -> list[dict[str, Any]]:
    train_idx, test_idx = model_selection.train_test_split(np.arange(len(X)), test_size=args.test_size, random_state=seed)
    X_train = X.iloc[train_idx].to_numpy(dtype=float)
    X_test = X.iloc[test_idx].to_numpy(dtype=float)
    y_train = y.iloc[train_idx].to_numpy(dtype=float)
    y_test = y.iloc[test_idx].to_numpy(dtype=float)

    scaler = preprocessing.StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    split_predictions = fit_predict_models(X_train_scaled, X_test_scaled, y_train, args.model_rng_seed, seed)
    if args.include_mlp:
        split_predictions["Multilayer perceptron"] = train_openhsr_mlp(
            X_train_scaled,
            X_test_scaled,
            y_train,
            args.neural_epochs,
            args.neural_batch_size,
            args.neural_lr,
            args.neural_model_seed,
            args.neural_validation_seed,
        )

    rows: list[dict[str, Any]] = []
    for model_name, pred in split_predictions.items():
        metric_row = evaluate_predictions(y_test, pred)
        metric_row.update({"split_seed": seed, "model": model_name, "n_train": int(len(train_idx)), "n_test": int(len(test_idx)), "y_test_mean": float(np.mean(y_test)), "y_test_sd": float(np.std(y_test, ddof=1))})
        rows.append(metric_row)
    return rows


def summarize_results(metrics_df: pd.DataFrame, reported_seed: int) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    metric_specs = [("mse", "lower"), ("mae", "lower"), ("rmse", "lower"), ("exact", "higher"), ("within_0_5", "higher")]
    reported_df = metrics_df.loc[metrics_df["split_seed"] == reported_seed]
    for model_name in MODEL_ORDER:
        model_df = metrics_df.loc[metrics_df["model"] == model_name]
        reported_row = reported_df.loc[reported_df["model"] == model_name]
        if reported_row.empty or model_df.empty:
            continue
        for metric_name, direction in metric_specs:
            values = model_df[metric_name].to_numpy(dtype=float)
            reported_value = float(reported_row.iloc[0][metric_name])
            if direction == "lower":
                as_good_or_better = float(np.mean(values <= reported_value) * 100.0)
                numeric_percentile = float(np.mean(values <= reported_value) * 100.0)
            else:
                as_good_or_better = float(np.mean(values >= reported_value) * 100.0)
                numeric_percentile = float(np.mean(values <= reported_value) * 100.0)
            rows.append(
                {
                    "model": model_name,
                    "metric": metric_name,
                    "direction": direction,
                    "random_mean": float(np.mean(values)),
                    "random_sd": float(np.std(values, ddof=1)),
                    "random_median": float(np.median(values)),
                    "random_p05": float(np.quantile(values, 0.05)),
                    "random_p95": float(np.quantile(values, 0.95)),
                    "random_min": float(np.min(values)),
                    "random_max": float(np.max(values)),
                    "reported_seed_value": reported_value,
                    "reported_seed_numeric_percentile": numeric_percentile,
                    "random_splits_as_good_or_better_pct": as_good_or_better,
                }
            )
    return pd.DataFrame(rows)


def compare_to_reported_table(metrics_df: pd.DataFrame, reported_seed: int) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    seed_df = metrics_df.loc[metrics_df["split_seed"] == reported_seed].set_index("model")
    for model_name, reported in REPORTED_TABLE.items():
        if model_name not in seed_df.index:
            continue
        observed = seed_df.loc[model_name]
        for metric_name in ["mse", "exact", "within_0_5"]:
            rows.append({"model": model_name, "metric": metric_name, "reported_table_value": reported[metric_name], "recomputed_seed_value": float(observed[metric_name]), "difference_recomputed_minus_reported": float(observed[metric_name] - reported[metric_name])})
    return pd.DataFrame(rows)


def write_environment(output_dir: Path, args: argparse.Namespace, features: list[str], raw: pd.DataFrame) -> None:
    env = {
        "python": sys.version,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "packages": {"numpy": np.__version__, "pandas": pd.__version__, "scikit_learn": sklearn.__version__, "matplotlib": package_version("matplotlib"), "torch": package_version("torch"), "schedulefree": package_version("schedulefree")},
        "upstream_repo": str(args.upstream_repo),
        "upstream_commit": get_git_commit(args.upstream_repo),
        "n_rows": int(len(raw)),
        "target_counts": {str(k): int(v) for k, v in raw["HSR"].value_counts().sort_index().items()},
        "feature_columns": features,
    }
    (output_dir / "environment.json").write_text(json.dumps(env, indent=2), encoding="utf-8")


def plot_split_audit(metrics_df: pd.DataFrame, reported_seed: int, output_dir: Path) -> None:
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9.5, "axes.labelsize": 10, "xtick.labelsize": 8.5, "ytick.labelsize": 9, "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42, "ps.fonttype": 42})
    metric_panels = [("mse", "MSE"), ("exact", "Exact agreement (%)"), ("within_0_5", "Within 0.5 star (%)")]
    colors = {"band": "#d7dde5", "iqr": "#8391a5", "median": "#253447", "seed": "#c43c35", "grid": "#edf0f4"}
    fig, axes = plt.subplots(1, 3, figsize=(10.8, 3.35), sharey=True, gridspec_kw={"wspace": 0.10})
    model_order = [model for model in MODEL_ORDER if model in set(metrics_df["model"])]
    y_positions = np.arange(len(model_order))

    for ax, (metric_name, x_label) in zip(axes, metric_panels):
        for ypos, model_name in zip(y_positions, model_order):
            values = metrics_df.loc[metrics_df["model"] == model_name, metric_name].to_numpy(dtype=float)
            seed_values = metrics_df.loc[(metrics_df["model"] == model_name) & (metrics_df["split_seed"] == reported_seed), metric_name].to_numpy(dtype=float)
            scale = 100.0 if metric_name in {"exact", "within_0_5"} else 1.0
            values = values * scale
            seed_value = seed_values[0] * scale if len(seed_values) else np.nan
            q05, q25, q50, q75, q95 = np.quantile(values, [0.05, 0.25, 0.50, 0.75, 0.95])
            ax.hlines(ypos, q05, q95, color=colors["band"], linewidth=7.5, zorder=1)
            ax.hlines(ypos, q25, q75, color=colors["iqr"], linewidth=7.5, zorder=2)
            ax.plot(q50, ypos, "o", color=colors["median"], markersize=4.6, zorder=3)
            ax.plot(seed_value, ypos, "D", color=colors["seed"], markersize=4.5, zorder=4)
        ax.set_xlabel(x_label)
        ax.grid(axis="x", color=colors["grid"], linewidth=0.8)
        ax.set_axisbelow(True)
        ax.tick_params(axis="y", length=0)

    axes[0].set_yticks(y_positions)
    axes[0].set_yticklabels(model_order)
    axes[0].invert_yaxis()
    handles = [
        plt.Line2D([0], [0], color=colors["band"], linewidth=7.5, label="5-95% of random splits"),
        plt.Line2D([0], [0], color=colors["iqr"], linewidth=7.5, label="25-75%"),
        plt.Line2D([0], [0], marker="o", color="none", markerfacecolor=colors["median"], markersize=5, label="median"),
        plt.Line2D([0], [0], marker="D", color="none", markerfacecolor=colors["seed"], markersize=5, label=f"seed {reported_seed}"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.53, -0.02), fontsize=8.5, handlelength=1.8)
    fig.subplots_adjust(left=0.18, right=0.99, top=0.97, bottom=0.25)
    fig.savefig(output_dir / "openhsr_random_split_audit.pdf", bbox_inches="tight")
    fig.savefig(output_dir / "openhsr_random_split_audit.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    start_time = time.time()
    np.random.seed(args.model_rng_seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    split_seeds = resolve_split_seeds(args)
    X, y, features, raw = load_upstream_features(args.openhsr_csv)

    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    config["split_seeds_resolved"] = split_seeds
    config["note"] = "This reimplements the executable OpenHSR notebook pipeline for classical models. The notebook's XGBR variable is sklearn GradientBoostingRegressor."
    (args.output_dir / "run_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    write_environment(args.output_dir, args, features, raw)

    all_rows: list[dict[str, Any]] = []
    for count, seed in enumerate(split_seeds, start=1):
        all_rows.extend(run_split(seed, X, y, args))
        if count == 1 or count % 25 == 0 or count == len(split_seeds):
            print(f"[{count:>4}/{len(split_seeds)}] completed split seed {seed}", flush=True)

    metrics_df = pd.DataFrame(all_rows)
    metrics_df = metrics_df[["split_seed", "model", "n_train", "n_test", "y_test_mean", "y_test_sd", "mse", "mae", "rmse", "exact", "within_0_5"]]
    metrics_df.to_csv(args.output_dir / "metrics_by_split.csv", index=False)
    summary_df = summarize_results(metrics_df, args.reported_seed)
    summary_df.to_csv(args.output_dir / "summary_by_model.csv", index=False)
    reported_seed_df = metrics_df.loc[metrics_df["split_seed"] == args.reported_seed].copy()
    reported_seed_df.to_csv(args.output_dir / "reported_seed_metrics.csv", index=False)
    compare_to_reported_table(metrics_df, args.reported_seed).to_csv(args.output_dir / "reported_table_comparison.csv", index=False)
    if not args.no_figure:
        plot_split_audit(metrics_df, args.reported_seed, args.output_dir)

    print("\nRecomputed reported-seed split:")
    print(reported_seed_df.assign(exact=lambda d: d["exact"] * 100.0, within_0_5=lambda d: d["within_0_5"] * 100.0)[["model", "mse", "exact", "within_0_5"]].to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print(f"\nOutputs written to: {args.output_dir}")
    print(f"Elapsed seconds: {time.time() - start_time:.1f}")


if __name__ == "__main__":
    main()
