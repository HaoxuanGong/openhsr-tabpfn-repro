from __future__ import annotations

import json
import math
import platform
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer


MISSING_CATEGORY = "__MISSING__"
TEXT_COMPONENT_PREFIX = "text_svd_"


@dataclass
class MixedFrames:
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    categorical_names: list[str]
    text_report: dict[str, Any]
    feature_names: list[str]


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def package_version(name: str) -> str | None:
    try:
        module = __import__(name)
        return getattr(module, "__version__", None)
    except Exception:
        return None


def environment_report() -> dict[str, Any]:
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "packages": {
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": package_version("sklearn"),
            "tabpfn": package_version("tabpfn"),
            "torch": package_version("torch"),
        },
        "time_unix": time.time(),
    }


def first_number(value: Any) -> float:
    if pd.isna(value):
        return np.nan
    match = re.search(r"[-+]?\d*\.?\d+", str(value))
    return float(match.group(0)) if match else np.nan


def load_openhsr(path: Path, target_column: str = "HSR") -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"OpenHSR CSV not found at {path}. Run scripts/download_openhsr.py "
            "or pass --openhsr-data/--openhsr-csv."
        )
    raw = pd.read_csv(path)
    if target_column not in raw.columns:
        raise KeyError(f"Target column {target_column!r} not found in {path}.")
    raw = raw.copy()
    raw["source_row_id"] = np.arange(len(raw), dtype=int)
    raw[target_column] = pd.to_numeric(raw[target_column], errors="coerce")
    raw = raw.loc[raw[target_column].notna()].copy().reset_index(drop=True)
    if "Size g" in raw.columns:
        raw["size_g_numeric"] = raw["Size g"].map(first_number)
        raw["size_g_text"] = raw["Size g"].astype("string").fillna("").astype(str)
    return raw


def build_openhsr_feature_units(
    df: pd.DataFrame,
    *,
    include_date_collected: bool = False,
) -> dict[str, dict[str, Any]]:
    candidates: dict[str, dict[str, Any]] = {}
    numeric_units = {
        "size_g": ["size_g_numeric"],
        "energy_kj": ["nutrient_energy_kj"],
        "energy_kcal": ["kcal_per_100g"],
        "protein": ["protein_g_per_100g"],
        "total_fat": ["fat_g_per_100g"],
        "saturated_fat": ["sat_fat_g_per_100g"],
        "carbohydrate": ["carbohydrates"],
        "sugars": ["sugars_g_per_100g"],
        "sodium": ["sodium_mg_per_100g"],
        "fibre": ["fiber_g_per_100g"],
    }
    categorical_units = {
        "product_type": ["Product type"],
        "category": ["category"],
        "country": ["country"],
        "retailer": ["retailer"],
        "data_source": ["data_source"],
    }
    text_units = {
        "product_name": ["product_name"],
        "ingredients": ["ingredients_text"],
        "allergen": ["allergen"],
        "size_text": ["size_g_text"],
    }
    if include_date_collected:
        categorical_units["date_collected"] = ["date_collected"]

    for name, columns in numeric_units.items():
        if all(column in df.columns for column in columns):
            candidates[name] = {"columns": columns, "type": "numeric"}
    for name, columns in categorical_units.items():
        if all(column in df.columns for column in columns):
            candidates[name] = {"columns": columns, "type": "categorical"}
    for name, columns in text_units.items():
        if all(column in df.columns for column in columns):
            candidates[name] = {"columns": columns, "type": "text"}
    return candidates


def clean_categorical(series: pd.Series) -> pd.Series:
    clean = series.astype("string").str.strip().str.lower()
    clean = clean.fillna(MISSING_CATEGORY)
    clean = clean.replace({"": MISSING_CATEGORY, "nan": MISSING_CATEGORY, "none": MISSING_CATEGORY})
    return clean.astype("category")


def align_test_categories(train: pd.Series, other: pd.Series) -> pd.Series:
    other_clean = clean_categorical(other).cat.set_categories(train.cat.categories)
    if MISSING_CATEGORY not in other_clean.cat.categories:
        other_clean = other_clean.cat.add_categories([MISSING_CATEGORY])
    return other_clean.fillna(MISSING_CATEGORY)


def make_text_corpus(df: pd.DataFrame, text_columns: list[str]) -> pd.Series:
    parts: list[pd.Series] = []
    for column in text_columns:
        if column in df.columns:
            values = df[column].fillna("").astype(str)
        else:
            values = pd.Series("", index=df.index)
        parts.append(column + ": " + values)
    if not parts:
        return pd.Series("", index=df.index)
    return pd.concat(parts, axis=1).agg(" [SEP] ".join, axis=1)


def selected_columns(
    selected_units: list[str],
    feature_units: dict[str, dict[str, Any]],
) -> tuple[list[str], list[str], list[str]]:
    numeric_columns: list[str] = []
    categorical_columns: list[str] = []
    text_columns: list[str] = []
    for unit in selected_units:
        if unit not in feature_units:
            raise KeyError(f"Unknown feature unit: {unit}")
        info = feature_units[unit]
        if info["type"] == "numeric":
            numeric_columns.extend(info["columns"])
        elif info["type"] == "categorical":
            categorical_columns.extend(info["columns"])
        elif info["type"] == "text":
            text_columns.extend(info["columns"])
        else:
            raise ValueError(f"Unknown feature type for {unit}: {info['type']}")
    return numeric_columns, categorical_columns, text_columns


def fit_text_svd(
    df: pd.DataFrame,
    train_rows: np.ndarray,
    test_rows: np.ndarray,
    text_columns: list[str],
    *,
    text_svd_components: int,
    text_max_features: int,
    random_seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    if not text_columns:
        return pd.DataFrame(index=train_rows), pd.DataFrame(index=test_rows), {"used": False, "source_columns": []}

    train_text = make_text_corpus(df.loc[train_rows], text_columns)
    test_text = make_text_corpus(df.loc[test_rows], text_columns)
    report: dict[str, Any] = {"used": True, "source_columns": text_columns}
    try:
        vectorizer = TfidfVectorizer(
            lowercase=True,
            strip_accents="unicode",
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.95,
            max_features=text_max_features,
            sublinear_tf=True,
        )
        train_tfidf = vectorizer.fit_transform(train_text)
        n_components = min(text_svd_components, max(1, min(train_tfidf.shape) - 1))
        svd = make_pipeline(
            TruncatedSVD(n_components=n_components, random_state=random_seed),
            Normalizer(copy=False),
        )
        train_features = svd.fit_transform(train_tfidf)
        test_features = svd.transform(vectorizer.transform(test_text))
        names = [f"{TEXT_COMPONENT_PREFIX}{i + 1:02d}" for i in range(n_components)]
        report.update({"tfidf_vocabulary_size": int(len(vectorizer.vocabulary_)), "svd_components": int(n_components)})
        return (
            pd.DataFrame(train_features, index=train_rows, columns=names),
            pd.DataFrame(test_features, index=test_rows, columns=names),
            report,
        )
    except ValueError as exc:
        report.update({"used": False, "error": str(exc), "svd_components": 0})
        return pd.DataFrame(index=train_rows), pd.DataFrame(index=test_rows), report


def build_mixed_frames(
    df: pd.DataFrame,
    train_rows: np.ndarray,
    test_rows: np.ndarray,
    selected_units: list[str],
    feature_units: dict[str, dict[str, Any]],
    *,
    text_svd_components: int,
    text_max_features: int,
    random_seed: int,
) -> MixedFrames:
    numeric_columns, categorical_columns, text_columns = selected_columns(selected_units, feature_units)
    numeric_columns = [c for c in numeric_columns if c in df.columns]
    categorical_columns = [c for c in categorical_columns if c in df.columns]
    text_columns = [c for c in text_columns if c in df.columns]

    X_train = pd.DataFrame(index=train_rows)
    X_test = pd.DataFrame(index=test_rows)
    for column in numeric_columns:
        X_train[column] = pd.to_numeric(df.loc[train_rows, column], errors="coerce")
        X_test[column] = pd.to_numeric(df.loc[test_rows, column], errors="coerce")
    for column in categorical_columns:
        X_train[column] = clean_categorical(df.loc[train_rows, column])
        X_test[column] = align_test_categories(X_train[column], df.loc[test_rows, column])

    train_text, test_text, text_report = fit_text_svd(
        df,
        train_rows,
        test_rows,
        text_columns,
        text_svd_components=text_svd_components,
        text_max_features=text_max_features,
        random_seed=random_seed,
    )
    X_train = pd.concat([X_train, train_text], axis=1)
    X_test = pd.concat([X_test, test_text], axis=1)
    return MixedFrames(
        X_train=X_train,
        X_test=X_test,
        categorical_names=categorical_columns,
        text_report=text_report,
        feature_names=list(X_train.columns),
    )


def round_to_half_star(values: np.ndarray | pd.Series) -> np.ndarray:
    values_array = np.asarray(values, dtype=float)
    rounded = np.round(values_array * 2.0) / 2.0
    return np.clip(rounded, 0.5, 5.0)


def hsr_to_class(values: np.ndarray | pd.Series) -> np.ndarray:
    return np.rint(np.asarray(values, dtype=float) * 2.0).astype(int)


def regressor_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    clipped = np.clip(y_pred, 0.5, 5.0)
    rounded = round_to_half_star(clipped)
    rounded_abs_error = np.abs(y_true - rounded)
    return {
        "raw_mae": float(mean_absolute_error(y_true, y_pred)),
        "raw_mse": float(mean_squared_error(y_true, y_pred)),
        "raw_rmse": float(math.sqrt(mean_squared_error(y_true, y_pred))),
        "raw_r2": float(r2_score(y_true, y_pred)),
        "clipped_mae": float(mean_absolute_error(y_true, clipped)),
        "clipped_mse": float(mean_squared_error(y_true, clipped)),
        "clipped_rmse": float(math.sqrt(mean_squared_error(y_true, clipped))),
        "clipped_r2": float(r2_score(y_true, clipped)),
        "rounded_exact_accuracy": float(np.mean(hsr_to_class(y_true) == hsr_to_class(rounded))),
        "rounded_mae": float(mean_absolute_error(y_true, rounded)),
        "rounded_rmse": float(math.sqrt(mean_squared_error(y_true, rounded))),
        "rounded_within_0_5_accuracy": float(np.mean(rounded_abs_error <= 0.5 + 1e-12)),
        "rounded_within_1_0_accuracy": float(np.mean(rounded_abs_error <= 1.0 + 1e-12)),
    }


def predictions_frame(row_ids: np.ndarray, y_true: np.ndarray, y_pred: np.ndarray) -> pd.DataFrame:
    clipped = np.clip(np.asarray(y_pred, dtype=float), 0.5, 5.0)
    rounded = round_to_half_star(clipped)
    return pd.DataFrame(
        {
            "row_id": row_ids.astype(int),
            "hsr_true": y_true.astype(float),
            "pred_raw": np.asarray(y_pred, dtype=float),
            "pred_clipped": clipped,
            "pred_rounded": rounded,
            "abs_raw_error": np.abs(y_true - y_pred),
            "abs_rounded_error": np.abs(y_true - rounded),
        }
    )

