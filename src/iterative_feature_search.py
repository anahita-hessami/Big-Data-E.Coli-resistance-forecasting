"""Leakage-safe iterative feature and model search for EU/EEA E. coli AMR.

The search is deliberately bounded. Candidate features must be scientifically
interpretable and available by the forecast origin year. Feature decisions use
expanding-window cross-validation on pre-test years only. The latest target year
is used once for confirmation after the specification has been frozen.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import (
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.inspection import permutation_importance
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline

from src.regional_amr import _metrics, _preprocessor


BASE_NUMERIC = [
    "resistance_pct", "resistance_lag1", "resistance_lag2", "resistance_lag3",
    "resistance_trend4", "resistance_volatility4", "log_tested",
    "breakpoint_change_from_prior",
]
BASE_CATEGORICAL = ["antibiotic", "ast_standard", "ast_version"]


def _safe_relative_change(recent: pd.Series, earlier: pd.Series) -> pd.Series:
    denominator = earlier.abs().where(earlier.abs().gt(1e-9))
    return ((recent - earlier) / denominator).clip(-5, 5)


def _row_summary(frame: pd.DataFrame, columns: list[str], operation: str) -> pd.Series:
    values = frame[columns]
    sufficient = values.notna().sum(axis=1).ge(2)
    if operation == "mean":
        result = values.mean(axis=1)
    elif operation == "sum":
        result = values.sum(axis=1, min_count=2)
    elif operation == "std":
        result = values.std(axis=1, ddof=0)
    else:
        raise ValueError(operation)
    return result.where(sufficient)


def engineer_enhanced_features(panel: pd.DataFrame) -> pd.DataFrame:
    """Add origin-time features without using a future AMR outcome."""
    out = panel.sort_values(["iso3", "antibiotic", "year"]).copy()
    amr_group = out.groupby(["iso3", "antibiotic"], sort=False)

    # Reporting intensity is explicitly named as reported surveillance burden;
    # it is not presented as population-corrected disease incidence.
    out["ast_tested_per_100k"] = 100_000 * out["tested"] / out["population"]
    out["reported_resistant_per_100k"] = 100_000 * out["resistant"] / out["population"]
    prior_tested = amr_group["tested"].shift(1)
    out["tested_growth1"] = _safe_relative_change(out["tested"], prior_tested)
    out["testing_volatility4"] = amr_group["log_tested"].transform(
        lambda values: values.rolling(4, min_periods=2).std(ddof=0)
    )

    consumption_lags = [
        "family_consumption_lag1", "family_consumption_lag2",
        "family_consumption_lag3",
    ]
    out["family_consumption_mean3"] = _row_summary(out, consumption_lags, "mean")
    out["family_consumption_sum3"] = _row_summary(out, consumption_lags, "sum")
    out["family_consumption_volatility3"] = _row_summary(out, consumption_lags, "std")
    out["family_consumption_change1"] = (
        out["family_consumption_lag1"] - out["family_consumption_lag2"]
    )
    out["family_consumption_trend3"] = (
        out["family_consumption_lag1"] - out["family_consumption_lag3"]
    ) / 2

    # Country-year variables must be lagged on a unique country-year table;
    # shifting the repeated antibiotic rows would create incorrect lags.
    country_columns = [
        "consumption_community", "consumption_hospital", "reserve_pct",
        "population", "age65_pct", "urban_pct", "gdp_per_capita",
        "health_expenditure_pct_gdp",
    ]
    country_year = (
        out[["iso3", "year"] + country_columns]
        .drop_duplicates(["iso3", "year"])
        .sort_values(["iso3", "year"])
    )
    country_group = country_year.groupby("iso3", sort=False)
    for column in ["consumption_community", "consumption_hospital", "reserve_pct"]:
        for lag in (1, 2, 3):
            country_year[f"{column}_lag{lag}"] = country_group[column].shift(lag)

    country_year["reserve_mean3"] = _row_summary(
        country_year, ["reserve_pct_lag1", "reserve_pct_lag2", "reserve_pct_lag3"], "mean"
    )
    country_year["reserve_change1"] = (
        country_year["reserve_pct_lag1"] - country_year["reserve_pct_lag2"]
    )
    country_year["hospital_community_ratio_lag1"] = (
        country_year["consumption_hospital_lag1"] /
        country_year["consumption_community_lag1"].replace(0, np.nan)
    )
    country_year["log_population"] = np.log1p(country_year["population"])
    country_year["population_growth1"] = _safe_relative_change(
        country_year["population"], country_group["population"].shift(1)
    )
    country_year["population_growth3"] = _safe_relative_change(
        country_year["population"], country_group["population"].shift(3)
    )
    country_year["age65_change1"] = (
        country_year["age65_pct"] - country_group["age65_pct"].shift(1)
    )
    country_year["urban_change1"] = (
        country_year["urban_pct"] - country_group["urban_pct"].shift(1)
    )
    country_year["log_gdp_per_capita"] = np.log1p(country_year["gdp_per_capita"])
    country_year["gdp_growth1"] = _safe_relative_change(
        country_year["gdp_per_capita"], country_group["gdp_per_capita"].shift(1)
    )
    country_year["health_expenditure_change1"] = (
        country_year["health_expenditure_pct_gdp"] -
        country_group["health_expenditure_pct_gdp"].shift(1)
    )
    added_country_columns = [
        column for column in country_year.columns
        if column not in {"iso3", "year", *country_columns}
    ]
    # Replace any earlier single-lag implementation with the consistently
    # calculated country-year series and avoid pandas _x/_y suffixes.
    out = out.drop(
        columns=[column for column in added_country_columns if column in out.columns],
        errors="ignore",
    )
    out = out.merge(
        country_year[["iso3", "year"] + added_country_columns],
        on=["iso3", "year"], how="left", validate="many_to_one",
    )

    # Weighted EU/EEA context excludes the focal country to prevent the feature
    # from simply reproducing its own observed resistance percentage.
    eu_totals = out.groupby(["antibiotic", "year"], as_index=False).agg(
        eu_resistant_total=("resistant", "sum"), eu_tested_total=("tested", "sum")
    )
    out = out.merge(eu_totals, on=["antibiotic", "year"], how="left", validate="many_to_one")
    excl_tested = out["eu_tested_total"] - out["tested"]
    out["eu_resistance_excl_country"] = (
        100 * (out["eu_resistant_total"] - out["resistant"]) / excl_tested.where(excl_tested.gt(0))
    )
    out["resistance_gap_to_eu"] = out["resistance_pct"] - out["eu_resistance_excl_country"]

    eu_series = (
        out.groupby(["antibiotic", "year"], as_index=False)
        .agg(eu_resistant=("resistant", "sum"), eu_tested=("tested", "sum"))
        .sort_values(["antibiotic", "year"])
    )
    eu_series["eu_resistance"] = 100 * eu_series["eu_resistant"] / eu_series["eu_tested"]
    eu_series["eu_resistance_lag1"] = eu_series.groupby("antibiotic")["eu_resistance"].shift(1)
    eu_series["eu_resistance_change1"] = (
        eu_series["eu_resistance"] - eu_series["eu_resistance_lag1"]
    )
    out = out.merge(
        eu_series[["antibiotic", "year", "eu_resistance_change1"]],
        on=["antibiotic", "year"], how="left", validate="many_to_one",
    )

    # Resistance in the other reported antibiotic groups is a proxy for the
    # country's broader multidrug-resistance environment.
    profile = out.groupby(["iso3", "year"])["resistance_pct"]
    count = profile.transform("count")
    total = profile.transform("sum")
    total_sq = out["resistance_pct"].pow(2).groupby([out["iso3"], out["year"]]).transform("sum")
    other_count = count - 1
    out["other_antibiotic_mean"] = (
        (total - out["resistance_pct"]) / other_count.where(other_count.gt(0))
    )
    other_variance = (
        (total_sq - out["resistance_pct"].pow(2)) /
        other_count.where(other_count.gt(0)) - out["other_antibiotic_mean"].pow(2)
    )
    out["other_antibiotic_sd"] = np.sqrt(other_variance.clip(lower=0))
    high = out["resistance_pct"].ge(25).astype(int)
    out["other_high_resistance_count"] = (
        high.groupby([out["iso3"], out["year"]]).transform("sum") - high
    )
    out["country_profile_gap"] = out["resistance_pct"] - out["other_antibiotic_mean"]

    out["origin_year_index"] = out["year"] - out["year"].min()
    out["trend_x_volatility"] = out["resistance_trend4"] * out["resistance_volatility4"]
    out["resistance_x_consumption"] = (
        out["resistance_pct"] * out["family_consumption_mean3"]
    )
    out["trend_x_consumption"] = (
        out["resistance_trend4"] * out["family_consumption_mean3"]
    )
    out["resistance_x_eu_gap"] = out["resistance_pct"] * out["resistance_gap_to_eu"]
    return out.drop(columns=["eu_resistant_total", "eu_tested_total"])


FEATURE_GROUPS = {
    "Existing consumption history": {
        "numeric": [
            "family_consumption_lag1", "family_consumption_lag2",
            "family_consumption_lag3",
        ],
        "categorical": ["antibiotic_family"],
    },
    "Consumption dynamics": {
        "numeric": [
            "family_consumption_mean3", "family_consumption_sum3",
            "family_consumption_volatility3", "family_consumption_change1",
            "family_consumption_trend3", "consumption_community_lag1",
            "consumption_hospital_lag1", "hospital_community_ratio_lag1",
        ],
        "categorical": [],
    },
    "Prescribing mix dynamics": {
        "numeric": [
            "reserve_pct_lag1", "reserve_pct_lag2", "reserve_pct_lag3",
            "reserve_mean3", "reserve_change1",
        ],
        "categorical": [],
    },
    "Testing and reported burden": {
        "numeric": [
            "ast_tested_per_100k", "reported_resistant_per_100k",
            "tested_growth1", "testing_volatility4",
        ],
        "categorical": [],
    },
    "Existing static demographics": {
        "numeric": [
            "population", "age65_pct", "urban_pct", "gdp_per_capita",
            "health_expenditure_pct_gdp",
        ],
        "categorical": [],
    },
    "Demographic change": {
        "numeric": [
            "log_population", "population_growth1", "population_growth3",
            "age65_change1", "urban_change1", "log_gdp_per_capita",
            "gdp_growth1", "health_expenditure_change1",
        ],
        "categorical": [],
    },
    "European AMR context": {
        "numeric": [
            "eu_resistance_excl_country", "resistance_gap_to_eu",
            "eu_resistance_change1",
        ],
        "categorical": [],
    },
    "Cross-antibiotic context": {
        "numeric": [
            "other_antibiotic_mean", "other_antibiotic_sd",
            "other_high_resistance_count", "country_profile_gap",
        ],
        "categorical": [],
    },
    "Mechanistic interactions": {
        "numeric": [
            "trend_x_volatility", "resistance_x_consumption",
            "trend_x_consumption", "resistance_x_eu_gap",
        ],
        "categorical": [],
    },
    "Origin time": {"numeric": ["origin_year_index"], "categorical": []},
    "Country fixed effect": {"numeric": [], "categorical": ["country"]},
}


def _model_specs(random_state: int = 42):
    return {
        "Ridge": (
            Ridge(), [{"alpha": value} for value in [0.1, 1.0, 10.0, 100.0, 300.0]]
        ),
        "Gradient boosting": (
            GradientBoostingRegressor(random_state=random_state),
            [
                {"n_estimators": n, "learning_rate": rate, "max_depth": depth,
                 "min_samples_leaf": leaf}
                for n in [100, 200] for rate in [0.05]
                for depth in [2, 3] for leaf in [3]
            ],
        ),
        "Random forest": (
            RandomForestRegressor(random_state=random_state, n_jobs=1),
            [
                {"n_estimators": 150, "max_depth": depth, "min_samples_leaf": leaf,
                 "max_features": feature_fraction}
                for depth in [None] for leaf in [2, 5]
                for feature_fraction in [0.7]
            ],
        ),
        "Extra trees": (
            ExtraTreesRegressor(random_state=random_state, n_jobs=1),
            [
                {"n_estimators": 150, "max_depth": depth, "min_samples_leaf": leaf,
                 "max_features": feature_fraction}
                for depth in [None] for leaf in [2, 5]
                for feature_fraction in [0.7]
            ],
        ),
        "Histogram gradient boosting": (
            HistGradientBoostingRegressor(random_state=random_state),
            [
                {"max_iter": 200, "learning_rate": rate, "max_leaf_nodes": leaves,
                 "l2_regularization": penalty}
                for rate in [0.05, 0.1] for leaves in [15, 31]
                for penalty in [1.0]
            ],
        ),
        "ANN": (
            MLPRegressor(
                random_state=random_state, max_iter=400, early_stopping=True,
                learning_rate_init=0.001,
            ),
            [
                {"hidden_layer_sizes": (32,), "alpha": 0.01}
            ],
        ),
    }


def _year_folds(data: pd.DataFrame, max_folds: int = 5):
    years = np.sort(data["target_year"].unique())
    minimum_training_years = max(2, len(years) - max_folds)
    folds = []
    for index in range(minimum_training_years, len(years)):
        validation_year = int(years[index])
        train_index = np.flatnonzero(data["target_year"].to_numpy() < validation_year)
        validation_index = np.flatnonzero(data["target_year"].to_numpy() == validation_year)
        if len(train_index) and len(validation_index):
            folds.append((train_index, validation_index, validation_year))
    if len(folds) < 2:
        raise ValueError("At least two expanding validation folds are required.")
    return folds


def _available(
    data: pd.DataFrame, numeric: Iterable[str], categorical: Iterable[str]
) -> tuple[list[str], list[str]]:
    numeric_out = [
        column for column in dict.fromkeys(numeric)
        if column in data and not data[column].isna().all()
        and data[column].nunique(dropna=True) > 1
    ]
    categorical_out = [
        column for column in dict.fromkeys(categorical)
        if column in data and not data[column].isna().all()
        and data[column].nunique(dropna=True) > 1
    ]
    return numeric_out, categorical_out


def _make_pipeline(estimator, numeric: list[str], categorical: list[str]) -> Pipeline:
    return Pipeline([
        ("preprocess", _preprocessor(numeric, categorical)),
        ("model", clone(estimator)),
    ])


def _cv_evaluate(
    data: pd.DataFrame,
    target: str,
    target_tested: str,
    numeric: list[str],
    categorical: list[str],
    estimator,
    folds,
):
    features = numeric + categorical
    predictions = []
    fold_rows = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for train_index, validation_index, validation_year in folds:
            train = data.iloc[train_index]
            validation = data.iloc[validation_index]
            model = _make_pipeline(estimator, numeric, categorical)
            model.fit(train[features], train[target])
            prediction = np.clip(model.predict(validation[features]), 0, 100)
            metrics = _metrics(
                validation[target], prediction, validation[target_tested]
            )
            fold_rows.append({"ValidationYear": validation_year, **metrics})
            predictions.append(pd.DataFrame({
                "observed": validation[target].to_numpy(),
                "prediction": prediction,
                "weight": validation[target_tested].to_numpy(),
            }))
    pooled = pd.concat(predictions, ignore_index=True)
    pooled_metrics = _metrics(pooled.observed, pooled.prediction, pooled.weight)
    fold_table = pd.DataFrame(fold_rows)
    return {
        "CV_RMSE": pooled_metrics["RMSE"],
        "CV_MAE": pooled_metrics["MAE"],
        "CV_R2": pooled_metrics["R2"],
        "CV_WeightedRMSE": pooled_metrics["WeightedRMSE"],
        "FoldRMSEMean": fold_table.RMSE.mean(),
        "FoldRMSESD": fold_table.RMSE.std(ddof=0),
        "FoldRMSE": fold_table.set_index("ValidationYear").RMSE.to_dict(),
    }


def _persistence_cv(data, target, target_tested, folds):
    frames = []
    fold_rows = []
    for _, validation_index, validation_year in folds:
        validation = data.iloc[validation_index]
        prediction = validation["resistance_pct"].to_numpy()
        metrics = _metrics(validation[target], prediction, validation[target_tested])
        fold_rows.append({"ValidationYear": validation_year, **metrics})
        frames.append(pd.DataFrame({
            "observed": validation[target].to_numpy(),
            "prediction": prediction,
            "weight": validation[target_tested].to_numpy(),
        }))
    pooled = pd.concat(frames, ignore_index=True)
    result = _metrics(pooled.observed, pooled.prediction, pooled.weight)
    fold_table = pd.DataFrame(fold_rows)
    return {
        "CV_RMSE": result["RMSE"], "CV_MAE": result["MAE"],
        "CV_R2": result["R2"], "CV_WeightedRMSE": result["WeightedRMSE"],
        "FoldRMSEMean": fold_table.RMSE.mean(),
        "FoldRMSESD": fold_table.RMSE.std(ddof=0),
        "FoldRMSE": fold_table.set_index("ValidationYear").RMSE.to_dict(),
    }


def _fold_win_rate(candidate: dict, reference: dict) -> float:
    common = sorted(set(candidate["FoldRMSE"]) & set(reference["FoldRMSE"]))
    if not common:
        return np.nan
    wins = [candidate["FoldRMSE"][year] < reference["FoldRMSE"][year] for year in common]
    return float(np.mean(wins))


def _tune_models(data, target, target_tested, numeric, categorical, folds, random_state=42):
    rows = []
    best = None
    for model_name, (base_estimator, parameters) in _model_specs(random_state).items():
        for parameter_number, params in enumerate(parameters, start=1):
            estimator = clone(base_estimator).set_params(**params)
            metrics = _cv_evaluate(
                data, target, target_tested, numeric, categorical, estimator, folds
            )
            row = {
                "Model": model_name, "ParameterSet": parameter_number,
                "Parameters": str(params),
                **{key: value for key, value in metrics.items() if key != "FoldRMSE"},
            }
            rows.append(row)
            if best is None or metrics["CV_RMSE"] < best["metrics"]["CV_RMSE"]:
                best = {
                    "name": model_name, "estimator": estimator,
                    "params": params, "metrics": metrics,
                }
    return best, pd.DataFrame(rows).sort_values("CV_RMSE").reset_index(drop=True)


@dataclass
class OptimizedHorizonResult:
    horizon: int
    selected_model_name: str
    selected_groups: list[str]
    numeric_features: list[str]
    categorical_features: list[str]
    performance: pd.DataFrame
    search_log: pd.DataFrame
    tuning_results: pd.DataFrame
    predictions: pd.DataFrame
    feature_importance: pd.DataFrame
    fitted_model: Pipeline


def optimize_horizon(
    panel: pd.DataFrame,
    horizon: int,
    output_dir: str | Path,
    minimum_improvement_pct: float = 0.5,
    minimum_fold_win_rate: float = 0.6,
    random_state: int = 42,
) -> OptimizedHorizonResult:
    target = f"target_resistance_h{horizon}"
    target_tested = f"target_tested_h{horizon}"
    data = panel.loc[panel[target].notna()].copy()
    data["target_year"] = data["year"] + horizon
    test_year = int(data.target_year.max())
    selection = data[data.target_year.lt(test_year)].reset_index(drop=True)
    test = data[data.target_year.eq(test_year)].copy()
    folds = _year_folds(selection)

    base_numeric, base_categorical = _available(
        selection, BASE_NUMERIC, BASE_CATEGORICAL
    )
    current_numeric = base_numeric.copy()
    current_categorical = base_categorical.copy()
    current_best, base_tuning = _tune_models(
        selection, target, target_tested, current_numeric, current_categorical,
        folds, random_state,
    )
    base_metrics = current_best["metrics"]
    current_metrics = base_metrics
    selected_groups = []
    log_rows = [{
        "Iteration": 0, "Stage": "Base model tuning", "Candidate": "Resistance history",
        "Accepted": True, "Model": current_best["name"],
        "FeatureCount": len(current_numeric) + len(current_categorical),
        "CV_RMSE": current_metrics["CV_RMSE"], "CV_MAE": current_metrics["CV_MAE"],
        "CV_WeightedRMSE": current_metrics["CV_WeightedRMSE"],
        "ImprovementPct": 0.0, "FoldWinRate": np.nan,
    }]

    # Stage 1: forward selection of complete, scientifically defined groups.
    remaining_groups = list(FEATURE_GROUPS)
    for iteration in range(1, len(FEATURE_GROUPS) + 1):
        candidates = []
        for group_name in remaining_groups:
            definition = FEATURE_GROUPS[group_name]
            group_numeric, group_categorical = _available(
                selection, definition["numeric"], definition["categorical"]
            )
            numeric = list(dict.fromkeys(current_numeric + group_numeric))
            categorical = list(dict.fromkeys(current_categorical + group_categorical))
            if numeric == current_numeric and categorical == current_categorical:
                continue
            metrics = _cv_evaluate(
                selection, target, target_tested, numeric, categorical,
                current_best["estimator"], folds,
            )
            improvement = 100 * (
                current_metrics["CV_RMSE"] - metrics["CV_RMSE"]
            ) / current_metrics["CV_RMSE"]
            win_rate = _fold_win_rate(metrics, current_metrics)
            row = {
                "Iteration": iteration, "Stage": "Group forward selection",
                "Candidate": group_name, "Accepted": False,
                "Model": current_best["name"],
                "FeatureCount": len(numeric) + len(categorical),
                "CV_RMSE": metrics["CV_RMSE"], "CV_MAE": metrics["CV_MAE"],
                "CV_WeightedRMSE": metrics["CV_WeightedRMSE"],
                "ImprovementPct": improvement, "FoldWinRate": win_rate,
                "numeric": numeric, "categorical": categorical, "metrics": metrics,
            }
            candidates.append(row)
        if not candidates:
            break
        best_candidate = min(candidates, key=lambda row: row["CV_RMSE"])
        accepted = (
            best_candidate["ImprovementPct"] >= minimum_improvement_pct
            and best_candidate["FoldWinRate"] >= minimum_fold_win_rate
        )
        for candidate in candidates:
            candidate["Accepted"] = accepted and candidate is best_candidate
            log_rows.append({key: value for key, value in candidate.items()
                             if key not in {"numeric", "categorical", "metrics"}})
        if not accepted:
            break
        current_numeric = best_candidate["numeric"]
        current_categorical = best_candidate["categorical"]
        current_metrics = best_candidate["metrics"]
        selected_groups.append(best_candidate["Candidate"])
        remaining_groups.remove(best_candidate["Candidate"])

    # Stage 2: one-at-a-time additions can recover a useful feature hidden in a
    # rejected group. A stricter fold-stability rule limits selection noise.
    # A pre-declared shortlist prevents an unbounded multiple-comparison search
    # while still checking the most interpretable individual variables.
    all_candidate_numeric = [
        "family_consumption_mean3", "family_consumption_trend3",
        "reserve_pct_lag1", "reserve_mean3", "ast_tested_per_100k",
        "reported_resistant_per_100k", "population_growth1",
        "eu_resistance_excl_country", "resistance_gap_to_eu",
        "other_antibiotic_mean", "country_profile_gap", "origin_year_index",
    ]
    all_candidate_categorical = ["country"]
    individual_iteration = 0
    while individual_iteration < 2:
        individual_iteration += 1
        candidates = []
        for feature in all_candidate_numeric:
            if feature in current_numeric or feature not in selection or selection[feature].isna().all():
                continue
            numeric = current_numeric + [feature]
            metrics = _cv_evaluate(
                selection, target, target_tested, numeric, current_categorical,
                current_best["estimator"], folds,
            )
            improvement = 100 * (current_metrics["CV_RMSE"] - metrics["CV_RMSE"]) / current_metrics["CV_RMSE"]
            candidates.append((feature, numeric, current_categorical, metrics, improvement, _fold_win_rate(metrics, current_metrics)))
        for feature in all_candidate_categorical:
            if feature in current_categorical or feature not in selection or selection[feature].isna().all():
                continue
            categorical = current_categorical + [feature]
            metrics = _cv_evaluate(
                selection, target, target_tested, current_numeric, categorical,
                current_best["estimator"], folds,
            )
            improvement = 100 * (current_metrics["CV_RMSE"] - metrics["CV_RMSE"]) / current_metrics["CV_RMSE"]
            candidates.append((feature, current_numeric, categorical, metrics, improvement, _fold_win_rate(metrics, current_metrics)))
        if not candidates:
            break
        best_feature = min(candidates, key=lambda item: item[3]["CV_RMSE"])
        feature, numeric, categorical, metrics, improvement, win_rate = best_feature
        accepted = improvement >= minimum_improvement_pct and win_rate >= minimum_fold_win_rate
        log_rows.append({
            "Iteration": individual_iteration, "Stage": "Individual forward selection",
            "Candidate": feature, "Accepted": accepted, "Model": current_best["name"],
            "FeatureCount": len(numeric) + len(categorical), "CV_RMSE": metrics["CV_RMSE"],
            "CV_MAE": metrics["CV_MAE"], "CV_WeightedRMSE": metrics["CV_WeightedRMSE"],
            "ImprovementPct": improvement, "FoldWinRate": win_rate,
        })
        if not accepted:
            break
        current_numeric, current_categorical, current_metrics = numeric, categorical, metrics

    # Stage 3: remove selected non-base variables when deletion is consistently
    # better. Base resistance-history variables are protected.
    backward_iteration = 0
    while backward_iteration < 2:
        backward_iteration += 1
        removable_numeric = [feature for feature in current_numeric if feature not in base_numeric]
        removable_categorical = [feature for feature in current_categorical if feature not in base_categorical]
        candidates = []
        for feature in removable_numeric:
            numeric = [item for item in current_numeric if item != feature]
            metrics = _cv_evaluate(
                selection, target, target_tested, numeric, current_categorical,
                current_best["estimator"], folds,
            )
            improvement = 100 * (current_metrics["CV_RMSE"] - metrics["CV_RMSE"]) / current_metrics["CV_RMSE"]
            candidates.append((feature, numeric, current_categorical, metrics, improvement, _fold_win_rate(metrics, current_metrics)))
        for feature in removable_categorical:
            categorical = [item for item in current_categorical if item != feature]
            metrics = _cv_evaluate(
                selection, target, target_tested, current_numeric, categorical,
                current_best["estimator"], folds,
            )
            improvement = 100 * (current_metrics["CV_RMSE"] - metrics["CV_RMSE"]) / current_metrics["CV_RMSE"]
            candidates.append((feature, current_numeric, categorical, metrics, improvement, _fold_win_rate(metrics, current_metrics)))
        if not candidates:
            break
        best_removal = min(candidates, key=lambda item: item[3]["CV_RMSE"])
        feature, numeric, categorical, metrics, improvement, win_rate = best_removal
        accepted = improvement >= minimum_improvement_pct and win_rate >= minimum_fold_win_rate
        log_rows.append({
            "Iteration": backward_iteration, "Stage": "Backward elimination",
            "Candidate": feature, "Accepted": accepted, "Model": current_best["name"],
            "FeatureCount": len(numeric) + len(categorical), "CV_RMSE": metrics["CV_RMSE"],
            "CV_MAE": metrics["CV_MAE"], "CV_WeightedRMSE": metrics["CV_WeightedRMSE"],
            "ImprovementPct": improvement, "FoldWinRate": win_rate,
        })
        if not accepted:
            break
        current_numeric, current_categorical, current_metrics = numeric, categorical, metrics

    # Retune all model families after the feature specification is frozen.
    final_best, final_tuning = _tune_models(
        selection, target, target_tested, current_numeric, current_categorical,
        folds, random_state,
    )
    final_metrics = final_best["metrics"]
    features = current_numeric + current_categorical
    final_model = _make_pipeline(final_best["estimator"], current_numeric, current_categorical)
    final_model.fit(selection[features], selection[target])
    test_prediction = np.clip(final_model.predict(test[features]), 0, 100)
    persistence_prediction = test["resistance_pct"].to_numpy()
    test_metrics = _metrics(test[target], test_prediction, test[target_tested])
    persistence_test = _metrics(test[target], persistence_prediction, test[target_tested])
    persistence_cv = _persistence_cv(selection, target, target_tested, folds)

    performance = pd.DataFrame([
        {"Horizon": horizon, "Specification": "Persistence", "Split": "Rolling CV",
         "Model": "Persistence", "RMSE": persistence_cv["CV_RMSE"],
         "MAE": persistence_cv["CV_MAE"], "R2": persistence_cv["CV_R2"],
         "WeightedRMSE": persistence_cv["CV_WeightedRMSE"]},
        {"Horizon": horizon, "Specification": "Resistance-history baseline", "Split": "Rolling CV",
         "Model": current_best["name"], "RMSE": base_metrics["CV_RMSE"],
         "MAE": base_metrics["CV_MAE"], "R2": base_metrics["CV_R2"],
         "WeightedRMSE": base_metrics["CV_WeightedRMSE"]},
        {"Horizon": horizon, "Specification": "Optimized features", "Split": "Rolling CV",
         "Model": final_best["name"], "RMSE": final_metrics["CV_RMSE"],
         "MAE": final_metrics["CV_MAE"], "R2": final_metrics["CV_R2"],
         "WeightedRMSE": final_metrics["CV_WeightedRMSE"]},
        {"Horizon": horizon, "Specification": "Persistence", "Split": "Latest-year confirmation",
         "Model": "Persistence", **persistence_test},
        {"Horizon": horizon, "Specification": "Optimized features", "Split": "Latest-year confirmation",
         "Model": final_best["name"], **test_metrics},
    ])
    predictions = test[[
        "region", "country", "iso3", "year", "target_year", "antibiotic",
        "resistance_pct", target, target_tested,
    ]].copy()
    predictions["prediction"] = test_prediction
    predictions["persistence_prediction"] = persistence_prediction
    predictions["horizon"] = horizon

    importance = permutation_importance(
        final_model, test[features], test[target], scoring="neg_root_mean_squared_error",
        n_repeats=15, random_state=random_state, n_jobs=1,
    )
    feature_importance = pd.DataFrame({
        "Feature": features, "ImportanceMean": importance.importances_mean,
        "ImportanceSD": importance.importances_std, "Horizon": horizon,
    }).sort_values("ImportanceMean", ascending=False)

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    safe_name = final_best["name"].lower().replace(" ", "_")
    joblib.dump(final_model, output / f"optimized_eu_ecoli_bsi_h{horizon}_{safe_name}.joblib")
    search_log = pd.DataFrame(log_rows)
    tuning_results = pd.concat([
        base_tuning.assign(Stage="Base feature tuning"),
        final_tuning.assign(Stage="Optimized feature retuning"),
    ], ignore_index=True)
    return OptimizedHorizonResult(
        horizon=horizon, selected_model_name=final_best["name"],
        selected_groups=selected_groups, numeric_features=current_numeric,
        categorical_features=current_categorical, performance=performance,
        search_log=search_log, tuning_results=tuning_results,
        predictions=predictions, feature_importance=feature_importance,
        fitted_model=final_model,
    )
