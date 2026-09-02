"""Research-grade validation utilities for the EU/EEA E. coli AMR model.

The functions in this module deliberately separate:

* frozen-specification backtesting, which measures retrospective robustness;
* nested backtesting, which re-runs selection inside every outer time split;
* uncertainty for the model-minus-persistence comparison;
* prequential conformal intervals; and
* a penalised country random-intercept benchmark for partial pooling.

All splits use ``target_year``. No row with a target at or after an evaluation
year is used to fit the corresponding forecast.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Iterable

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin, clone

from src.iterative_feature_search import (
    BASE_CATEGORICAL,
    BASE_NUMERIC,
    FEATURE_GROUPS,
    _available,
    _cv_evaluate,
    _make_pipeline,
    _model_specs,
    _year_folds,
    optimize_horizon,
)
from src.regional_amr import _metrics


RANDOM_STATE = 42


def prepare_horizon(panel: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """Return the labelled panel for one direct forecast horizon."""
    target = f"target_resistance_h{horizon}"
    target_tested = f"target_tested_h{horizon}"
    required = {target, target_tested, "year", "resistance_pct", "iso3", "antibiotic"}
    missing = sorted(required.difference(panel.columns))
    if missing:
        raise ValueError(f"Panel is missing required columns: {missing}")
    data = panel.loc[panel[target].notna() & panel[target_tested].notna()].copy()
    data["target_year"] = data["year"].astype(int) + int(horizon)
    return data.sort_values(["target_year", "iso3", "antibiotic"]).reset_index(drop=True)


def load_frozen_specification(
    project_root: str | Path, horizon: int
) -> tuple[BaseEstimator, list[str], list[str]]:
    """Load the previously selected model and its explicit feature schema."""
    root = Path(project_root)
    feature_file = root / "outputs/optimization/optimized_selected_features.csv"
    table = pd.read_csv(feature_file)
    selected = table.loc[table["Horizon"].eq(horizon)]
    if selected.empty:
        raise ValueError(f"No frozen feature specification for horizon {horizon}.")
    numeric = selected.loc[selected["Type"].eq("numeric"), "Feature"].tolist()
    categorical = selected.loc[selected["Type"].eq("categorical"), "Feature"].tolist()
    candidates = sorted((root / "outputs/optimization").glob(
        f"optimized_eu_ecoli_bsi_h{horizon}_*.joblib"
    ))
    if len(candidates) != 1:
        raise ValueError(
            f"Expected one optimized model for horizon {horizon}; found {len(candidates)}."
        )
    return joblib.load(candidates[0]), numeric, categorical


def eligible_outer_years(
    data: pd.DataFrame,
    minimum_training_target_years: int = 5,
    max_origins: int | None = None,
    minimum_outer_origins: int = 2,
) -> list[int]:
    years = sorted(int(value) for value in data["target_year"].unique())
    eligible = years[minimum_training_target_years:]
    if max_origins is not None:
        eligible = eligible[-max_origins:]
    if len(eligible) < minimum_outer_origins:
        raise ValueError(
            f"At least {minimum_outer_origins} eligible outer target years are required."
        )
    return eligible


def _prediction_frame(
    test: pd.DataFrame,
    target: str,
    target_tested: str,
    prediction: np.ndarray,
    horizon: int,
    evaluation: str,
) -> pd.DataFrame:
    columns = [
        "country", "iso3", "year", "target_year", "antibiotic",
        "resistance_pct", target, target_tested,
    ]
    available = [column for column in columns if column in test]
    result = test[available].copy()
    result = result.rename(columns={target: "observed", target_tested: "target_tested"})
    result["prediction"] = np.clip(np.asarray(prediction, dtype=float), 0, 100)
    result["persistence_prediction"] = np.clip(
        result["resistance_pct"].to_numpy(dtype=float), 0, 100
    )
    result["model_error"] = result["observed"] - result["prediction"]
    result["persistence_error"] = (
        result["observed"] - result["persistence_prediction"]
    )
    result["horizon"] = horizon
    result["evaluation"] = evaluation
    return result


def summarize_backtest(predictions: pd.DataFrame) -> pd.DataFrame:
    """Calculate origin-level and pooled performance against persistence."""
    rows: list[dict] = []
    groups: list[tuple[object, pd.DataFrame]] = list(predictions.groupby("target_year"))
    groups.append(("Pooled", predictions))
    for target_year, frame in groups:
        model = _metrics(frame["observed"], frame["prediction"], frame["target_tested"])
        persistence = _metrics(
            frame["observed"], frame["persistence_prediction"], frame["target_tested"]
        )
        skill = 1 - model["RMSE"] / persistence["RMSE"] if persistence["RMSE"] else np.nan
        weighted_skill = (
            1 - model["WeightedRMSE"] / persistence["WeightedRMSE"]
            if persistence["WeightedRMSE"] else np.nan
        )
        rows.append({
            "TargetYear": target_year,
            "N": len(frame),
            "ModelRMSE": model["RMSE"],
            "PersistenceRMSE": persistence["RMSE"],
            "RMSEDifference": model["RMSE"] - persistence["RMSE"],
            "Skill": skill,
            "ModelMAE": model["MAE"],
            "PersistenceMAE": persistence["MAE"],
            "ModelWeightedRMSE": model["WeightedRMSE"],
            "PersistenceWeightedRMSE": persistence["WeightedRMSE"],
            "WeightedSkill": weighted_skill,
            "ModelR2": model["R2"],
            "ModelWins": bool(model["RMSE"] < persistence["RMSE"]),
        })
    return pd.DataFrame(rows)


def frozen_specification_backtest(
    panel: pd.DataFrame,
    horizon: int,
    frozen_model: BaseEstimator,
    numeric_features: Iterable[str],
    categorical_features: Iterable[str],
    minimum_training_target_years: int = 5,
    max_origins: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Refit a frozen model specification at several historical cutoffs.

    Because the specification was chosen using the complete development era,
    this is a robustness analysis, not an unbiased estimate of the adaptive
    model-development pipeline.
    """
    data = prepare_horizon(panel, horizon)
    numeric, categorical = _available(
        data, list(numeric_features), list(categorical_features)
    )
    features = numeric + categorical
    outer_years = eligible_outer_years(
        data, minimum_training_target_years, max_origins
    )
    frames = []
    for target_year in outer_years:
        train = data.loc[data["target_year"].lt(target_year)]
        test = data.loc[data["target_year"].eq(target_year)]
        model = clone(frozen_model)
        model.fit(train[features], train[f"target_resistance_h{horizon}"])
        prediction = model.predict(test[features])
        frames.append(_prediction_frame(
            test, f"target_resistance_h{horizon}", f"target_tested_h{horizon}",
            prediction, horizon, "Frozen specification",
        ))
    predictions = pd.concat(frames, ignore_index=True)
    return predictions, summarize_backtest(predictions)


def nested_pipeline_backtest(
    panel: pd.DataFrame,
    horizon: int,
    max_origins: int = 3,
    minimum_training_target_years: int = 5,
    random_state: int = RANDOM_STATE,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Re-run the complete bounded search inside every outer time split."""
    data = prepare_horizon(panel, horizon)
    outer_years = eligible_outer_years(
        data, minimum_training_target_years, max_origins, minimum_outer_origins=1
    )
    target = f"target_resistance_h{horizon}"
    target_tested = f"target_tested_h{horizon}"
    frames = []
    specifications = []
    with TemporaryDirectory(prefix=f"amr_nested_h{horizon}_") as temporary:
        for outer_year in outer_years:
            masked = panel.copy()
            future = masked["year"].astype(int).add(horizon).gt(outer_year)
            masked.loc[future, [target, target_tested]] = np.nan
            result = optimize_horizon(
                masked, horizon, Path(temporary) / str(outer_year),
                random_state=random_state,
            )
            prediction = result.predictions.rename(columns={
                target: "observed", target_tested: "target_tested"
            }).copy()
            prediction["model_error"] = prediction["observed"] - prediction["prediction"]
            prediction["persistence_error"] = (
                prediction["observed"] - prediction["persistence_prediction"]
            )
            prediction["evaluation"] = "Nested pipeline"
            frames.append(prediction)
            specifications.append({
                "Horizon": horizon,
                "OuterTargetYear": outer_year,
                "SelectedModel": result.selected_model_name,
                "SelectedGroups": " | ".join(result.selected_groups),
                "NumericFeatures": " | ".join(result.numeric_features),
                "CategoricalFeatures": " | ".join(result.categorical_features),
                "FeatureCount": len(result.numeric_features) + len(result.categorical_features),
            })
    predictions = pd.concat(frames, ignore_index=True)
    return predictions, summarize_backtest(predictions), pd.DataFrame(specifications)


def _weighted_rmse(error: np.ndarray, weight: np.ndarray) -> float:
    valid = np.isfinite(error) & np.isfinite(weight) & (weight > 0)
    if not valid.any():
        return np.nan
    return float(np.sqrt(np.average(np.square(error[valid]), weights=weight[valid])))


def two_way_cluster_bootstrap(
    predictions: pd.DataFrame,
    n_bootstrap: int = 2000,
    random_state: int = RANDOM_STATE,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Bootstrap country and target-year clusters with pigeonhole weights."""
    required = {"iso3", "target_year", "model_error", "persistence_error", "target_tested"}
    if missing := required.difference(predictions.columns):
        raise ValueError(f"Predictions are missing bootstrap columns: {sorted(missing)}")
    rng = np.random.default_rng(random_state)
    countries = predictions["iso3"].dropna().unique()
    years = predictions["target_year"].dropna().unique()
    country_codes = pd.Categorical(predictions["iso3"], categories=countries).codes
    year_codes = pd.Categorical(predictions["target_year"], categories=years).codes
    rows = []
    for replicate in range(n_bootstrap):
        country_counts = np.bincount(
            rng.integers(0, len(countries), len(countries)), minlength=len(countries)
        )
        year_counts = np.bincount(
            rng.integers(0, len(years), len(years)), minlength=len(years)
        )
        frequency = country_counts[country_codes] * year_counts[year_codes]
        valid = frequency.gt(0) if isinstance(frequency, pd.Series) else frequency > 0
        if not np.any(valid):
            continue
        ordinary_weight = frequency[valid].astype(float)
        tested_weight = ordinary_weight * predictions["target_tested"].to_numpy()[valid]
        model_error = predictions["model_error"].to_numpy()[valid]
        persistence_error = predictions["persistence_error"].to_numpy()[valid]
        model_rmse = _weighted_rmse(model_error, ordinary_weight)
        persistence_rmse = _weighted_rmse(persistence_error, ordinary_weight)
        model_weighted = _weighted_rmse(model_error, tested_weight)
        persistence_weighted = _weighted_rmse(persistence_error, tested_weight)
        rows.append({
            "Replicate": replicate,
            "RMSEDifference": model_rmse - persistence_rmse,
            "Skill": 1 - model_rmse / persistence_rmse if persistence_rmse else np.nan,
            "WeightedRMSEDifference": model_weighted - persistence_weighted,
            "WeightedSkill": (
                1 - model_weighted / persistence_weighted if persistence_weighted else np.nan
            ),
        })
    distribution = pd.DataFrame(rows)
    summary_rows = []
    for metric in ["RMSEDifference", "Skill", "WeightedRMSEDifference", "WeightedSkill"]:
        values = distribution[metric].dropna()
        summary_rows.append({
            "Metric": metric,
            "Estimate": values.mean(),
            "Lower95": values.quantile(0.025),
            "Upper95": values.quantile(0.975),
            "ProbabilityModelWorse": (
                float((values >= 0).mean()) if "Difference" in metric
                else float((values <= 0).mean())
            ),
            "BootstrapReplicates": len(values),
        })
    return distribution, pd.DataFrame(summary_rows)


def _weighted_quantile(values: np.ndarray, quantile: float, weights: np.ndarray) -> float:
    order = np.argsort(values)
    values = values[order]
    weights = weights[order]
    cumulative = np.cumsum(weights)
    cutoff = quantile * cumulative[-1]
    return float(values[min(np.searchsorted(cumulative, cutoff, side="left"), len(values) - 1)])


def prequential_conformal_intervals(
    predictions: pd.DataFrame,
    alphas: tuple[float, ...] = (0.20, 0.05),
    minimum_calibration_rows: int = 80,
    recency_decay: float = 0.90,
    antibiotic_specific_minimum: int = 35,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Add intervals using only earlier rolling-origin forecast errors."""
    output = predictions.sort_values(["target_year", "iso3", "antibiotic"]).copy()
    years = sorted(output["target_year"].unique())
    for alpha in alphas:
        level = int(round(100 * (1 - alpha)))
        output[f"lower_{level}"] = np.nan
        output[f"upper_{level}"] = np.nan
        output[f"interval_source_{level}"] = "Unavailable"
    for year in years:
        historical = output.loc[output["target_year"].lt(year)].copy()
        current_index = output.index[output["target_year"].eq(year)]
        if len(historical) < minimum_calibration_rows:
            continue
        historical["absolute_error"] = historical["model_error"].abs()
        age = year - historical["target_year"].to_numpy(dtype=float)
        historical["recency_weight"] = np.power(recency_decay, np.maximum(age - 1, 0))
        for index in current_index:
            antibiotic = output.at[index, "antibiotic"]
            group = historical.loc[historical["antibiotic"].eq(antibiotic)]
            if len(group) >= antibiotic_specific_minimum:
                calibration = group
                source = "Antibiotic-specific earlier origins"
            else:
                calibration = historical
                source = "Pooled earlier origins"
            errors = calibration["absolute_error"].to_numpy(dtype=float)
            weights = calibration["recency_weight"].to_numpy(dtype=float)
            for alpha in alphas:
                level = int(round(100 * (1 - alpha)))
                finite_quantile = min(1.0, np.ceil((len(errors) + 1) * (1 - alpha)) / len(errors))
                radius = _weighted_quantile(errors, finite_quantile, weights)
                prediction = float(output.at[index, "prediction"])
                output.at[index, f"lower_{level}"] = max(0.0, prediction - radius)
                output.at[index, f"upper_{level}"] = min(100.0, prediction + radius)
                output.at[index, f"interval_source_{level}"] = source
    summary = []
    for alpha in alphas:
        level = int(round(100 * (1 - alpha)))
        valid = output[f"lower_{level}"].notna()
        covered = (
            output.loc[valid, "observed"].ge(output.loc[valid, f"lower_{level}"])
            & output.loc[valid, "observed"].le(output.loc[valid, f"upper_{level}"])
        )
        width = output.loc[valid, f"upper_{level}"] - output.loc[valid, f"lower_{level}"]
        summary.append({
            "NominalCoverage": 1 - alpha,
            "EmpiricalCoverage": covered.mean() if len(covered) else np.nan,
            "AverageWidth": width.mean() if len(width) else np.nan,
            "MedianWidth": width.median() if len(width) else np.nan,
            "EvaluatedRows": int(valid.sum()),
            "EvaluatedTargetYears": int(output.loc[valid, "target_year"].nunique()),
        })
    return output, pd.DataFrame(summary)


class PenalizedCountryInterceptRegressor(BaseEstimator, RegressorMixin):
    """Base model plus a shrinkage-estimated country random intercept.

    The group effect is ``sum(residual) / (n + shrinkage)``. Small-country
    effects are therefore pulled more strongly toward the European mean, and an
    unseen country receives the mean effect of zero.
    """

    def __init__(self, base_estimator: BaseEstimator, shrinkage: float = 10.0,
                 group_column: str = "country"):
        self.base_estimator = base_estimator
        self.shrinkage = shrinkage
        self.group_column = group_column

    def fit(self, X: pd.DataFrame, y: pd.Series):
        if self.group_column not in X:
            raise ValueError(f"Missing grouping column {self.group_column!r}.")
        self.base_estimator_ = clone(self.base_estimator)
        base_X = X.drop(columns=[self.group_column])
        self.base_estimator_.fit(base_X, y)
        residual = np.asarray(y, dtype=float) - self.base_estimator_.predict(base_X)
        table = pd.DataFrame({"group": X[self.group_column].astype(str), "residual": residual})
        summary = table.groupby("group")["residual"].agg(["sum", "count"])
        self.group_effects_ = (
            summary["sum"] / (summary["count"] + float(self.shrinkage))
        ).to_dict()
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        base = self.base_estimator_.predict(X.drop(columns=[self.group_column]))
        effect = X[self.group_column].astype(str).map(self.group_effects_).fillna(0).to_numpy()
        return np.clip(np.asarray(base, dtype=float) + effect, 0, 100)


def partial_pooling_backtest(
    panel: pd.DataFrame,
    horizon: int,
    estimator: BaseEstimator,
    numeric_features: Iterable[str],
    categorical_features: Iterable[str],
    shrinkage_grid: tuple[float, ...] = (2.0, 5.0, 10.0, 20.0, 50.0),
    minimum_training_target_years: int = 5,
    max_origins: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Nested tuning and evaluation of a penalised country intercept."""
    data = prepare_horizon(panel, horizon)
    numeric, categorical = _available(
        data, list(numeric_features), [c for c in categorical_features if c != "country"]
    )
    base_estimator = clone(estimator.named_steps["model"] if hasattr(estimator, "named_steps") else estimator)
    base_pipeline = _make_pipeline(base_estimator, numeric, categorical)
    features = numeric + categorical + ["country"]
    outer_years = eligible_outer_years(data, minimum_training_target_years, max_origins)
    prediction_frames = []
    tuning_rows = []
    target = f"target_resistance_h{horizon}"
    target_tested = f"target_tested_h{horizon}"
    for outer_year in outer_years:
        outer_train = data.loc[data["target_year"].lt(outer_year)].copy()
        outer_test = data.loc[data["target_year"].eq(outer_year)].copy()
        inner_year = int(outer_train["target_year"].max())
        inner_train = outer_train.loc[outer_train["target_year"].lt(inner_year)]
        inner_valid = outer_train.loc[outer_train["target_year"].eq(inner_year)]
        best = None
        for shrinkage in shrinkage_grid:
            model = PenalizedCountryInterceptRegressor(base_pipeline, shrinkage)
            model.fit(inner_train[features], inner_train[target])
            prediction = model.predict(inner_valid[features])
            metrics = _metrics(inner_valid[target], prediction, inner_valid[target_tested])
            tuning_rows.append({
                "Horizon": horizon, "OuterTargetYear": outer_year,
                "InnerValidationYear": inner_year, "Shrinkage": shrinkage, **metrics,
            })
            if best is None or metrics["RMSE"] < best["RMSE"]:
                best = {"shrinkage": shrinkage, **metrics}
        final = PenalizedCountryInterceptRegressor(base_pipeline, best["shrinkage"])
        final.fit(outer_train[features], outer_train[target])
        prediction_frames.append(_prediction_frame(
            outer_test, target, target_tested, final.predict(outer_test[features]),
            horizon, "Penalized country partial pooling",
        ))
    predictions = pd.concat(prediction_frames, ignore_index=True)
    tuning = pd.DataFrame(tuning_rows)
    return predictions, summarize_backtest(predictions), tuning


def _benjamini_hochberg(p_values: pd.Series) -> pd.Series:
    values = p_values.to_numpy(dtype=float)
    order = np.argsort(values)
    ranked = values[order]
    adjusted = ranked * len(values) / np.arange(1, len(values) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    output = np.empty_like(adjusted)
    output[order] = np.clip(adjusted, 0, 1)
    return pd.Series(output, index=p_values.index)


def _candidate_numeric_features(data: pd.DataFrame, base: list[str]) -> list[str]:
    candidates = []
    for definition in FEATURE_GROUPS.values():
        candidates.extend(definition["numeric"])
    return [
        feature for feature in dict.fromkeys(candidates)
        if feature not in base and feature in data and data[feature].nunique(dropna=True) > 1
    ]


def _circular_residual_null(
    data: pd.DataFrame, target: str, fitted: np.ndarray, rng: np.random.Generator
) -> np.ndarray:
    null = np.asarray(fitted, dtype=float).copy()
    residual = data[target].to_numpy(dtype=float) - fitted
    for indices in data.groupby(["iso3", "antibiotic"], sort=False).indices.values():
        indices = np.asarray(indices)
        if len(indices) < 2:
            null[indices] += residual[indices]
            continue
        shift = int(rng.integers(1, len(indices)))
        null[indices] += np.roll(residual[indices], shift)
    return np.clip(null, 0, 100)


def multiplicity_controlled_feature_screen(
    panel: pd.DataFrame,
    horizon: int,
    estimator: BaseEstimator,
    n_null_repetitions: int = 50,
    n_stability_bootstrap: int = 1000,
    random_state: int = RANDOM_STATE,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Screen extra features with temporal stability, BH and max-null control.

    This is supplementary to nested pipeline evaluation. The circular residual
    null preserves the fitted resistance-history signal and within-series
    residual structure better than shuffling individual rows.
    """
    rng = np.random.default_rng(random_state)
    data = prepare_horizon(panel, horizon)
    latest = int(data["target_year"].max())
    selection = data.loc[data["target_year"].lt(latest)].reset_index(drop=True)
    target = f"target_resistance_h{horizon}"
    target_tested = f"target_tested_h{horizon}"
    folds = _year_folds(selection)
    base_numeric, base_categorical = _available(selection, BASE_NUMERIC, BASE_CATEGORICAL)
    candidate_features = _candidate_numeric_features(selection, base_numeric)
    model_estimator = clone(
        estimator.named_steps["model"] if hasattr(estimator, "named_steps") else estimator
    )
    baseline = _cv_evaluate(
        selection, target, target_tested, base_numeric, base_categorical,
        model_estimator, folds,
    )
    observed_rows = []
    observed_fold_improvements: dict[str, list[float]] = {}
    for feature in candidate_features:
        metrics = _cv_evaluate(
            selection, target, target_tested, base_numeric + [feature],
            base_categorical, model_estimator, folds,
        )
        common = sorted(set(baseline["FoldRMSE"]) & set(metrics["FoldRMSE"]))
        fold_improvement = [
            100 * (baseline["FoldRMSE"][year] - metrics["FoldRMSE"][year])
            / baseline["FoldRMSE"][year] for year in common
        ]
        observed_fold_improvements[feature] = fold_improvement
        boot_selected = []
        for _ in range(n_stability_bootstrap):
            sampled = rng.choice(fold_improvement, size=len(fold_improvement), replace=True)
            boot_selected.append(
                np.mean(sampled) >= 0.5 and np.mean(np.asarray(sampled) > 0) >= 0.6
            )
        observed_rows.append({
            "Feature": feature,
            "ObservedImprovementPct": 100 * (
                baseline["CV_RMSE"] - metrics["CV_RMSE"]
            ) / baseline["CV_RMSE"],
            "FoldWinRate": np.mean(np.asarray(fold_improvement) > 0),
            "StabilitySelectionProbability": np.mean(boot_selected),
        })
    observed = pd.DataFrame(observed_rows)

    base_pipeline = _make_pipeline(model_estimator, base_numeric, base_categorical)
    base_pipeline.fit(selection[base_numeric + base_categorical], selection[target])
    fitted = base_pipeline.predict(selection[base_numeric + base_categorical])
    null_rows = []
    for repetition in range(n_null_repetitions):
        null_data = selection.copy()
        null_data["null_target"] = _circular_residual_null(
            selection, target, fitted, rng
        )
        null_baseline = _cv_evaluate(
            null_data, "null_target", target_tested, base_numeric, base_categorical,
            model_estimator, folds,
        )
        repetition_rows = []
        for feature in candidate_features:
            metrics = _cv_evaluate(
                null_data, "null_target", target_tested, base_numeric + [feature],
                base_categorical, model_estimator, folds,
            )
            improvement = 100 * (
                null_baseline["CV_RMSE"] - metrics["CV_RMSE"]
            ) / null_baseline["CV_RMSE"]
            repetition_rows.append((feature, improvement))
        maximum = max(value for _, value in repetition_rows)
        for feature, improvement in repetition_rows:
            null_rows.append({
                "Repetition": repetition, "Feature": feature,
                "NullImprovementPct": improvement,
                "NullMaximumImprovementPct": maximum,
            })
    null_distribution = pd.DataFrame(null_rows)
    p_values = []
    max_p_values = []
    for row in observed.itertuples():
        feature_null = null_distribution.loc[
            null_distribution["Feature"].eq(row.Feature), "NullImprovementPct"
        ]
        maxima = null_distribution.drop_duplicates("Repetition")["NullMaximumImprovementPct"]
        p_values.append((1 + (feature_null >= row.ObservedImprovementPct).sum()) / (1 + len(feature_null)))
        max_p_values.append((1 + (maxima >= row.ObservedImprovementPct).sum()) / (1 + len(maxima)))
    observed["RawPermutationP"] = p_values
    observed["BH_QValue"] = _benjamini_hochberg(observed["RawPermutationP"])
    observed["MaxNullAdjustedP"] = max_p_values
    observed["RetainAfterMultiplicityControl"] = (
        observed["ObservedImprovementPct"].ge(0.5)
        & observed["FoldWinRate"].ge(0.6)
        & observed["StabilitySelectionProbability"].ge(0.7)
        & observed["BH_QValue"].le(0.10)
        & observed["MaxNullAdjustedP"].le(0.10)
    )
    return observed.sort_values("ObservedImprovementPct", ascending=False), null_distribution
