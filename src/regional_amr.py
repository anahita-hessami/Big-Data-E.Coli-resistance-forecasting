"""Utilities for the regional bloodstream E. coli AMR forecasting project.

The module deliberately keeps EUCAST and CLSI/FDA outcomes in separate regional
datasets. It never converts aggregated resistance percentages between standards.
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
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import BaseCrossValidator, GridSearchCV
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


EARS_REQUIRED = {
    "region", "country", "iso3", "year", "pathogen", "specimen",
    "antibiotic", "resistant", "tested", "ast_standard", "ast_version",
}
CONSUMPTION_REQUIRED = {
    "country", "iso3", "year", "sector", "antibiotic_family",
    "ddd_per_1000_per_day",
}
DEMOGRAPHIC_REQUIRED = {"country", "iso3", "year", "population", "age65_pct"}
MAPPING_REQUIRED = {"antibiotic", "antibiotic_family"}


def _require_columns(df: pd.DataFrame, required: set[str], name: str) -> None:
    missing = sorted(required.difference(df.columns))
    if missing:
        raise ValueError(f"{name} is missing required columns: {missing}")


def read_csv(path: str | Path, required: set[str], name: str) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Missing {name}: {path}")
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip().str.lower()
    _require_columns(df, required, name)
    return df


def validate_ears(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Validate a harmonised EARS-Net country-year extract.

    Returns the cleaned data and an audit table. Public aggregate data are kept
    under their native reported breakpoint definition.
    """
    _require_columns(df, EARS_REQUIRED, "EARS-Net data")
    out = df.copy()
    text_cols = [
        "region", "country", "iso3", "pathogen", "specimen", "antibiotic",
        "ast_standard", "ast_version",
    ]
    for col in text_cols:
        out[col] = out[col].astype(str).str.strip()
    out["iso3"] = out["iso3"].str.upper()
    out["year"] = pd.to_numeric(out["year"], errors="raise").astype(int)
    for col in ["resistant", "tested"]:
        out[col] = pd.to_numeric(out[col], errors="raise")
    if (out["tested"] <= 0).any():
        raise ValueError("All tested denominators must be greater than zero.")
    if ((out["resistant"] < 0) | (out["resistant"] > out["tested"])).any():
        raise ValueError("Resistant counts must lie between zero and tested.")
    if not out["pathogen"].str.contains("coli", case=False, na=False).all():
        raise ValueError("The primary dataset must contain E. coli only.")
    if not out["specimen"].str.contains("blood|invasive", case=False, na=False).all():
        raise ValueError("The primary dataset must contain blood/invasive isolates only.")

    out["resistance_pct"] = 100 * out["resistant"] / out["tested"]
    keys = ["iso3", "year", "antibiotic"]
    duplicated = out.duplicated(keys, keep=False)
    if duplicated.any():
        sample = out.loc[duplicated, keys].head().to_dict("records")
        raise ValueError(f"Duplicate country-year-antibiotic rows found: {sample}")

    post_2020_non_eucast = (
        out["year"].ge(2020)
        & ~out["ast_standard"].str.upper().eq("EUCAST")
    )
    if post_2020_non_eucast.any():
        warnings.warn(
            f"{int(post_2020_non_eucast.sum())} EU rows from 2020 onward are not marked EUCAST; "
            "review the source and do not silently pool them.",
            stacklevel=2,
        )

    out = out.sort_values(keys).reset_index(drop=True)
    prior_version = out.groupby(["iso3", "antibiotic"])["ast_version"].shift(1)
    out["breakpoint_change_from_prior"] = (
        prior_version.notna() & out["ast_version"].ne(prior_version)
    ).astype(int)
    audit = pd.DataFrame({
        "metric": [
            "rows", "countries", "years", "antibiotics", "minimum_year",
            "maximum_year", "median_tested", "breakpoint_version_changes",
            "post_2020_non_eucast_rows",
        ],
        "value": [
            len(out), out["iso3"].nunique(), out["year"].nunique(),
            out["antibiotic"].nunique(), out["year"].min(), out["year"].max(),
            float(out["tested"].median()), int(out["breakpoint_change_from_prior"].sum()),
            int(post_2020_non_eucast.sum()),
        ],
    })
    return out, audit


def validate_consumption(df: pd.DataFrame) -> pd.DataFrame:
    _require_columns(df, CONSUMPTION_REQUIRED, "ESAC-Net consumption data")
    out = df.copy()
    out["iso3"] = out["iso3"].astype(str).str.upper().str.strip()
    out["year"] = pd.to_numeric(out["year"], errors="raise").astype(int)
    out["sector"] = out["sector"].astype(str).str.lower().str.strip()
    out["antibiotic_family"] = out["antibiotic_family"].astype(str).str.strip()
    out["ddd_per_1000_per_day"] = pd.to_numeric(
        out["ddd_per_1000_per_day"], errors="raise"
    )
    if (out["ddd_per_1000_per_day"] < 0).any():
        raise ValueError("Consumption cannot be negative.")
    key = ["iso3", "year", "sector", "antibiotic_family"]
    if out.duplicated(key).any():
        raise ValueError(f"Duplicate ESAC-Net rows found for key {key}.")
    for col in [
        "access_pct", "broad_narrow_ratio", "oral_parenteral_ratio",
        "reserve_pct",
    ]:
        if col not in out:
            out[col] = np.nan
        out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def validate_demographics(df: pd.DataFrame) -> pd.DataFrame:
    _require_columns(df, DEMOGRAPHIC_REQUIRED, "demographic data")
    out = df.copy()
    out["iso3"] = out["iso3"].astype(str).str.upper().str.strip()
    out["year"] = pd.to_numeric(out["year"], errors="raise").astype(int)
    numeric = [
        "population", "age65_pct", "urban_pct", "gdp_per_capita",
        "health_expenditure_pct_gdp",
    ]
    for col in numeric:
        if col not in out:
            out[col] = np.nan
        out[col] = pd.to_numeric(out[col], errors="coerce")
    if out.duplicated(["iso3", "year"]).any():
        raise ValueError("Demographic data must have one row per country-year.")
    return out


def _slope(values: Iterable[float]) -> float:
    arr = np.asarray(list(values), dtype=float)
    ok = np.isfinite(arr)
    if ok.sum() < 2:
        return np.nan
    x = np.arange(len(arr))[ok]
    return float(np.polyfit(x, arr[ok], 1)[0])


def build_eu_panel(
    ears: pd.DataFrame,
    consumption: pd.DataFrame,
    demographics: pd.DataFrame,
    antibiotic_mapping: pd.DataFrame,
    horizons: Iterable[int] = (1, 3, 5),
) -> pd.DataFrame:
    """Create leakage-safe country-antibiotic-year features and direct targets."""
    _require_columns(antibiotic_mapping, MAPPING_REQUIRED, "antibiotic mapping")
    mapping = antibiotic_mapping[list(MAPPING_REQUIRED)].drop_duplicates()
    if mapping.duplicated("antibiotic").any():
        raise ValueError("Each AMR antibiotic must map to one consumption family.")

    panel = ears.merge(mapping, on="antibiotic", how="left", validate="many_to_one")
    if panel["antibiotic_family"].isna().any():
        missing = sorted(panel.loc[panel["antibiotic_family"].isna(), "antibiotic"].unique())
        raise ValueError(f"Missing consumption-family mappings for: {missing}")

    family = (
        consumption.groupby(["iso3", "year", "antibiotic_family"], as_index=False)
        .agg(family_consumption=("ddd_per_1000_per_day", "sum"))
        .sort_values(["iso3", "antibiotic_family", "year"])
    )
    for lag in (1, 2, 3):
        family[f"family_consumption_lag{lag}"] = family.groupby(
            ["iso3", "antibiotic_family"]
        )["family_consumption"].shift(lag)

    sector = (
        consumption.groupby(["iso3", "year", "sector"], as_index=False)
        .agg(sector_consumption=("ddd_per_1000_per_day", "sum"))
        .pivot(index=["iso3", "year"], columns="sector", values="sector_consumption")
        .add_prefix("consumption_")
        .reset_index()
    )
    behaviour = (
        consumption.groupby(["iso3", "year"], as_index=False)
        .agg(
            access_pct=("access_pct", "mean"),
            broad_narrow_ratio=("broad_narrow_ratio", "mean"),
            oral_parenteral_ratio=("oral_parenteral_ratio", "mean"),
            reserve_pct=("reserve_pct", "mean"),
        )
        .sort_values(["iso3", "year"])
    )
    for col in [
        "access_pct", "broad_narrow_ratio", "oral_parenteral_ratio",
        "reserve_pct",
    ]:
        behaviour[f"{col}_lag1"] = behaviour.groupby("iso3")[col].shift(1)

    panel = panel.merge(
        family, on=["iso3", "year", "antibiotic_family"], how="left", validate="many_to_one"
    )
    panel = panel.merge(sector, on=["iso3", "year"], how="left", validate="many_to_one")
    panel = panel.merge(behaviour, on=["iso3", "year"], how="left", validate="many_to_one")
    panel = panel.merge(
        demographics.drop(columns=["country"], errors="ignore"),
        on=["iso3", "year"], how="left", validate="many_to_one",
    )

    panel = panel.sort_values(["iso3", "antibiotic", "year"]).reset_index(drop=True)
    groups = panel.groupby(["iso3", "antibiotic"], group_keys=False)
    for lag in (1, 2, 3):
        panel[f"resistance_lag{lag}"] = groups["resistance_pct"].shift(lag)
    panel["resistance_trend4"] = groups["resistance_pct"].transform(
        lambda s: s.rolling(4, min_periods=2).apply(_slope, raw=False)
    )
    panel["resistance_volatility4"] = groups["resistance_pct"].transform(
        lambda s: s.rolling(4, min_periods=2).std(ddof=0)
    )
    panel["log_tested"] = np.log1p(panel["tested"])

    target_source = panel[["iso3", "antibiotic", "year", "resistance_pct", "tested"]].copy()
    for horizon in horizons:
        future = target_source.rename(columns={
            "year": "target_year",
            "resistance_pct": f"target_resistance_h{horizon}",
            "tested": f"target_tested_h{horizon}",
        })
        merge_year = panel["year"] + int(horizon)
        panel = panel.assign(_target_year=merge_year).merge(
            future,
            left_on=["iso3", "antibiotic", "_target_year"],
            right_on=["iso3", "antibiotic", "target_year"],
            how="left",
            validate="many_to_one",
        ).drop(columns=["_target_year", "target_year"])
    return panel


class ExpandingYearSplit(BaseCrossValidator):
    """Expanding-window cross-validation using target year supplied as groups."""

    def __init__(self, min_train_years: int = 3):
        self.min_train_years = min_train_years

    def split(self, X, y=None, groups=None):
        if groups is None:
            raise ValueError("Target years must be passed as groups.")
        years = np.asarray(groups, dtype=int)
        unique = np.sort(np.unique(years))
        for i in range(self.min_train_years, len(unique)):
            train_years = unique[:i]
            valid_year = unique[i]
            train_idx = np.flatnonzero(np.isin(years, train_years))
            valid_idx = np.flatnonzero(years == valid_year)
            if len(train_idx) and len(valid_idx):
                yield train_idx, valid_idx

    def get_n_splits(self, X=None, y=None, groups=None):
        if groups is None:
            return 0
        return max(0, len(np.unique(groups)) - self.min_train_years)


NUMERIC_FEATURES = [
    "resistance_pct", "resistance_lag1", "resistance_lag2", "resistance_lag3",
    "resistance_trend4", "resistance_volatility4", "log_tested",
    "family_consumption_lag1", "family_consumption_lag2", "family_consumption_lag3",
    "consumption_community", "consumption_hospital", "access_pct_lag1",
    "broad_narrow_ratio_lag1", "oral_parenteral_ratio_lag1", "reserve_pct_lag1",
    "population",
    "age65_pct", "urban_pct", "gdp_per_capita", "health_expenditure_pct_gdp",
    "breakpoint_change_from_prior",
]
CATEGORICAL_FEATURES = ["antibiotic", "antibiotic_family", "ast_standard", "ast_version"]

BASE_NUMERIC_FEATURES = [
    "resistance_pct", "resistance_lag1", "resistance_lag2", "resistance_lag3",
    "resistance_trend4", "resistance_volatility4", "log_tested",
    "breakpoint_change_from_prior",
]
CONSUMPTION_FEATURES = [
    "family_consumption_lag1", "family_consumption_lag2", "family_consumption_lag3",
    "consumption_community", "consumption_hospital",
]
PRESCRIBING_FEATURES = [
    "access_pct_lag1", "broad_narrow_ratio_lag1", "oral_parenteral_ratio_lag1",
    "reserve_pct_lag1",
]
DEMOGRAPHIC_FEATURES = [
    "population", "age65_pct", "urban_pct", "gdp_per_capita",
    "health_expenditure_pct_gdp",
]


def _available_features(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    numeric = [c for c in NUMERIC_FEATURES if c in df.columns and not df[c].isna().all()]
    categorical = [c for c in CATEGORICAL_FEATURES if c in df.columns and not df[c].isna().all()]
    return numeric, categorical


def _preprocessor(numeric: list[str], categorical: list[str]) -> ColumnTransformer:
    num = Pipeline([
        (
            "imputer",
            SimpleImputer(
                strategy="median", add_indicator=True, keep_empty_features=True
            ),
        ),
        ("scale", StandardScaler()),
    ])
    cat = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    return ColumnTransformer([("num", num, numeric), ("cat", cat, categorical)])


def _model_grids(random_state: int = 42):
    return {
        "Ridge": (
            Ridge(), {"model__alpha": [0.1, 1.0, 10.0, 100.0]}
        ),
        "Random forest": (
            # A single worker is deliberately used for Colab and other restricted
            # notebook runtimes where process spawning can terminate unexpectedly.
            RandomForestRegressor(random_state=random_state, n_jobs=1),
            {
                "model__n_estimators": [200],
                "model__max_depth": [4, 8, None],
                "model__min_samples_leaf": [2, 5],
            },
        ),
        "Gradient boosting": (
            GradientBoostingRegressor(random_state=random_state),
            {
                "model__n_estimators": [100, 200],
                "model__learning_rate": [0.03, 0.05],
                "model__max_depth": [2, 3],
            },
        ),
        "ANN": (
            MLPRegressor(random_state=random_state, max_iter=1500, early_stopping=True),
            {
                "model__hidden_layer_sizes": [(32,), (64, 32)],
                "model__alpha": [0.0001, 0.01],
            },
        ),
    }


def _metrics(y_true, y_pred, weights=None) -> dict[str, float]:
    result = {
        "RMSE": float(mean_squared_error(y_true, y_pred) ** 0.5),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "R2": float(r2_score(y_true, y_pred)) if len(y_true) > 1 else np.nan,
    }
    if weights is not None and np.isfinite(weights).all() and np.sum(weights) > 0:
        result["WeightedRMSE"] = float(
            np.sqrt(np.average((np.asarray(y_true) - np.asarray(y_pred)) ** 2, weights=weights))
        )
    else:
        result["WeightedRMSE"] = np.nan
    return result


@dataclass
class HorizonResult:
    horizon: int
    performance: pd.DataFrame
    predictions: pd.DataFrame
    selected_model_name: str
    fitted_model: object
    numeric_features: list[str]
    categorical_features: list[str]
    feature_importance: pd.DataFrame


def fit_horizon(
    panel: pd.DataFrame,
    horizon: int,
    output_dir: str | Path | None = None,
    random_state: int = 42,
) -> HorizonResult:
    """Tune on expanding years, select on penultimate target year, test once on last."""
    target = f"target_resistance_h{horizon}"
    target_tested = f"target_tested_h{horizon}"
    if target not in panel:
        raise ValueError(f"Panel has no {target} column.")
    data = panel.loc[panel[target].notna()].copy()
    data["target_year"] = data["year"] + horizon
    years = np.sort(data["target_year"].unique())
    if len(years) < 6:
        raise ValueError(
            f"Horizon {horizon} needs at least six distinct target years; found {len(years)}."
        )
    validation_year, test_year = int(years[-2]), int(years[-1])
    train = data[data["target_year"] < validation_year].copy()
    valid = data[data["target_year"] == validation_year].copy()
    test = data[data["target_year"] == test_year].copy()
    numeric, categorical = _available_features(train)
    features = numeric + categorical
    if not numeric or not categorical:
        raise ValueError("Insufficient numeric or categorical features after audit.")

    rows = []
    fitted_candidates = {}
    persistence_valid = np.clip(valid["resistance_pct"].to_numpy(), 0, 100)
    rows.append({"Model": "Persistence", "Split": "Validation", **_metrics(
        valid[target], persistence_valid, valid[target_tested]
    )})

    cv = ExpandingYearSplit(min_train_years=max(2, min(4, train["target_year"].nunique() - 1)))
    if cv.get_n_splits(groups=train["target_year"]) < 1:
        raise ValueError("Not enough chronological training years for GridSearchCV.")

    for name, (estimator, grid) in _model_grids(random_state).items():
        pipe = Pipeline([
            ("preprocess", _preprocessor(numeric, categorical)),
            ("model", estimator),
        ])
        search = GridSearchCV(
            pipe, grid, scoring="neg_root_mean_squared_error", cv=cv,
            n_jobs=1, refit=True, error_score="raise",
        )
        search.fit(train[features], train[target], groups=train["target_year"])
        pred = np.clip(search.predict(valid[features]), 0, 100)
        rows.append({
            "Model": name, "Split": "Validation", "BestParams": str(search.best_params_),
            **_metrics(valid[target], pred, valid[target_tested]),
        })
        fitted_candidates[name] = search.best_estimator_

    perf = pd.DataFrame(rows)
    selected = (
        perf.loc[perf["Model"].ne("Persistence")]
        .sort_values("RMSE").iloc[0]["Model"]
    )
    combined = pd.concat([train, valid], ignore_index=True)
    final_model = clone(fitted_candidates[selected])
    final_model.fit(combined[features], combined[target])
    test_pred = np.clip(final_model.predict(test[features]), 0, 100)
    persistence_test = np.clip(test["resistance_pct"].to_numpy(), 0, 100)
    perf = pd.concat([
        perf,
        pd.DataFrame([
            {"Model": selected, "Split": "Test", **_metrics(test[target], test_pred, test[target_tested])},
            {"Model": "Persistence", "Split": "Test", **_metrics(
                test[target], persistence_test, test[target_tested]
            )},
        ]),
    ], ignore_index=True)

    predictions = test[[
        "region", "country", "iso3", "year", "target_year", "antibiotic",
        "ast_standard", "ast_version", "resistance_pct", target, target_tested,
    ]].copy()
    predictions["prediction"] = test_pred
    predictions["persistence_prediction"] = persistence_test
    predictions["horizon"] = horizon

    importance_result = permutation_importance(
        final_model,
        test[features],
        test[target],
        scoring="neg_root_mean_squared_error",
        n_repeats=10,
        random_state=random_state,
        n_jobs=1,
    )
    feature_importance = pd.DataFrame({
        "Feature": features,
        "ImportanceMean": importance_result.importances_mean,
        "ImportanceSD": importance_result.importances_std,
        "Horizon": horizon,
    }).sort_values("ImportanceMean", ascending=False)

    if output_dir is not None:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        joblib.dump(final_model, out / f"eu_ecoli_bsi_h{horizon}_{selected.lower().replace(' ', '_')}.joblib")
    return HorizonResult(
        horizon, perf, predictions, selected, final_model, numeric, categorical,
        feature_importance,
    )


def evaluate_feature_groups(
    panel: pd.DataFrame,
    horizon: int,
    fitted_result: HorizonResult,
    material_change_pct: float = 1.0,
) -> pd.DataFrame:
    """Evaluate pre-specified feature additions using one fixed tuned estimator.

    The algorithm and tuned hyperparameters are held constant. Each feature set
    is fitted on the same chronological rows, evaluated first on validation,
    then refitted on train+validation and evaluated on the untouched test year.
    """
    target = f"target_resistance_h{horizon}"
    target_tested = f"target_tested_h{horizon}"
    data = panel.loc[panel[target].notna()].copy()
    data["target_year"] = data["year"] + horizon
    years = np.sort(data["target_year"].unique())
    if len(years) < 6:
        raise ValueError(f"Horizon {horizon} has insufficient target years for ablation.")
    validation_year, test_year = int(years[-2]), int(years[-1])
    train = data[data.target_year < validation_year].copy()
    valid = data[data.target_year.eq(validation_year)].copy()
    test = data[data.target_year.eq(test_year)].copy()
    combined = pd.concat([train, valid], ignore_index=True)

    base_num = [c for c in BASE_NUMERIC_FEATURES if c in data and not train[c].isna().all()]
    consumption = [c for c in CONSUMPTION_FEATURES if c in data and not train[c].isna().all()]
    prescribing = [c for c in PRESCRIBING_FEATURES if c in data and not train[c].isna().all()]
    demographics = [c for c in DEMOGRAPHIC_FEATURES if c in data and not train[c].isna().all()]
    base_cat = [c for c in ["antibiotic", "ast_standard", "ast_version"] if c in data]
    family_cat = ["antibiotic_family"] if "antibiotic_family" in data else []
    feature_sets = {
        "Base resistance history": (base_num, base_cat),
        "Base + consumption": (base_num + consumption, base_cat + family_cat),
        "Base + prescribing behaviour": (base_num + prescribing, base_cat),
        "Base + demographics": (base_num + demographics, base_cat),
        "Full model": (
            base_num + consumption + prescribing + demographics,
            base_cat + family_cat,
        ),
    }
    estimator = clone(fitted_result.fitted_model.named_steps["model"])
    rows = []
    for set_name, (numeric, categorical) in feature_sets.items():
        numeric = list(dict.fromkeys(numeric))
        categorical = list(dict.fromkeys(categorical))
        features = numeric + categorical
        if not numeric or not categorical:
            continue
        validation_model = Pipeline([
            ("preprocess", _preprocessor(numeric, categorical)),
            ("model", clone(estimator)),
        ])
        validation_model.fit(train[features], train[target])
        validation_prediction = np.clip(validation_model.predict(valid[features]), 0, 100)
        rows.append({
            "Horizon": horizon, "FeatureSet": set_name, "Split": "Validation",
            "FeatureCount": len(features), "FeaturesUsed": "; ".join(features),
            **_metrics(valid[target], validation_prediction, valid[target_tested]),
        })

        test_model = Pipeline([
            ("preprocess", _preprocessor(numeric, categorical)),
            ("model", clone(estimator)),
        ])
        test_model.fit(combined[features], combined[target])
        test_prediction = np.clip(test_model.predict(test[features]), 0, 100)
        rows.append({
            "Horizon": horizon, "FeatureSet": set_name, "Split": "Test",
            "FeatureCount": len(features), "FeaturesUsed": "; ".join(features),
            **_metrics(test[target], test_prediction, test[target_tested]),
        })

    result = pd.DataFrame(rows)
    persistence = []
    for split_name, frame in [("Validation", valid), ("Test", test)]:
        persistence.append({
            "Horizon": horizon, "FeatureSet": "Persistence", "Split": split_name,
            "FeatureCount": 1, "FeaturesUsed": "resistance_pct",
            **_metrics(frame[target], frame["resistance_pct"], frame[target_tested]),
        })
    result = pd.concat([result, pd.DataFrame(persistence)], ignore_index=True)

    base_rmse = result[result.FeatureSet.eq("Base resistance history")].set_index("Split")["RMSE"]
    persistence_rmse = result[result.FeatureSet.eq("Persistence")].set_index("Split")["RMSE"]
    result["DeltaRMSE_vs_Base"] = result.apply(
        lambda r: r.RMSE - base_rmse.get(r.Split, np.nan), axis=1
    )
    result["RelativeRMSEChange_vs_Base_pct"] = result.apply(
        lambda r: 100 * (r.RMSE - base_rmse.get(r.Split, np.nan)) /
        base_rmse.get(r.Split, np.nan), axis=1
    )
    result["DeltaRMSE_vs_Persistence"] = result.apply(
        lambda r: r.RMSE - persistence_rmse.get(r.Split, np.nan), axis=1
    )

    def label(row):
        if row.FeatureSet in {"Base resistance history", "Persistence"}:
            return "Reference"
        change = row.RelativeRMSEChange_vs_Base_pct
        if change <= -material_change_pct:
            return "Improved"
        if change >= material_change_pct:
            return "Worsened"
        return "No material change"

    result["EvidenceLabel"] = result.apply(label, axis=1)
    return result.sort_values(["Split", "RMSE"]).reset_index(drop=True)


def evaluate_individual_feature_additions(
    panel: pd.DataFrame,
    horizon: int,
    fitted_result: HorizonResult,
    material_change_pct: float = 1.0,
) -> pd.DataFrame:
    """Add every non-base numeric feature separately under fixed evaluation rules.

    This complements grouped ablation. It identifies whether an apparent group
    benefit is broad or is mainly attributable to one variable. Selection labels
    are calculated separately for validation and test, but only validation should
    be used to make model-development decisions.
    """
    target = f"target_resistance_h{horizon}"
    target_tested = f"target_tested_h{horizon}"
    data = panel.loc[panel[target].notna()].copy()
    data["target_year"] = data["year"] + horizon
    years = np.sort(data["target_year"].unique())
    if len(years) < 6:
        raise ValueError(f"Horizon {horizon} has insufficient target years for ablation.")
    validation_year, test_year = int(years[-2]), int(years[-1])
    train = data[data.target_year < validation_year].copy()
    valid = data[data.target_year.eq(validation_year)].copy()
    test = data[data.target_year.eq(test_year)].copy()
    combined = pd.concat([train, valid], ignore_index=True)

    base_num = [c for c in BASE_NUMERIC_FEATURES if c in data and not train[c].isna().all()]
    base_cat = [c for c in ["antibiotic", "ast_standard", "ast_version"] if c in data]
    candidate_groups = {
        **{c: "Consumption" for c in CONSUMPTION_FEATURES},
        **{c: "Prescribing behaviour" for c in PRESCRIBING_FEATURES},
        **{c: "Demographics" for c in DEMOGRAPHIC_FEATURES},
    }
    candidates = [
        c for c in candidate_groups if c in data and not train[c].isna().all()
        and train[c].nunique(dropna=True) > 1
    ]
    estimator = clone(fitted_result.fitted_model.named_steps["model"])

    def evaluate_set(feature_name: str | None, split_name: str):
        frame_train = train if split_name == "Validation" else combined
        frame_eval = valid if split_name == "Validation" else test
        numeric = base_num + ([feature_name] if feature_name else [])
        categorical = base_cat.copy()
        if feature_name in CONSUMPTION_FEATURES and "antibiotic_family" in data:
            categorical.append("antibiotic_family")
        features = numeric + categorical
        model = Pipeline([
            ("preprocess", _preprocessor(numeric, categorical)),
            ("model", clone(estimator)),
        ])
        model.fit(frame_train[features], frame_train[target])
        prediction = np.clip(model.predict(frame_eval[features]), 0, 100)
        return _metrics(frame_eval[target], prediction, frame_eval[target_tested])

    rows = []
    for split_name in ["Validation", "Test"]:
        base_metrics = evaluate_set(None, split_name)
        rows.append({
            "Horizon": horizon, "Feature": "Base resistance history",
            "Group": "Reference", "Split": split_name,
            "FeaturesUsed": "; ".join(base_num + base_cat), **base_metrics,
        })
        for feature in candidates:
            metrics = evaluate_set(feature, split_name)
            extra_categorical = (
                ["antibiotic_family"]
                if feature in CONSUMPTION_FEATURES and "antibiotic_family" in data
                else []
            )
            rows.append({
                "Horizon": horizon, "Feature": feature,
                "Group": candidate_groups[feature], "Split": split_name,
                "FeaturesUsed": "; ".join(base_num + [feature] + base_cat + extra_categorical),
                **metrics,
            })

    result = pd.DataFrame(rows)
    base_rmse = result[result.Feature.eq("Base resistance history")].set_index("Split")["RMSE"]
    result["DeltaRMSE_vs_Base"] = result.apply(
        lambda r: r.RMSE - base_rmse.get(r.Split, np.nan), axis=1
    )
    result["RelativeRMSEChange_vs_Base_pct"] = result.apply(
        lambda r: 100 * (r.RMSE - base_rmse.get(r.Split, np.nan)) /
        base_rmse.get(r.Split, np.nan), axis=1
    )

    def label(row):
        if row.Feature == "Base resistance history":
            return "Reference"
        change = row.RelativeRMSEChange_vs_Base_pct
        if change <= -material_change_pct:
            return "Improved"
        if change >= material_change_pct:
            return "Worsened"
        return "No material change"

    result["EvidenceLabel"] = result.apply(label, axis=1)
    return result.sort_values(["Split", "RMSE"]).reset_index(drop=True)


def breakpoint_audit(registry: pd.DataFrame) -> pd.DataFrame:
    required = {
        "region", "surveillance_system", "pathogen", "specimen", "antibiotic",
        "year_start", "year_end", "ast_standard", "ast_version", "data_level",
        "comparability_status",
    }
    _require_columns(registry, required, "breakpoint registry")
    summary = (
        registry.groupby(["region", "surveillance_system", "ast_standard", "data_level"],
                         dropna=False, as_index=False)
        .agg(
            rows=("antibiotic", "size"),
            antibiotics=("antibiotic", "nunique"),
            earliest_year=("year_start", "min"),
            latest_year=("year_end", "max"),
            comparable_rows=("comparability_status", lambda s: s.eq("comparable").sum()),
        )
    )
    return summary
