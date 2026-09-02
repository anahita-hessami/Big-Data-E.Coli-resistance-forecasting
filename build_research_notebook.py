"""Build the complete, executable research-grade ML notebook."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def markdown(text: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": text.strip().splitlines(True),
    }


def code(text: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": text.strip().splitlines(True),
    }


cells = [
    markdown(r"""
# Regional bloodstream E. coli AMR forecasting

## Complete machine-learning and research-grade validation notebook

This notebook visibly performs the full workflow: loading the real public data,
exploratory analysis, feature inspection, chronological splitting, construction
of five candidate algorithms, GridSearchCV tuning, model comparison, confirmation
testing, refitting, forward forecasting, uncertainty estimation and model saving.
"""),
    markdown(r"""
### Intended use

The outcome is the country–antibiotic-group resistance percentage reported for
bloodstream/invasive E. coli surveillance. The forecasts are ecological
surveillance estimates, not patient-level susceptibility predictions or treatment
recommendations. EU/EEA outcomes remain under their native EARS-Net/EUCAST
definition and are not pooled directly with US or Japanese CLSI aggregates.
"""),
    markdown("## 1. Environment and project paths"),
    code(r"""
from pathlib import Path
import os, sys, subprocess, warnings

ROOT = Path.cwd()
if not (ROOT / "src").exists() and (ROOT / "regional_ecoli_amr").exists():
    ROOT = ROOT / "regional_ecoli_amr"
if not (ROOT / "src").exists():
    raise FileNotFoundError(
        "Project folder not found. Upload and unzip the complete project, then run again."
    )
os.chdir(ROOT)
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

print("Project root:", ROOT)
print("Python:", sys.version.split()[0])
"""),
    code(r"""
# Install exact project requirements only when a required package is absent.
try:
    import sklearn, pandas, numpy, matplotlib, seaborn, joblib
except ModuleNotFoundError:
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-q", "-r", str(ROOT / "requirements.txt")],
        check=True,
    )
"""),
    code(r"""
import json
from copy import deepcopy

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import (
    mean_absolute_error,
    r2_score,
    root_mean_squared_error,
)
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

try:
    from IPython.display import display, Image, Markdown
except ModuleNotFoundError:
    display = print
    Image = lambda filename=None, **kwargs: filename
    Markdown = lambda value: value

try:
    get_ipython().run_line_magic("matplotlib", "inline")
except NameError:
    pass

sns.set_theme(style="whitegrid")
RANDOM_STATE = 42
HORIZONS = (1, 3, 5)
MODEL_OUTPUT = ROOT / "outputs" / "notebook_ml"
FIGURE_OUTPUT = MODEL_OUTPUT / "figures"
MODEL_OUTPUT.mkdir(parents=True, exist_ok=True)
FIGURE_OUTPUT.mkdir(parents=True, exist_ok=True)
"""),
    markdown("## 2. Dataset confirmation and initial inspection"),
    code(r"""
input_paths = {
    "EARS-Net AMR": ROOT / "data/input/ears_net_ecoli_blood.csv",
    "ESAC-Net consumption": ROOT / "data/input/esac_net_consumption.csv",
    "Demographics": ROOT / "data/input/demographics.csv",
    "Antibiotic mapping": ROOT / "data/input/antibiotic_mapping.csv",
}
status = pd.DataFrame([
    {
        "Dataset": name,
        "Path": str(path),
        "Found": path.exists(),
        "Rows": len(pd.read_csv(path)) if path.exists() else np.nan,
    }
    for name, path in input_paths.items()
])
display(status)
if not status["Found"].all():
    raise FileNotFoundError("One or more required official input CSV files are missing.")
"""),
    code(r"""
panel_path = ROOT / "data/processed/eu_ecoli_bsi_enhanced_feature_panel.csv"
if not panel_path.exists():
    subprocess.run([sys.executable, "-m", "src.run_official_pipeline"], check=True)
    subprocess.run([sys.executable, "-m", "src.run_iterative_optimization"], check=True)

panel = pd.read_csv(panel_path)
selected_features = pd.read_csv(
    ROOT / "outputs/optimization/optimized_selected_features.csv"
)

summary = pd.DataFrame({
    "Measure": [
        "Rows", "Columns", "Countries", "Antibiotic groups",
        "First origin year", "Last origin year",
    ],
    "Value": [
        len(panel), panel.shape[1], panel["country"].nunique(),
        panel["antibiotic"].nunique(), int(panel["year"].min()),
        int(panel["year"].max()),
    ],
})
display(summary)
"""),
    code(r"""
label_summary = []
for horizon in HORIZONS:
    target = f"target_resistance_h{horizon}"
    labelled = panel[target].notna()
    label_summary.append({
        "Horizon": horizon,
        "LabelledRows": int(labelled.sum()),
        "UnlabelledRowsForFutureUse": int((~labelled).sum()),
        "FirstTargetYear": int((panel.loc[labelled, "year"] + horizon).min()),
        "LastObservedTargetYear": int((panel.loc[labelled, "year"] + horizon).max()),
    })
display(pd.DataFrame(label_summary))
"""),
    code(r"""
missingness = (
    panel.isna().mean().mul(100).sort_values(ascending=False)
    .rename("MissingPct").reset_index().rename(columns={"index": "Feature"})
)
display(missingness.head(25).round({"MissingPct": 1}))

fig, ax = plt.subplots(figsize=(10, 6))
sns.barplot(data=missingness.head(20), x="MissingPct", y="Feature", color="#4C78A8", ax=ax)
ax.set_title("Highest feature missingness")
ax.set_xlabel("Missing values (%)")
fig.tight_layout()
plt.show()
"""),
    markdown(r"""
## 3. Exploratory visual analysis

The following charts are calculated from the included official-source inputs.
They show reporting coverage, resistance trends, feature relationships and
feature availability before modelling.
"""),
    code(r"""
eda_figures = [
    "01_country_year_coverage_heatmap.png",
    "06_weighted_resistance_trends.png",
    "07_latest_country_antibiotic_heatmap.png",
    "09_prescribing_behaviour_trends.png",
    "13_predictor_correlation_heatmap.png",
    "27_feature_availability_over_time.png",
]
for filename in eda_figures:
    path = ROOT / "outputs/figures" / filename
    if path.exists():
        display(Markdown(f"### {filename.replace('_', ' ').replace('.png', '')}"))
        display(Image(filename=str(path)))
"""),
    markdown(r"""
## 4. Dimensionality-reduction judgement

PCA is not used. After horizon-specific selection, the models use a moderate
number of scientifically interpretable variables, with categorical variables
one-hot encoded inside the pipeline. PCA would mix resistance history,
consumption, demographic and surveillance variables into less interpretable
components. Regularised Ridge and tree ensembles already provide protection
against correlated predictors. PCA can be reconsidered for future genomic or
high-dimensional patient-level data.
"""),
    markdown("## 5. Features available to each forecasting horizon"),
    code(r"""
feature_matrix = (
    selected_features.assign(Selected=1)
    .pivot_table(index=["Feature", "Type"], columns="Horizon", values="Selected", fill_value=0)
    .reset_index()
)
display(feature_matrix)
"""),
    markdown(r"""
These are frozen feature specifications produced by the earlier leakage-safe
forward-selection stage. This notebook re-trains and compares all candidate
algorithms using those frozen inputs. Feature selection is not repeated on the
confirmation year.
"""),
    markdown("## 6. Chronological validation design"),
    code(r'''
def chronological_splits(frame, max_folds=5):
    """Return expanding-window train/validation indices ordered by target year."""
    years = np.sort(frame["target_year"].unique())
    first_validation = max(2, len(years) - max_folds)
    splits = []
    for validation_year in years[first_validation:]:
        train_idx = np.flatnonzero(frame["target_year"].to_numpy() < validation_year)
        valid_idx = np.flatnonzero(frame["target_year"].to_numpy() == validation_year)
        if len(train_idx) and len(valid_idx):
            splits.append((train_idx, valid_idx))
    if len(splits) < 2:
        raise ValueError("At least two chronological validation folds are required.")
    return splits


split_rows = []
for horizon in HORIZONS:
    target = f"target_resistance_h{horizon}"
    labelled = panel.loc[panel[target].notna()].copy()
    labelled["target_year"] = labelled["year"] + horizon
    confirmation_year = int(labelled["target_year"].max())
    selection = labelled.loc[labelled["target_year"] < confirmation_year].reset_index(drop=True)
    for fold, (train_idx, valid_idx) in enumerate(chronological_splits(selection), 1):
        split_rows.append({
            "Horizon": horizon,
            "Fold": fold,
            "TrainThroughTargetYear": int(selection.iloc[train_idx]["target_year"].max()),
            "ValidationTargetYear": int(selection.iloc[valid_idx]["target_year"].iloc[0]),
            "TrainingRows": len(train_idx),
            "ValidationRows": len(valid_idx),
            "HeldOutConfirmationYear": confirmation_year,
        })
display(pd.DataFrame(split_rows))
'''),
    markdown(r"""
The newest observed target year is not used for GridSearchCV or model choice.
Each validation fold trains only on earlier target years. This prevents random
train/test leakage across time.
"""),
    markdown("## 7. Preprocessing pipeline"),
    code(r"""
def build_preprocessor(numeric_features, categorical_features):
    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
        ("scaler", StandardScaler()),
    ])
    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    return ColumnTransformer([
        ("numeric", numeric_pipe, numeric_features),
        ("categorical", categorical_pipe, categorical_features),
    ])
"""),
    markdown("## 8. Candidate ML models and GridSearchCV parameter grids"),
    code(r"""
def candidate_models():
    return {
        "Ridge": (
            Ridge(),
            {"model__alpha": [0.1, 1.0, 10.0, 100.0, 300.0]},
        ),
        "Gradient Boosting": (
            GradientBoostingRegressor(random_state=RANDOM_STATE),
            {
                "model__n_estimators": [100, 200],
                "model__learning_rate": [0.05],
                "model__max_depth": [2, 3],
                "model__min_samples_leaf": [3],
            },
        ),
        "Random Forest": (
            RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=1),
            {
                "model__n_estimators": [150],
                "model__max_depth": [None],
                "model__min_samples_leaf": [2, 5],
                "model__max_features": [0.7],
            },
        ),
        "Extra Trees": (
            ExtraTreesRegressor(random_state=RANDOM_STATE, n_jobs=1),
            {
                "model__n_estimators": [150],
                "model__max_depth": [None],
                "model__min_samples_leaf": [2, 5],
                "model__max_features": [0.7],
            },
        ),
        "ANN": (
            MLPRegressor(
                random_state=RANDOM_STATE,
                max_iter=400,
                early_stopping=True,
                learning_rate_init=0.001,
            ),
            {
                "model__hidden_layer_sizes": [(32,)],
                "model__alpha": [0.01],
            },
        ),
    }


grid_description = []
for name, (_, grid) in candidate_models().items():
    combinations = int(np.prod([len(values) for values in grid.values()]))
    grid_description.append({
        "Model": name,
        "ParameterCombinationsPerHorizon": combinations,
        "Grid": json.dumps(grid, default=str),
    })
display(pd.DataFrame(grid_description))
"""),
    markdown(r"""
Ridge is a regularised linear benchmark. Gradient Boosting, Random Forest and
Extra Trees capture nonlinearities and interactions. ANN is included because it
was requested as a possible enhancement, but it is selected only if temporal
validation supports it.
"""),
    markdown("## 9. Metric functions"),
    code(r"""
def regression_metrics(observed, predicted, weights=None):
    observed = np.asarray(observed, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    result = {
        "RMSE": root_mean_squared_error(observed, predicted),
        "MAE": mean_absolute_error(observed, predicted),
        "R2": r2_score(observed, predicted),
    }
    if weights is not None:
        weights = np.asarray(weights, dtype=float)
        valid = np.isfinite(weights) & (weights > 0)
        result["WeightedRMSE"] = float(
            np.sqrt(np.average((observed[valid] - predicted[valid]) ** 2, weights=weights[valid]))
        )
    return result
"""),
    markdown("## 10. Train, tune and test the ML models"),
    code(r"""
RUN_MODEL_TRAINING = True

candidate_results = []
confirmation_results = []
confirmation_predictions = []
best_parameters = []
fitted_confirmation_models = {}
fitted_full_models = {}
feature_sets = {}

if not RUN_MODEL_TRAINING:
    raise ValueError("RUN_MODEL_TRAINING must remain True in the complete ML notebook.")

warnings.filterwarnings("ignore", category=UserWarning)

for horizon in HORIZONS:
    print(f"Training horizon {horizon}...")
    target = f"target_resistance_h{horizon}"
    target_tested = f"target_tested_h{horizon}"
    data = panel.loc[panel[target].notna()].copy()
    data["target_year"] = data["year"] + horizon
    confirmation_year = int(data["target_year"].max())
    selection = data.loc[data["target_year"] < confirmation_year].reset_index(drop=True)
    confirmation = data.loc[data["target_year"] == confirmation_year].copy()

    numeric = selected_features.loc[
        (selected_features["Horizon"] == horizon)
        & (selected_features["Type"] == "numeric"), "Feature"
    ].tolist()
    categorical = selected_features.loc[
        (selected_features["Horizon"] == horizon)
        & (selected_features["Type"] == "categorical"), "Feature"
    ].tolist()
    numeric = [name for name in numeric if name in selection and not selection[name].isna().all()]
    categorical = [
        name for name in categorical
        if name in selection and not selection[name].isna().all()
    ]
    features = numeric + categorical
    feature_sets[horizon] = {"numeric": numeric, "categorical": categorical, "all": features}
    cv = chronological_splits(selection)

    searches = {}
    for model_name, (estimator, parameter_grid) in candidate_models().items():
        pipeline = Pipeline([
            ("preprocess", build_preprocessor(numeric, categorical)),
            ("model", estimator),
        ])
        search = GridSearchCV(
            estimator=pipeline,
            param_grid=parameter_grid,
            scoring="neg_root_mean_squared_error",
            cv=cv,
            n_jobs=1,
            refit=True,
            return_train_score=False,
            error_score="raise",
        )
        search.fit(selection[features], selection[target])
        searches[model_name] = search
        candidate_results.append({
            "Horizon": horizon,
            "Model": model_name,
            "BestCV_RMSE": -search.best_score_,
            "BestParameters": json.dumps(search.best_params_, default=str),
            "ValidationFolds": len(cv),
        })

    best_name = min(searches, key=lambda name: -searches[name].best_score_)
    best_search = searches[best_name]
    confirmation_model = best_search.best_estimator_
    fitted_confirmation_models[horizon] = confirmation_model
    prediction = np.clip(
        confirmation_model.predict(confirmation[features]), 0, 100
    )
    persistence = confirmation["resistance_pct"].to_numpy()

    model_metrics = regression_metrics(
        confirmation[target], prediction, confirmation[target_tested]
    )
    persistence_metrics = regression_metrics(
        confirmation[target], persistence, confirmation[target_tested]
    )
    confirmation_results.extend([
        {
            "Horizon": horizon,
            "Forecast": best_name,
            "ConfirmationTargetYear": confirmation_year,
            **model_metrics,
        },
        {
            "Horizon": horizon,
            "Forecast": "Persistence",
            "ConfirmationTargetYear": confirmation_year,
            **persistence_metrics,
        },
    ])
    best_parameters.append({
        "Horizon": horizon,
        "SelectedModel": best_name,
        "BestCV_RMSE": -best_search.best_score_,
        "BestParameters": json.dumps(best_search.best_params_, default=str),
        "NumericFeatureCount": len(numeric),
        "CategoricalFeatureCount": len(categorical),
    })

    prediction_frame = confirmation[
        ["country", "iso3", "year", "target_year", "antibiotic",
         "resistance_pct", target, target_tested]
    ].copy()
    prediction_frame = prediction_frame.rename(
        columns={target: "observed", target_tested: "target_tested"}
    )
    prediction_frame["model_prediction"] = prediction
    prediction_frame["persistence_prediction"] = persistence
    prediction_frame["Horizon"] = horizon
    prediction_frame["SelectedModel"] = best_name
    confirmation_predictions.append(prediction_frame)

    # Refit the frozen winning pipeline on every labelled row before forecasting.
    full_model = clone(best_search.best_estimator_)
    full_model.fit(data[features], data[target])
    fitted_full_models[horizon] = full_model
    safe_name = best_name.lower().replace(" ", "_")
    joblib.dump(
        full_model,
        MODEL_OUTPUT / f"final_refit_h{horizon}_{safe_name}.joblib",
    )

candidate_results = pd.DataFrame(candidate_results)
confirmation_results = pd.DataFrame(confirmation_results)
confirmation_predictions = pd.concat(confirmation_predictions, ignore_index=True)
best_parameters = pd.DataFrame(best_parameters)

candidate_results.to_csv(MODEL_OUTPUT / "candidate_gridsearch_results.csv", index=False)
confirmation_results.to_csv(MODEL_OUTPUT / "confirmation_performance.csv", index=False)
confirmation_predictions.to_csv(MODEL_OUTPUT / "confirmation_predictions.csv", index=False)
best_parameters.to_csv(MODEL_OUTPUT / "selected_models_and_parameters.csv", index=False)

print("Training complete.")
"""),
    markdown("## 11. GridSearchCV model comparison"),
    code(r"""
display(
    candidate_results.sort_values(["Horizon", "BestCV_RMSE"])
    .round({"BestCV_RMSE": 3})
)

fig, ax = plt.subplots(figsize=(11, 6))
sns.barplot(
    data=candidate_results,
    x="Horizon", y="BestCV_RMSE", hue="Model", ax=ax
)
ax.set_title("Chronological GridSearchCV comparison")
ax.set_ylabel("Validation RMSE (lower is better)")
fig.tight_layout()
fig.savefig(FIGURE_OUTPUT / "01_gridsearch_model_comparison.png", dpi=180)
plt.show()
"""),
    code(r"""
display(best_parameters.round({"BestCV_RMSE": 3}))
"""),
    markdown(r"""
The model is selected using chronological validation RMSE, not R² and not the
confirmation year. The latest observed target year is opened only after the
model family, parameters and feature specification have been frozen.
"""),
    markdown("## 12. Confirmation testing against persistence"),
    code(r"""
display(confirmation_results.round({
    "RMSE": 3, "MAE": 3, "R2": 3, "WeightedRMSE": 3,
}))

fig, ax = plt.subplots(figsize=(10, 6))
sns.barplot(
    data=confirmation_results, x="Horizon", y="RMSE", hue="Forecast", ax=ax
)
ax.set_title("Latest-year confirmation: selected model versus persistence")
ax.set_ylabel("RMSE (resistance percentage points)")
fig.tight_layout()
fig.savefig(FIGURE_OUTPUT / "02_confirmation_rmse.png", dpi=180)
plt.show()
"""),
    code(r"""
fig, axes = plt.subplots(1, 3, figsize=(17, 5))
for ax, horizon in zip(axes, HORIZONS):
    frame = confirmation_predictions.loc[
        confirmation_predictions["Horizon"] == horizon
    ]
    sns.scatterplot(
        data=frame,
        x="observed", y="model_prediction", hue="antibiotic",
        alpha=0.75, legend=False, ax=ax,
    )
    ax.plot([0, 100], [0, 100], "--", color="black", linewidth=1)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.set_title(f"{horizon}-year")
    ax.set_xlabel("Observed resistance (%)")
    ax.set_ylabel("Predicted resistance (%)")
fig.suptitle("Held-out observed versus predicted resistance", y=1.02)
fig.tight_layout()
fig.savefig(FIGURE_OUTPUT / "03_observed_vs_predicted.png", dpi=180)
plt.show()
"""),
    markdown("## 13. Evidence-based forecast decision"),
    code(r"""
research_summary = pd.read_csv(
    ROOT / "outputs/research_grade/research_grade_summary.csv"
)
decision_table = best_parameters[["Horizon", "SelectedModel"]].copy()
decision_table["RecommendedMethod"] = decision_table.apply(
    lambda row: "Persistence" if row["Horizon"] == 1 else row["SelectedModel"],
    axis=1,
)
decision_table["Reason"] = [
    "Nested backtesting did not show reliable improvement over persistence.",
    "Repeated nested and frozen backtests beat persistence.",
    "Frozen backtests were strong; nested evidence is limited to one valid cutoff.",
]
display(decision_table)
"""),
    markdown(r"""
Although a one-year Ridge model is trained and evaluated, it is not used as the
recommended operational forecast because its apparent validation improvement
did not generalise reliably. The persistence rule is retained at one year.
"""),
    markdown("## 14. Generate genuine forward forecasts"),
    code(r"""
latest_origin_year = int(panel["year"].max())
latest_origin = panel.loc[panel["year"] == latest_origin_year].copy()
future_frames = []

for horizon in HORIZONS:
    features = feature_sets[horizon]["all"]
    model = fitted_full_models[horizon]
    model_forecast = np.clip(model.predict(latest_origin[features]), 0, 100)
    persistence_forecast = latest_origin["resistance_pct"].to_numpy()
    use_persistence = horizon == 1
    recommended = persistence_forecast if use_persistence else model_forecast

    historical_errors = pd.read_csv(
        ROOT / f"outputs/research_grade/frozen_backtest_predictions_h{horizon}.csv"
    )
    error_column = "persistence_error" if use_persistence else "model_error"
    absolute_errors = historical_errors[error_column].abs().dropna()
    radius80 = float(absolute_errors.quantile(0.80))
    radius95 = float(absolute_errors.quantile(0.95))

    future = latest_origin[
        ["country", "iso3", "antibiotic", "antibiotic_family",
         "year", "resistance_pct", "tested"]
    ].copy()
    future = future.rename(columns={
        "year": "OriginYear",
        "resistance_pct": "LatestObservedResistancePct",
    })
    future["Horizon"] = horizon
    future["ForecastTargetYear"] = latest_origin_year + horizon
    future["SelectedMLModel"] = best_parameters.loc[
        best_parameters["Horizon"] == horizon, "SelectedModel"
    ].iloc[0]
    future["ModelForecastPct"] = model_forecast
    future["PersistenceForecastPct"] = persistence_forecast
    future["RecommendedMethod"] = (
        "Persistence" if use_persistence
        else future["SelectedMLModel"]
    )
    future["RecommendedForecastPct"] = recommended
    future["Lower80Pct"] = np.clip(recommended - radius80, 0, 100)
    future["Upper80Pct"] = np.clip(recommended + radius80, 0, 100)
    future["Lower95Pct"] = np.clip(recommended - radius95, 0, 100)
    future["Upper95Pct"] = np.clip(recommended + radius95, 0, 100)
    future_frames.append(future)

future_forecasts = pd.concat(future_frames, ignore_index=True)
future_forecasts.to_csv(MODEL_OUTPUT / "future_forecasts.csv", index=False)

forecast_summary = (
    future_forecasts.groupby(
        ["Horizon", "ForecastTargetYear", "RecommendedMethod", "antibiotic"],
        as_index=False,
    )
    .agg(
        MeanForecastPct=("RecommendedForecastPct", "mean"),
        MedianForecastPct=("RecommendedForecastPct", "median"),
        Countries=("iso3", "nunique"),
    )
)
forecast_summary.to_csv(MODEL_OUTPUT / "future_forecast_summary.csv", index=False)
display(forecast_summary.round({
    "MeanForecastPct": 2,
    "MedianForecastPct": 2,
}))
"""),
    code(r"""
display(
    future_forecasts.sort_values(
        ["ForecastTargetYear", "RecommendedForecastPct"],
        ascending=[True, False],
    ).head(30).round({
        "LatestObservedResistancePct": 2,
        "ModelForecastPct": 2,
        "PersistenceForecastPct": 2,
        "RecommendedForecastPct": 2,
        "Lower80Pct": 2,
        "Upper80Pct": 2,
        "Lower95Pct": 2,
        "Upper95Pct": 2,
    })
)
"""),
    code(r"""
fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)
for ax, horizon in zip(axes, HORIZONS):
    frame = forecast_summary.loc[forecast_summary["Horizon"] == horizon]
    sns.barplot(
        data=frame.sort_values("MeanForecastPct", ascending=False),
        x="MeanForecastPct", y="antibiotic", color="#4C78A8", ax=ax,
    )
    target_year = int(frame["ForecastTargetYear"].iloc[0])
    ax.set_title(f"{target_year} forecast ({horizon}-year)")
    ax.set_xlabel("Mean predicted resistance (%)")
    ax.set_ylabel("Antibiotic group")
fig.suptitle("Forward forecast by antibiotic group", y=1.02)
fig.tight_layout()
fig.savefig(FIGURE_OUTPUT / "04_forward_forecast_by_antibiotic.png", dpi=180)
plt.show()
"""),
    code(r"""
fig, axes = plt.subplots(1, 3, figsize=(18, 11))
for ax, horizon in zip(axes, HORIZONS):
    frame = future_forecasts.loc[future_forecasts["Horizon"] == horizon]
    heat = frame.pivot_table(
        index="country", columns="antibiotic",
        values="RecommendedForecastPct", aggfunc="mean",
    )
    sns.heatmap(heat, cmap="YlOrRd", vmin=0, vmax=100, ax=ax, cbar=horizon == 5)
    ax.set_title(f"Target {latest_origin_year + horizon}")
    ax.set_xlabel("")
    ax.set_ylabel("Country" if horizon == 1 else "")
fig.suptitle("Country–antibiotic forward resistance forecasts", y=1.01)
fig.tight_layout()
fig.savefig(FIGURE_OUTPUT / "05_forward_forecast_heatmaps.png", dpi=180)
plt.show()
"""),
    markdown(r"""
The forecast table contains both the trained ML estimate and persistence. The
RecommendedForecastPct column follows the validation decision: persistence for
2024 and the selected Extra Trees models for 2026 and 2028. The 80% and 95%
intervals are empirical error bands derived from earlier forecast origins.
They represent forecast uncertainty, not confidence about a causal effect.
"""),
    markdown("## 15. Confirm that the fitted models and forecast files were saved"),
    code(r"""
saved_artifacts = []
for path in sorted(MODEL_OUTPUT.glob("*")):
    if path.is_file():
        saved_artifacts.append({
            "File": path.name,
            "SizeKB": round(path.stat().st_size / 1024, 1),
        })
display(pd.DataFrame(saved_artifacts))
"""),
    markdown("## 16. Research-grade rolling-origin validation"),
    code(r"""
rolling = pd.read_csv(
    ROOT / "outputs/research_grade/rolling_origin_performance_all_horizons.csv"
)
pooled = pd.read_csv(
    ROOT / "outputs/research_grade/research_grade_summary.csv"
)
pooled_display = pooled.copy()
pooled_display["SkillPct"] = 100 * pooled_display["Skill"]
pooled_display["WeightedSkillPct"] = 100 * pooled_display["WeightedSkill"]
display(pooled_display.round({
    "ModelRMSE": 3, "PersistenceRMSE": 3,
    "SkillPct": 1, "WeightedSkillPct": 1,
}))
display(Image(filename=str(
    ROOT / "outputs/research_grade/figures/01_rolling_origin_skill.png"
)))
"""),
    markdown("## 17. Clustered uncertainty and conformal coverage"),
    code(r"""
bootstrap = pd.read_csv(
    ROOT / "outputs/research_grade/cluster_bootstrap_summary.csv"
)
coverage = pd.read_csv(
    ROOT / "outputs/research_grade/conformal_coverage_summary.csv"
)
bootstrap_display = bootstrap.copy()
bootstrap_display["ProbabilityModelWorsePct"] = (
    100 * bootstrap_display["ProbabilityModelWorse"]
)
coverage_display = coverage.copy()
coverage_display["NominalCoveragePct"] = 100 * coverage_display["NominalCoverage"]
coverage_display["EmpiricalCoveragePct"] = 100 * coverage_display["EmpiricalCoverage"]
display(bootstrap_display.round({
    "Estimate": 3, "Lower95": 3, "Upper95": 3,
    "ProbabilityModelWorsePct": 1,
}))
display(coverage_display.round({
    "NominalCoveragePct": 1, "EmpiricalCoveragePct": 1,
    "AverageWidth": 2, "MedianWidth": 2,
}))
display(Image(filename=str(
    ROOT / "outputs/research_grade/figures/02_cluster_bootstrap_skill.png"
)))
display(Image(filename=str(
    ROOT / "outputs/research_grade/figures/03_conformal_intervals.png"
)))
"""),
    markdown("## 18. Partial pooling and multiple-comparison controls"),
    code(r"""
pooling = pooled.loc[
    pooled["Evaluation"].isin(["Frozen specification", "Partial pooling"])
]
screen = pd.read_csv(
    ROOT / "outputs/research_grade/multiplicity_controlled_feature_screen.csv"
)
retained = screen.loc[screen["RetainAfterMultiplicityControl"] == True]
pooling_display = pooling.copy()
pooling_display["SkillPct"] = 100 * pooling_display["Skill"]
pooling_display["WeightedSkillPct"] = 100 * pooling_display["WeightedSkill"]
display(pooling_display.round({
    "ModelRMSE": 3, "PersistenceRMSE": 3,
    "SkillPct": 1, "WeightedSkillPct": 1,
}))
display(retained)
display(Image(filename=str(
    ROOT / "outputs/research_grade/figures/04_multiplicity_controlled_features.png"
)))
"""),
    markdown(r"""
The partial-pooling benchmark did not improve RMSE in the present same-country
temporal evaluation. In the 20-null-repetition engineering screen, only
origin_year_index and reserve_pct_lag2 at the three-year horizon passed all
configured rules. Publication-level claims require at least 200 null
repetitions because 20 produces coarse adjusted p-values.
"""),
    markdown("## 19. Final interpretation"),
    markdown(r"""
1. Five ML algorithms are actually fitted and tuned inside this notebook.
2. Ridge wins the one-year chronological GridSearchCV, while Extra Trees wins
   the three- and five-year searches.
3. Persistence remains the recommended one-year forecast because the ML
   advantage did not survive stricter repeated testing.
4. The three-year model has the strongest overall evidence because it beats
   persistence across repeated fully nested historical cutoffs.
5. The five-year model is promising, but the history supports only one fully
   nested cutoff and therefore requires prospective confirmation.
6. The generated forward outputs target 2024, 2026 and 2028 from the latest
   2023 AMR origin data.

These predictions should support surveillance research and hypothesis
generation only. They must not be used to select antibiotics for individual
patients.
"""),
]


notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {"name": "python", "version": "3.12"},
        "colab": {"name": "Regional_Ecoli_BSI_AMR_Complete_ML.ipynb"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

(ROOT / "notebooks").mkdir(parents=True, exist_ok=True)
path = ROOT / "notebooks" / "Regional_Ecoli_BSI_AMR_Complete_ML.ipynb"
path.write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + "\n")
print(path)
