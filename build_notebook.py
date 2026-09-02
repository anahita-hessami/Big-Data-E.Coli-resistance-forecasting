"""Build the Phase 1 Jupyter notebook without requiring nbformat."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def md(source: str):
    return {"cell_type": "markdown", "metadata": {}, "source": source}


def code(source: str):
    return {
        "cell_type": "code", "execution_count": None, "metadata": {},
        "outputs": [], "source": source,
    }


cells = [
    md("""# Regional bloodstream *E. coli* AMR forecasting — Phase 1: EU/EEA

**Primary development data:** EARS-Net resistance outcomes, ESAC-Net antimicrobial consumption and prescribing-behaviour proxies, plus public demographic indicators.
**Forecast horizons:** 1, 3 and 5 years.
**Outcome:** country–antibiotic–year bloodstream/invasive *E. coli* resistance percentage.
**Laboratory definition:** native EARS-Net aggregate; EUCAST is required from 2020, while earlier standards remain mixed/not reported.

## Research question

> To what extent do recent resistance trends, antibiotic consumption, prescribing behaviour and demographic changes improve 1-, 3- and 5-year forecasts of bloodstream *E. coli* resistance across EU/EEA countries?

This is a population-surveillance research model. It is not a patient-level susceptibility estimate and does not recommend antibiotic treatment."""),
    md("""## Revised regional design

The earlier WHO GLASS baseline grouped country resistance profiles and forecast one year ahead. It achieved test RMSE 5.57 percentage points versus 5.87 for persistence, using gradient boosting on annual change. The revised project addresses its most important scientific limitation: aggregated EUCAST and CLSI/FDA outcomes cannot be assumed to be equivalent.

The present notebook therefore develops the EU model only. US and Japanese notebooks will retain their native outcome definitions. A later hierarchical stage can share predictor relationships without pretending that regional resistance labels are identical."""),
    code("""# Colab/local setup
from pathlib import Path
import sys

candidates = [Path.cwd(), Path.cwd() / "regional_ecoli_amr", Path("/content/regional_ecoli_amr")]
PROJECT_ROOT = next((p for p in candidates if (p / "src/regional_amr.py").exists()), None)
if PROJECT_ROOT is None:
    raise FileNotFoundError("Upload and unzip the complete regional_ecoli_amr project, then rerun this cell.")
sys.path.insert(0, str(PROJECT_ROOT))

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
try:
    from IPython import get_ipython
    from IPython.display import display, HTML, Image, IFrame, Markdown
    shell = get_ipython()
    if shell is not None:
        shell.run_line_magic("matplotlib", "inline")
except ImportError:
    display = print
    HTML = Image = IFrame = Markdown = lambda value=None, **kwargs: value or kwargs.get("filename")

from src.regional_amr import (
    EARS_REQUIRED, CONSUMPTION_REQUIRED, DEMOGRAPHIC_REQUIRED, MAPPING_REQUIRED,
    breakpoint_audit, build_eu_panel, evaluate_feature_groups,
    evaluate_individual_feature_additions, fit_horizon, read_csv,
    validate_consumption, validate_demographics, validate_ears,
)
from src.visualizations import (
    create_interactive_maps, plot_data_overview, plot_development_workflow,
    plot_feature_ablation, plot_feature_exploration,
    plot_importance_stability, plot_individual_feature_ablation,
    plot_model_results, plot_predictor_overview, write_visualisation_index,
)

INPUT_DIR = PROJECT_ROOT / "data/input"
PROCESSED_DIR = PROJECT_ROOT / "data/processed"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
sns.set_theme(style="whitegrid")

def show_visualisations(paths, section_name):
    # Reliably render saved figures inside Jupyter/Google Colab.
    existing = [Path(path) for path in paths if Path(path).exists()]
    if not existing:
        print(f"No {section_name} visualisations were generated.")
        return
    display(Markdown(f"### {section_name}: {len(existing)} generated output(s)"))
    for path in existing:
        title = path.stem.replace("_", " ").title()
        display(Markdown(f"#### {title}"))
        if path.suffix.lower() in {".png", ".jpg", ".jpeg", ".gif"}:
            display(Image(filename=str(path)))
        elif path.suffix.lower() == ".html":
            try:
                display(HTML(filename=str(path)))
            except Exception:
                display(HTML(f'<a href="{path}" target="_blank">Open interactive map</a>'))

print("Project root:", PROJECT_ROOT)"""),
    md("""## Step 1 — Dataset confirmation and initial inspection

Required files:

1. `ears_net_ecoli_blood.csv` — resistant and tested counts by country, year and antibiotic;
2. `esac_net_consumption.csv` — consumption by country, year, sector and antibiotic family;
3. `demographics.csv` — country-year population and age structure, with optional socioeconomic indicators;
4. `antibiotic_mapping.csv` — maps every AMR antibiotic to its matching consumption family.

The templates in `data/templates` specify the required columns. The real files in `data/input` can be reproduced by running `python src/prepare_official_inputs.py`. The code refuses duplicated outcome rows, non-blood specimens, non-*E. coli* pathogens, impossible counts and missing antibiotic-family mappings."""),
    code("""def locate_input(filename):
    # Accept files in data/input, beside the notebook, in the current folder,
    # or in /content after a direct Google Colab upload.
    candidates = [
        INPUT_DIR / filename,
        PROJECT_ROOT / filename,
        Path.cwd() / filename,
        Path("/content") / filename,
    ]
    return next((path for path in candidates if path.exists()), candidates[0])

paths = {
    "EARS-Net": locate_input("ears_net_ecoli_blood.csv"),
    "ESAC-Net": locate_input("esac_net_consumption.csv"),
    "Demographics": locate_input("demographics.csv"),
    "Antibiotic mapping": locate_input("antibiotic_mapping.csv"),
}
missing = {name: str(path) for name, path in paths.items() if not path.exists()}
DATA_READY = not missing
display(pd.DataFrame([
    {"Dataset": name, "Resolved path": str(path), "Found": path.exists()}
    for name, path in paths.items()
]))
if missing:
    display(Markdown("## ⚠️ Data-driven visualisations are paused"))
    print("The workflow chart will display, but resistance, feature and model charts require all four official exports:")
    for name, path in missing.items():
        print(f"- {name}: {path}")
else:
    print("All Phase 1 input files are present.")"""),
    code("""manifest_path = PROJECT_ROOT / "data/SOURCE_MANIFEST.csv"
if manifest_path.exists():
    display(pd.read_csv(manifest_path))
else:
    print("Source manifest not found. Run: python src/prepare_official_inputs.py")"""),
    code("""if DATA_READY:
    ears_raw = read_csv(paths["EARS-Net"], EARS_REQUIRED, "EARS-Net")
    consumption_raw = read_csv(paths["ESAC-Net"], CONSUMPTION_REQUIRED, "ESAC-Net")
    demographics_raw = read_csv(paths["Demographics"], DEMOGRAPHIC_REQUIRED, "demographics")
    mapping = read_csv(paths["Antibiotic mapping"], MAPPING_REQUIRED, "antibiotic mapping")

    ears, ears_audit = validate_ears(ears_raw)
    consumption = validate_consumption(consumption_raw)
    demographics = validate_demographics(demographics_raw)

    display(ears_audit)
    display(ears.head())
    display(pd.DataFrame({
        "dtype": ears.dtypes.astype(str),
        "missing_n": ears.isna().sum(),
        "missing_pct": 100 * ears.isna().mean(),
        "unique_n": ears.nunique(dropna=True),
    }))
else:
    print("Inspection will run after the four official input files are added.")"""),
    md("""### Breakpoint and provenance audit

The audit below is mandatory even before modelling. It records the surveillance system, specimen, AST standard, version and aggregation level. Because the current sources are aggregated, their percentages remain under their native regional standards. A standard indicator is useful documentation but is not treated as a mathematical conversion between EUCAST and CLSI/FDA."""),
    code("""registry = pd.read_csv(PROJECT_ROOT / "data/templates/breakpoint_registry.csv")
display(registry)
display(breakpoint_audit(registry))

if DATA_READY:
    version_audit = (
        ears.groupby(["year", "ast_standard", "ast_version"], as_index=False)
        .agg(rows=("antibiotic", "size"), countries=("iso3", "nunique"),
             antibiotics=("antibiotic", "nunique"))
    )
    display(version_audit)"""),
    md("""## Visual audit 1 — Coverage, testing and resistance structure

These figures are generated before modelling so that limited reporting, small testing denominators, breakpoint-version changes and unusual antibiotic distributions are visible rather than hidden inside a summary table.

The visual suite includes country–year coverage, antibiotic coverage, testing-volume distribution, breakpoint-version composition, resistance distributions, weighted resistance trends and the latest country–antibiotic heatmap."""),
    code("""visualisation_files = []
stage_files = plot_development_workflow(OUTPUT_DIR)
if DATA_READY:
    stage_files.extend(plot_data_overview(ears, OUTPUT_DIR))
else:
    print("Data-audit figures will be generated after EARS-Net data are added.")
visualisation_files.extend(stage_files)
show_visualisations(stage_files, "Workflow and data audit")"""),
    md("""## Step 2 — Dimensionality-reduction judgment

PCA is not automatically required. The supervised model uses a modest number of interpretable engineered features, while categorical antibiotic and breakpoint-version fields are one-hot encoded. PCA would make prescribing and demographic contributions harder to explain. It may later be used only for visualising multivariate country resistance profiles, not as a default forecasting step."""),
    md("""## Step 3 — Label attribute

The resistance percentage at each future horizon is a labelled continuous outcome. The primary forecasting task is therefore **supervised regression**. A later descriptive clustering analysis can profile countries, but cluster membership will not be used as though it were an externally validated clinical risk label."""),
    md("""## Step 4 — Final modelling task

Three direct regression models are fitted separately:

- year `t` → resistance at `t+1`;
- year `t` → resistance at `t+3`;
- year `t` → resistance at `t+5`.

Direct models avoid repeatedly feeding one-year predictions back into the model, which can compound error. Five-year forecasts still require adequate historical coverage and should be reported with greater uncertainty."""),
    md("""## Step 5 — Problem statement and practical value

Country-level AMR forecasts often rely mainly on previous resistance. That approach does not show whether prescribing changes, antibiotic-family consumption and demographic shifts add useful early-warning information. This project tests those additions under chronological validation while preserving the laboratory definition used by each region.

The intended use is surveillance prioritisation and stewardship scenario planning—not individual treatment selection or causal attribution."""),
    md("""## Feature construction

The primary features include:

- current resistance and three annual lags;
- four-observation resistance trend and volatility;
- log testing volume;
- one- to three-year lags of matching antibiotic-family consumption;
- community and hospital consumption;
- lagged Access percentage, broad-to-narrow ratio and oral-to-parenteral ratio;
- lagged hospital Reserve-antibiotic percentage, where available;
- population, percentage aged 65+, urbanisation and optional economic/health-system indicators;
- breakpoint standard/version and a breakpoint-change flag.

The current official ESAC-Net workbook provides a historical Reserve percentage but not historical Access, broad-to-narrow or oral-to-parenteral series. Unavailable indicators remain missing and are excluded automatically; they are not imputed from invented values. Lagged prescribing variables reduce—but do not eliminate—reverse causation. Feature importance is predictive, not proof that a prescribing behaviour caused resistance."""),
    code("""if DATA_READY:
    panel_full = build_eu_panel(ears, consumption, demographics, mapping, horizons=(1, 3, 5))
    MODEL_START_YEAR = int(consumption.year.min())
    panel = panel_full[panel_full.year.ge(MODEL_START_YEAR)].copy()
    panel_full.to_csv(PROCESSED_DIR / "eu_ecoli_bsi_full_historical_panel.csv", index=False)
    panel.to_csv(PROCESSED_DIR / "eu_ecoli_bsi_forecasting_panel.csv", index=False)
    print("Model origin-year window:", MODEL_START_YEAR, "to", panel.year.max())
    print("Earlier EARS-Net rows are retained only to construct resistance lags.")
    print("Panel rows:", len(panel))
    print("Countries:", panel.iso3.nunique())
    print("Antibiotics:", panel.antibiotic.nunique())
    print("Years:", panel.year.min(), "to", panel.year.max())
    display(panel.head())
    display(panel[["iso3", "year", "antibiotic", "resistance_pct",
                   "resistance_trend4", "resistance_volatility4",
                   "family_consumption_lag1", "access_pct_lag1", "reserve_pct_lag1",
                   "target_resistance_h1", "target_resistance_h3",
                   "target_resistance_h5"]].isna().mean().mul(100).rename("missing_pct"))
else:
    print("Feature construction is ready and will run when official inputs are present.")"""),
    md("""## Visual audit 2 — Consumption, prescribing behaviour and demographics

These figures examine whether the proposed predictors have enough coverage and meaningful variation to justify modelling. Scatter plots display associations only; their fitted lines must not be interpreted as causal effects."""),
    code("""if DATA_READY:
    stage_files = plot_predictor_overview(panel, consumption, OUTPUT_DIR)
    visualisation_files.extend(stage_files)
    show_visualisations(stage_files, "Predictor exploration")
else:
    print("Predictor visualisations will be generated after the EU panel is constructed.")"""),
    md("""## Exploratory feature evidence — before model fitting

Feature usefulness is assessed in three separate layers:

1. **Data suitability:** missingness, number of unique values, variation and availability over time;
2. **Exploratory association:** Pearson correlation for linear association, Spearman correlation for monotonic association and mutual information for possible nonlinear dependence;
3. **Predictive contribution:** chronological feature-group ablation after the model is fitted.

Correlation and mutual information are screening evidence only. A feature is not retained merely because it correlates with future resistance; it must improve validation performance when added to the same model under the same chronological split."""),
    code("""feature_evidence_tables = {}
if DATA_READY:
    for horizon in (1, 3, 5):
        target = f"target_resistance_h{horizon}"
        if target in panel and panel[target].notna().any():
            paths_created, evidence = plot_feature_exploration(
                panel, OUTPUT_DIR, horizon=horizon
            )
            visualisation_files.extend(paths_created)
            feature_evidence_tables[horizon] = evidence
            evidence.to_csv(
                OUTPUT_DIR / f"feature_evidence_h{horizon}.csv", index=False
            )
            print(f"Feature evidence for the {horizon}-year horizon")
            display(evidence.head(20))
            show_visualisations(
                paths_created, f"Exploratory feature evidence — {horizon}-year horizon"
            )
else:
    print("Feature-evidence plots will be generated after the EU panel is constructed.")"""),
    md("""## Step 6 — Model training, tuning and testing

Candidate models are Ridge regression, Random Forest, Gradient Boosting and a small ANN. Hyperparameters are tuned with expanding-year `GridSearchCV`. Model families are compared on the penultimate target year; the last target year is evaluated once. Persistence is the mandatory benchmark.

An ANN is included because it may capture nonlinear relationships, but it will not be selected merely because it is more complex."""),
    code("""results = {}
if DATA_READY:
    for horizon in (1, 3, 5):
        try:
            result = fit_horizon(panel, horizon=horizon, output_dir=OUTPUT_DIR)
            results[horizon] = result
            result.performance.assign(Horizon=horizon).to_csv(
                OUTPUT_DIR / f"performance_h{horizon}.csv", index=False
            )
            result.predictions.to_csv(
                OUTPUT_DIR / f"test_predictions_h{horizon}.csv", index=False
            )
            result.feature_importance.to_csv(
                OUTPUT_DIR / f"permutation_importance_h{horizon}.csv", index=False
            )
            print(f"Horizon {horizon}: selected {result.selected_model_name}")
            display(result.performance.sort_values(["Split", "RMSE"]))
        except ValueError as exc:
            print(f"Horizon {horizon} not fitted: {exc}")
else:
    print("No models trained: official inputs have not yet been added.")"""),
    code("""if results:
    performance_all = pd.concat(
        [r.performance.assign(Horizon=h) for h, r in results.items()],
        ignore_index=True,
    )
    performance_all.to_csv(OUTPUT_DIR / "all_horizon_performance.csv", index=False)
    test_plot = performance_all[performance_all.Split.eq("Test")].copy()
    plt.figure(figsize=(9, 5))
    sns.barplot(data=test_plot, x="Horizon", y="RMSE", hue="Model")
    plt.title("Untouched temporal test RMSE by forecast horizon")
    plt.ylabel("RMSE, percentage points")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "test_rmse_by_horizon.png", dpi=180)
    plt.show()
else:
    print("Performance charts will appear after fitting.")"""),
    md("""## Does each added feature group improve the model?

This is the decisive feature test. The selected algorithm and its tuned hyperparameters are held constant while the predictor set changes. Every version uses the same chronological training, validation and untouched test years.

The base model contains resistance history, trend, volatility, testing volume and laboratory metadata. Consumption, prescribing behaviour and demographics are then added separately, followed by the full model. An addition is labelled **Improved** only when its RMSE is at least 1% lower than the base model; changes smaller than 1% are labelled **No material change**. Model-development decisions should be based on validation results. The final test result is confirmation, not another opportunity to choose features."""),
    code("""ablation_tables = []
if results:
    for horizon, fitted_result in results.items():
        ablation = evaluate_feature_groups(
            panel, horizon=horizon, fitted_result=fitted_result,
            material_change_pct=1.0,
        )
        ablation.to_csv(
            OUTPUT_DIR / f"feature_group_ablation_h{horizon}.csv", index=False
        )
        ablation_tables.append(ablation)

    feature_ablation = pd.concat(ablation_tables, ignore_index=True)
    feature_ablation.to_csv(
        OUTPUT_DIR / "feature_group_ablation_all_horizons.csv", index=False
    )

    validation_decisions = feature_ablation[
        feature_ablation.Split.eq("Validation")
        & ~feature_ablation.FeatureSet.isin(["Base resistance history", "Persistence"])
    ][["Horizon", "FeatureSet", "FeaturesUsed", "RMSE", "MAE", "R2",
       "RelativeRMSEChange_vs_Base_pct", "EvidenceLabel"]]
    display(validation_decisions.sort_values(["Horizon", "RMSE"]))

    for row in validation_decisions.itertuples():
        direction = "reduced" if row.RelativeRMSEChange_vs_Base_pct < 0 else "increased"
        print(
            f"H{row.Horizon} | {row.FeatureSet}: {row.EvidenceLabel}; "
            f"validation RMSE {direction} by "
            f"{abs(row.RelativeRMSEChange_vs_Base_pct):.1f}% versus base."
        )
else:
    print("Feature-group ablation will run after the horizon models are fitted.")"""),
    md("""### Individual-feature additions

Grouped results can hide the fact that only one member of a group is useful. The next table adds every consumption, prescribing and demographic variable separately to the same base model. This provides feature-level evidence without allowing the final test year to influence feature selection."""),
    code("""individual_ablation_tables = []
if results:
    for horizon, fitted_result in results.items():
        individual = evaluate_individual_feature_additions(
            panel, horizon=horizon, fitted_result=fitted_result,
            material_change_pct=1.0,
        )
        individual.to_csv(
            OUTPUT_DIR / f"individual_feature_ablation_h{horizon}.csv", index=False
        )
        individual_ablation_tables.append(individual)

    individual_feature_ablation = pd.concat(
        individual_ablation_tables, ignore_index=True
    )
    individual_feature_ablation.to_csv(
        OUTPUT_DIR / "individual_feature_ablation_all_horizons.csv", index=False
    )
    individual_validation = individual_feature_ablation[
        individual_feature_ablation.Split.eq("Validation")
        & individual_feature_ablation.Feature.ne("Base resistance history")
    ][["Horizon", "Feature", "Group", "RMSE", "MAE", "R2",
       "RelativeRMSEChange_vs_Base_pct", "EvidenceLabel"]]
    display(individual_validation.sort_values(["Horizon", "RMSE"]))
else:
    print("Individual-feature ablation will run after model fitting.")"""),
    md("""## Visual audit 3 — Model performance and forecast diagnostics

The final visual suite compares models and horizons, then checks whether apparently strong average performance hides calibration problems, biased residuals, weak antibiotics, difficult countries or dependence on a small number of features.

Generated outputs include RMSE comparisons, observed-versus-predicted plots, residual diagnostics, antibiotic-specific performance, calibration plots, permutation importance, country–antibiotic error heatmaps and interactive geographical maps of observed resistance, predicted resistance and forecast error."""),
    code("""if results:
    stage_files = plot_model_results(results, OUTPUT_DIR)
    stage_files.extend(plot_feature_ablation(feature_ablation, OUTPUT_DIR))
    stage_files.extend(plot_individual_feature_ablation(
        individual_feature_ablation, OUTPUT_DIR
    ))
    stage_files.extend(plot_importance_stability(results, OUTPUT_DIR))
    generated_maps = create_interactive_maps(results, OUTPUT_DIR)
    # Preserve and display packaged maps even when Plotly is unavailable in a
    # restricted execution environment. In Colab, Plotly regenerates them.
    if not generated_maps:
        generated_maps = sorted((OUTPUT_DIR / "figures").glob("21_map*.html"))
    stage_files.extend(generated_maps)
    visualisation_files.extend(stage_files)
    show_visualisations(stage_files, "Model performance and forecast diagnostics")
    visualisation_index = write_visualisation_index(visualisation_files, OUTPUT_DIR)
    print(f"Created {len(visualisation_files)} visualisation files.")
    display(pd.read_csv(visualisation_index))
else:
    print("Model diagnostic visualisations will be generated after models are fitted.")"""),
    md("""## Step 7 — Interpretation framework

When results are available, interpretation must answer:

1. Does the selected model beat persistence at each horizon?
2. How quickly does RMSE increase from one to five years?
3. Is performance consistent across antibiotics and countries?
4. Do consumption, prescribing and demographic features improve chronological validation RMSE when added separately?
5. Are forecasts sensitive to breakpoint-version changes?
6. Are prediction errors larger in countries with small testing denominators?

The notebook now answers the feature question using five complementary diagnostics:

- **missingness and variation** show whether a feature is measurable enough to use;
- **Pearson correlation** detects simple linear association;
- **Spearman correlation** detects increasing or decreasing rank relationships;
- **mutual information** screens for nonlinear dependence;
- **group ablation and permutation importance** test whether the information improves forecasts.

The ablation result has priority. A strong correlation with no validation improvement is not evidence that the feature helps forecasting. A validation improvement that disappears on the final test year should be reported as unstable. Negative permutation importance suggests that a feature may be noise, redundant or harmful in that fitted model. These are predictive findings, not causal claims."""),
    md("""## Bounded iterative feature and model optimisation

The extended search constructs 41 additional origin-time candidates covering consumption trajectory, prescribing-mix trajectory, surveillance testing intensity, reported resistant-isolate burden, demographic change, EU/EEA resistance context, resistance across other antibiotic groups, interactions, time and country effects.

For every horizon, the search:

1. compares Ridge, Random Forest, Extra Trees, Gradient Boosting, Histogram Gradient Boosting and ANN;
2. uses expanding-year validation folds before the latest target year;
3. accepts a candidate only if pooled validation RMSE improves by at least 0.5% and it wins in at least 60% of folds;
4. stops when no remaining group or shortlisted individual feature meets both rules;
5. evaluates the frozen specification on the latest year.

The latest year has been inspected in earlier project development, so it is described as a **confirmation year**, not a pristine external test. A newer EARS-Net release is still needed for fully external temporal validation."""),
    code("""optimization_dir = OUTPUT_DIR / "optimization"
optimization_performance_path = optimization_dir / "optimized_performance_all_horizons.csv"
if optimization_performance_path.exists():
    optimization_performance = pd.read_csv(optimization_performance_path)
    optimization_recommendations = pd.read_csv(
        optimization_dir / "forecast_recommendations.csv"
    )
    optimized_features = pd.read_csv(
        optimization_dir / "optimized_selected_features.csv"
    )
    display(optimization_performance)
    display(optimization_recommendations)
    display(optimized_features.groupby("Horizon")["Feature"].apply(list).to_frame())
    for horizon in (1, 3, 5):
        search_log = pd.read_csv(
            optimization_dir / f"iterative_search_log_h{horizon}.csv"
        )
        print(f"Accepted search steps for the {horizon}-year horizon")
        display(search_log[search_log.Accepted.eq(True)][
            ["Iteration", "Stage", "Candidate", "Model", "FeatureCount",
             "CV_RMSE", "ImprovementPct", "FoldWinRate"]
        ])
else:
    print("Run `python -m src.run_iterative_optimization` to create the optimized results.")"""),
    code("""optimization_figures = sorted(
    (optimization_dir / "figures").glob("*.png")
) if optimization_dir.exists() else []
show_visualisations(optimization_figures, "Iterative optimization diagnostics")"""),
    md("""### Optimisation decision

- **One year:** retain persistence. The new Ridge specification improved rolling validation but failed confirmation, demonstrating why the latest-period check is necessary.
- **Three years:** use Extra Trees with demographic-change, country and lagged consumption information.
- **Five years:** use Extra Trees with country, lagged consumption, testing/reporting-burden and origin-time information.

Country identity improves forecasts for countries represented in training, but limits transportability to new countries. Reported resistant isolates per 100,000 is a surveillance-burden indicator, not a population-incidence estimate corrected for laboratory coverage. All optimized relationships remain predictive rather than causal."""),
    md("""## Next regional stages

- **US:** construct a separate bloodstream/systemic *E. coli* outcome table under documented CLSI/FDA definitions and breakpoint adoption dates.
- **Japan:** construct a separate JANIS blood-isolate table with the exact CLSI edition used by each report.
- **External comparison:** compare direction, ranking and calibration only for comparable antibiotic–specimen definitions.
- **Hierarchical extension:** share global predictor effects while preserving regional intercepts, slopes and observation definitions.

No pooled global outcome will be created from aggregate percentages unless breakpoint comparability is demonstrated or raw MIC data become available."""),
]

for index, cell in enumerate(cells, start=1):
    # Jupyter's v4.5 schema requires a short unique cell identifier.
    cell["id"] = f"cell-{index:02d}"

notebook = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3"},
        "colab": {"provenance": []},
    },
    "cells": cells,
}

(ROOT / "notebooks" / "archive").mkdir(parents=True, exist_ok=True)
(ROOT / "notebooks" / "archive" / "Regional_Ecoli_BSI_AMR_Phase1.ipynb").write_text(
    json.dumps(notebook, indent=1, ensure_ascii=False), encoding="utf-8"
)
print(ROOT / "notebooks" / "archive" / "Regional_Ecoli_BSI_AMR_Phase1.ipynb")
