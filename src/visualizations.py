"""Scientifically scoped visualisations for the regional E. coli AMR project."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error
from sklearn.feature_selection import mutual_info_regression

from src.regional_amr import (
    BASE_NUMERIC_FEATURES, CONSUMPTION_FEATURES, DEMOGRAPHIC_FEATURES,
    PRESCRIBING_FEATURES,
)


def _folder(output_dir) -> Path:
    path = Path(output_dir) / "figures"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _finish(fig, path: Path):
    fig.tight_layout()
    fig.savefig(path, dpi=190, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_development_workflow(output_dir) -> list[Path]:
    """One-page visual explanation of the complete model-development sequence."""
    from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

    out = _folder(output_dir)
    steps = [
        ("1. Confirm data", "Coverage, missingness, counts\nand breakpoint versions"),
        ("2. Explore outcomes", "Resistance distributions, trends\nand country differences"),
        ("3. Engineer features", "Lags, trend, volatility, use,\nprescribing and demographics"),
        ("4. Test feature value", "Correlation + mutual information\n+ chronological group ablation"),
        ("5. Train models", "Ridge, forest, boosting and ANN\nwith expanding-year tuning"),
        ("6. Select honestly", "Penultimate-year validation;\nlast year untouched for testing"),
        ("7. Explain forecasts", "Calibration, residuals, importance,\nerror heatmaps and maps"),
    ]
    fig, ax = plt.subplots(figsize=(14, 10))
    ax.set_xlim(0, 10); ax.set_ylim(0, 10); ax.axis("off")
    positions = [(1, 8), (5.5, 8), (1, 5.7), (5.5, 5.7), (1, 3.4), (5.5, 3.4), (3.25, 1.1)]
    centers = []
    for (title, text), (x, y) in zip(steps, positions):
        box = FancyBboxPatch((x, y), 3.5, 1.35, boxstyle="round,pad=0.15",
                             facecolor="#EAF2F8", edgecolor="#2E6F9E", linewidth=1.6)
        ax.add_patch(box)
        ax.text(x + .18, y + .92, title, fontsize=12, fontweight="bold", va="center")
        ax.text(x + .18, y + .45, text, fontsize=10, va="center")
        centers.append((x + 1.75, y + .68))
    for start, end in zip(centers[:-1], centers[1:]):
        ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=14,
                                     linewidth=1.3, color="#555555",
                                     connectionstyle="arc3,rad=0.08"))
    ax.set_title("How the regional E. coli AMR forecasting model is developed", fontsize=17, pad=15)
    return [_finish(fig, out / "00_model_development_workflow.png")]


def feature_evidence_table(panel: pd.DataFrame, horizon: int = 1) -> pd.DataFrame:
    """Summarise feature quality and exploratory association with a future target."""
    target = f"target_resistance_h{horizon}"
    candidate = list(dict.fromkeys(
        BASE_NUMERIC_FEATURES + CONSUMPTION_FEATURES + PRESCRIBING_FEATURES + DEMOGRAPHIC_FEATURES
    ))
    candidate = [c for c in candidate if c in panel and panel[c].notna().any()]
    data = panel.loc[panel[target].notna(), candidate + [target]].copy()
    if not candidate or data.empty:
        return pd.DataFrame()
    X = data[candidate].apply(pd.to_numeric, errors="coerce")
    X_imputed = X.fillna(X.median()).fillna(0)
    nonconstant = [c for c in candidate if X_imputed[c].nunique() > 1]
    mi_map = {c: 0.0 for c in candidate}
    if nonconstant:
        mi = mutual_info_regression(X_imputed[nonconstant], data[target], random_state=42)
        mi_map.update(dict(zip(nonconstant, mi)))
    rows = []
    for feature in candidate:
        pair = data[[feature, target]].dropna()
        pearson = pair[feature].corr(pair[target], method="pearson") if pair[feature].nunique() > 1 else np.nan
        spearman = pair[feature].corr(pair[target], method="spearman") if pair[feature].nunique() > 1 else np.nan
        if feature in BASE_NUMERIC_FEATURES:
            group = "Resistance history/base"
        elif feature in CONSUMPTION_FEATURES:
            group = "Consumption"
        elif feature in PRESCRIBING_FEATURES:
            group = "Prescribing behaviour"
        else:
            group = "Demographics"
        rows.append({
            "Feature": feature, "Group": group, "Horizon": horizon,
            "AvailableN": int(pair.shape[0]), "MissingPct": float(100 * data[feature].isna().mean()),
            "UniqueValues": int(data[feature].nunique(dropna=True)),
            "StandardDeviation": float(data[feature].std()) if pair.shape[0] else np.nan,
            "Pearson": pearson, "Spearman": spearman,
            "MutualInformation": float(mi_map[feature]),
        })
    return pd.DataFrame(rows).sort_values("MutualInformation", ascending=False)


def plot_feature_exploration(
    panel: pd.DataFrame, output_dir, horizon: int = 1
) -> tuple[list[Path], pd.DataFrame]:
    """Exploratory feature diagnostics that precede model fitting."""
    out = _folder(output_dir)
    paths = []
    evidence = feature_evidence_table(panel, horizon)
    if evidence.empty:
        return paths, evidence
    target = f"target_resistance_h{horizon}"

    # These three data-quality plots do not change by forecast horizon, so create
    # them once when the one-year exploratory analysis is requested.
    if horizon == 1:
        missing = evidence.sort_values("MissingPct")
        fig, ax = plt.subplots(figsize=(10, max(6, .38 * len(missing))))
        colors = missing["MissingPct"].map(lambda v: "#E15759" if v > 30 else "#4E79A7")
        ax.barh(missing["Feature"], missing["MissingPct"], color=colors)
        ax.axvline(30, linestyle="--", color="black", linewidth=1, label="30% review threshold")
        ax.set_xlim(0, 100); ax.set_xlabel("Missing observations (%)")
        ax.set_title("Feature completeness before modelling"); ax.legend()
        paths.append(_finish(fig, out / "22_feature_missingness.png"))

    top_corr = evidence.assign(abs_spearman=evidence.Spearman.abs()).nlargest(15, "abs_spearman")
    corr_long = top_corr.melt(id_vars="Feature", value_vars=["Pearson", "Spearman"],
                              var_name="Correlation", value_name="Value")
    fig, ax = plt.subplots(figsize=(11, max(6, .45 * top_corr.shape[0])))
    sns.barplot(data=corr_long, y="Feature", x="Value", hue="Correlation", ax=ax)
    ax.axvline(0, color="black", linewidth=.8)
    ax.set_xlim(-1, 1); ax.set_title(f"Feature correlation with {horizon}-year resistance target")
    ax.set_xlabel("Correlation coefficient"); ax.set_ylabel("")
    paths.append(_finish(fig, out / f"23_feature_target_correlations_h{horizon}.png"))

    mi = evidence.nlargest(15, "MutualInformation").sort_values("MutualInformation")
    fig, ax = plt.subplots(figsize=(10, max(6, .42 * len(mi))))
    sns.barplot(data=mi, y="Feature", x="MutualInformation", hue="Group", dodge=False, ax=ax)
    ax.set_title(f"Nonlinear feature–target dependence: {horizon}-year horizon")
    ax.set_xlabel("Mutual information"); ax.set_ylabel("")
    ax.legend(title="Feature group", bbox_to_anchor=(1.02, 1), loc="upper left")
    paths.append(_finish(fig, out / f"24_mutual_information_h{horizon}.png"))

    if horizon == 1:
        distribution_features = evidence[evidence.MissingPct.lt(80)].Feature.head(12).tolist()
        ncols = 3; nrows = int(np.ceil(len(distribution_features) / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(15, 3.8 * nrows), squeeze=False)
        for ax, feature in zip(axes.flat, distribution_features):
            sns.histplot(panel[feature].dropna(), bins=25, kde=True, ax=ax, color="#59A14F")
            ax.set_title(feature); ax.set_ylabel("Rows")
        for ax in axes.flat[len(distribution_features):]:
            ax.axis("off")
        fig.suptitle("Feature distributions and potential skew", fontsize=15, y=1.01)
        paths.append(_finish(fig, out / "25_feature_distributions.png"))

    relationship_features = evidence[evidence.MissingPct.lt(60)].Feature.head(6).tolist()
    ncols = 2; nrows = int(np.ceil(len(relationship_features) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(13, 4.5 * nrows), squeeze=False)
    for ax, feature in zip(axes.flat, relationship_features):
        subset = panel[[feature, target]].dropna().copy()
        try:
            subset["bin"] = pd.qcut(subset[feature], q=min(10, subset[feature].nunique()), duplicates="drop")
            binned = subset.groupby("bin", observed=True).agg(
                feature_mean=(feature, "mean"), target_mean=(target, "mean"),
                target_sem=(target, "sem"), n=(target, "size"),
            ).reset_index()
            ax.errorbar(binned.feature_mean, binned.target_mean, yerr=binned.target_sem,
                        marker="o", capsize=3, linewidth=1.5)
            ax.set_xlabel(feature); ax.set_ylabel(f"Mean {horizon}-year resistance (%)")
            ax.set_title(f"Binned relationship: {feature}")
        except ValueError:
            ax.text(.5, .5, "Insufficient variation", ha="center", va="center")
            ax.axis("off")
    for ax in axes.flat[len(relationship_features):]:
        ax.axis("off")
    fig.suptitle("Exploratory feature–outcome relationships", fontsize=15, y=1.01)
    paths.append(_finish(fig, out / f"26_binned_feature_relationships_h{horizon}.png"))

    if horizon == 1:
        features = evidence.Feature.tolist()
        availability = panel.groupby("year")[features].apply(lambda x: 100 * x.notna().mean())
        fig, ax = plt.subplots(figsize=(max(11, .65 * len(features)),
                                        max(5, .45 * availability.shape[0])))
        sns.heatmap(availability, cmap="YlGnBu", vmin=0, vmax=100, ax=ax,
                    cbar_kws={"label": "Available (%)"})
        ax.set_title("Feature availability over time")
        ax.set_xlabel("Feature"); ax.set_ylabel("Year")
        paths.append(_finish(fig, out / "27_feature_availability_over_time.png"))
    return paths, evidence


def plot_data_overview(ears: pd.DataFrame, output_dir) -> list[Path]:
    """Coverage, laboratory, resistance-distribution and trend figures."""
    out = _folder(output_dir)
    paths = []

    coverage = ears.pivot_table(
        index="country", columns="year", values="antibiotic", aggfunc="nunique", fill_value=0
    )
    fig, ax = plt.subplots(figsize=(max(10, 0.65 * coverage.shape[1]),
                                    min(20, max(7, 0.32 * coverage.shape[0]))))
    sns.heatmap(coverage, cmap="Blues", linewidths=.15, ax=ax,
                cbar_kws={"label": "Antibiotics reported"})
    ax.set_title("EARS-Net bloodstream E. coli reporting coverage")
    ax.set_xlabel("Year"); ax.set_ylabel("Country")
    paths.append(_finish(fig, out / "01_country_year_coverage_heatmap.png"))

    antibiotic_coverage = (
        ears.groupby("antibiotic", as_index=False)
        .agg(countries=("iso3", "nunique"), country_years=("year", "size"),
             median_tested=("tested", "median"))
        .sort_values("countries")
    )
    fig, ax = plt.subplots(figsize=(10, max(5, .43 * len(antibiotic_coverage))))
    sns.barplot(data=antibiotic_coverage, y="antibiotic", x="countries", ax=ax, color="#4C78A8")
    ax.set_title("Country coverage by antibiotic")
    ax.set_xlabel("Number of countries"); ax.set_ylabel("")
    paths.append(_finish(fig, out / "02_antibiotic_country_coverage.png"))

    fig, ax = plt.subplots(figsize=(9, 5))
    sns.histplot(np.log10(ears["tested"].clip(lower=1)), bins=35, kde=True, ax=ax, color="#59A14F")
    ax.set_title("Distribution of testing volume")
    ax.set_xlabel("log10(number of isolates tested)"); ax.set_ylabel("Rows")
    paths.append(_finish(fig, out / "03_testing_volume_distribution.png"))

    version = ears.pivot_table(
        index="year", columns="ast_version", values="antibiotic", aggfunc="size", fill_value=0
    )
    fig, ax = plt.subplots(figsize=(11, 5))
    version.plot.area(ax=ax, alpha=.85, linewidth=.5)
    ax.set_title("Reported breakpoint-version composition over time")
    ax.set_xlabel("Year"); ax.set_ylabel("Country–antibiotic rows")
    ax.legend(title="AST version", bbox_to_anchor=(1.02, 1), loc="upper left")
    paths.append(_finish(fig, out / "04_breakpoint_version_timeline.png"))

    order = (
        ears.groupby("antibiotic")["resistance_pct"].median().sort_values().index
    )
    fig, ax = plt.subplots(figsize=(11, max(6, .5 * len(order))))
    sns.boxplot(data=ears, y="antibiotic", x="resistance_pct", order=order,
                showfliers=False, ax=ax, color="#F28E2B")
    ax.set_xlim(0, 100)
    ax.set_title("Distribution of reported resistance by antibiotic")
    ax.set_xlabel("Resistance (%)"); ax.set_ylabel("")
    paths.append(_finish(fig, out / "05_resistance_distribution_by_antibiotic.png"))

    weighted = (
        ears.groupby(["year", "antibiotic"], as_index=False)
        .agg(resistant=("resistant", "sum"), tested=("tested", "sum"))
    )
    weighted["weighted_resistance_pct"] = 100 * weighted["resistant"] / weighted["tested"]
    top_agents = (
        ears.groupby("antibiotic")["iso3"].nunique().nlargest(min(8, ears.antibiotic.nunique())).index
    )
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.lineplot(data=weighted[weighted.antibiotic.isin(top_agents)], x="year",
                 y="weighted_resistance_pct", hue="antibiotic", marker="o", ax=ax)
    ax.set_ylim(0, 100)
    ax.set_title("EU/EEA weighted resistance trends: highest-coverage antibiotics")
    ax.set_xlabel("Year"); ax.set_ylabel("Resistance (%)")
    ax.legend(title="Antibiotic", bbox_to_anchor=(1.02, 1), loc="upper left")
    paths.append(_finish(fig, out / "06_weighted_resistance_trends.png"))

    latest_year = int(ears.year.max())
    latest = ears[ears.year.eq(latest_year)]
    latest_agents = latest.groupby("antibiotic")["iso3"].nunique().nlargest(min(12, latest.antibiotic.nunique())).index
    heat = latest[latest.antibiotic.isin(latest_agents)].pivot(
        index="country", columns="antibiotic", values="resistance_pct"
    )
    fig, ax = plt.subplots(figsize=(max(11, .85 * heat.shape[1]),
                                    min(20, max(7, .34 * heat.shape[0]))))
    sns.heatmap(heat, cmap="YlOrRd", vmin=0, vmax=100, linewidths=.15, ax=ax,
                cbar_kws={"label": "Resistance (%)"})
    ax.set_title(f"Latest reported resistance profile ({latest_year})")
    ax.set_xlabel("Antibiotic"); ax.set_ylabel("Country")
    paths.append(_finish(fig, out / "07_latest_country_antibiotic_heatmap.png"))
    return paths


def plot_predictor_overview(
    panel: pd.DataFrame, consumption: pd.DataFrame, output_dir
) -> list[Path]:
    """Consumption, prescribing-behaviour, demographic and correlation figures."""
    out = _folder(output_dir)
    paths = []

    sector = (
        consumption.groupby(["year", "sector"], as_index=False)["ddd_per_1000_per_day"].median()
    )
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.lineplot(data=sector, x="year", y="ddd_per_1000_per_day", hue="sector",
                 marker="o", ax=ax)
    ax.set_title("Median reported antibiotic consumption by healthcare sector")
    ax.set_xlabel("Year"); ax.set_ylabel("Median DDD per 1,000 inhabitants per day")
    paths.append(_finish(fig, out / "08_consumption_by_sector_trend.png"))

    behaviour = (
        consumption.groupby("year", as_index=False)
        .agg(access_pct=("access_pct", "median"),
             broad_narrow_ratio=("broad_narrow_ratio", "median"),
             oral_parenteral_ratio=("oral_parenteral_ratio", "median"),
             reserve_pct=("reserve_pct", "median"))
    )
    labels = {
        "access_pct": "Access antibiotics (%)",
        "broad_narrow_ratio": "Broad-to-narrow ratio",
        "oral_parenteral_ratio": "Oral-to-parenteral ratio",
        "reserve_pct": "Reserve antibiotics in hospital use (%)",
    }
    available_behaviour = [col for col in labels if behaviour[col].notna().any()]
    if available_behaviour:
        fig, axes = plt.subplots(
            1, len(available_behaviour),
            figsize=(5 * len(available_behaviour), 4.5), squeeze=False,
        )
        for ax, col in zip(axes.flat, available_behaviour):
            label = labels[col]
            sns.lineplot(data=behaviour, x="year", y=col, marker="o", ax=ax)
            ax.set_title(label); ax.set_xlabel("Year"); ax.set_ylabel(label)
        paths.append(_finish(fig, out / "09_prescribing_behaviour_trends.png"))

    latest_year = int(consumption.year.max())
    latest = consumption[consumption.year.eq(latest_year)]
    family = latest.pivot_table(index="country", columns="antibiotic_family",
                                values="ddd_per_1000_per_day", aggfunc="sum")
    if not family.empty:
        fig, ax = plt.subplots(figsize=(max(10, .8 * family.shape[1]),
                                        min(20, max(7, .34 * family.shape[0]))))
        sns.heatmap(family, cmap="PuBuGn", linewidths=.15, ax=ax,
                    cbar_kws={"label": "DDD per 1,000/day"})
        ax.set_title(f"Antibiotic-family consumption by country ({latest_year})")
        ax.set_xlabel("Antibiotic family"); ax.set_ylabel("Country")
        paths.append(_finish(fig, out / "10_latest_consumption_heatmap.png"))

    scatter = panel.dropna(subset=["family_consumption_lag1", "target_resistance_h1"])
    if not scatter.empty:
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.scatterplot(data=scatter, x="family_consumption_lag1", y="target_resistance_h1",
                        size="log_tested", sizes=(15, 160), alpha=.55, ax=ax)
        sns.regplot(data=scatter, x="family_consumption_lag1", y="target_resistance_h1",
                    scatter=False, color="black", line_kws={"linewidth": 1.5}, ax=ax)
        ax.set_title("Lagged family consumption versus next-year resistance")
        ax.set_xlabel("Previous-year family consumption")
        ax.set_ylabel("Next-year resistance (%)")
        paths.append(_finish(fig, out / "11_consumption_vs_next_resistance.png"))

    demographic = panel.dropna(subset=["age65_pct", "target_resistance_h1"])
    if not demographic.empty:
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.scatterplot(data=demographic, x="age65_pct", y="target_resistance_h1",
                        size="log_tested", sizes=(15, 160), alpha=.55, ax=ax)
        sns.regplot(data=demographic, x="age65_pct", y="target_resistance_h1",
                    scatter=False, color="black", line_kws={"linewidth": 1.5}, ax=ax)
        ax.set_title("Population ageing versus next-year resistance")
        ax.set_xlabel("Population aged 65+ (%)"); ax.set_ylabel("Next-year resistance (%)")
        paths.append(_finish(fig, out / "12_age65_vs_next_resistance.png"))

    numeric = [
        "resistance_pct", "resistance_lag1", "resistance_trend4",
        "resistance_volatility4", "family_consumption_lag1",
        "consumption_community", "consumption_hospital", "access_pct_lag1",
        "broad_narrow_ratio_lag1", "reserve_pct_lag1", "age65_pct", "urban_pct",
        "health_expenditure_pct_gdp", "target_resistance_h1",
    ]
    numeric = [c for c in numeric if c in panel and panel[c].notna().any()]
    if len(numeric) >= 3:
        corr = panel[numeric].corr()
        fig, ax = plt.subplots(figsize=(max(9, .75 * len(numeric)), max(7, .65 * len(numeric))))
        sns.heatmap(corr, cmap="vlag", center=0, vmin=-1, vmax=1, annot=True,
                    fmt=".2f", square=True, ax=ax)
        ax.set_title("Predictor correlation matrix")
        paths.append(_finish(fig, out / "13_predictor_correlation_heatmap.png"))
    return paths


def _calibration_table(predictions: pd.DataFrame) -> pd.DataFrame:
    data = predictions.copy()
    target = f"target_resistance_h{int(data.horizon.iloc[0])}"
    bins = min(10, max(2, data["prediction"].nunique()))
    data["prediction_bin"] = pd.qcut(data["prediction"], q=bins, duplicates="drop")
    return data.groupby("prediction_bin", observed=True, as_index=False).agg(
        mean_prediction=("prediction", "mean"),
        mean_observed=(target, "mean"),
        n=(target, "size"),
    )


def plot_model_results(results: dict, output_dir) -> list[Path]:
    """Performance, calibration, error and feature-importance figures."""
    out = _folder(output_dir)
    paths = []
    if not results:
        return paths
    performance = pd.concat(
        [r.performance.assign(Horizon=h) for h, r in results.items()], ignore_index=True
    )
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.barplot(data=performance, x="Horizon", y="RMSE", hue="Model", errorbar=None, ax=ax)
    ax.set_title("Model RMSE across forecast horizons and evaluation splits")
    ax.set_xlabel("Forecast horizon (years)"); ax.set_ylabel("RMSE, percentage points")
    ax.legend(title="Model", bbox_to_anchor=(1.02, 1), loc="upper left")
    paths.append(_finish(fig, out / "14_model_rmse_all_horizons.png"))

    horizons = sorted(results)
    fig, axes = plt.subplots(1, len(horizons), figsize=(6 * len(horizons), 5), squeeze=False)
    for ax, horizon in zip(axes[0], horizons):
        pred = results[horizon].predictions
        target = f"target_resistance_h{horizon}"
        sns.scatterplot(data=pred, x=target, y="prediction", alpha=.65, ax=ax)
        ax.plot([0, 100], [0, 100], "--", color="black", linewidth=1)
        ax.set_xlim(0, 100); ax.set_ylim(0, 100)
        ax.set_title(f"{horizon}-year observed vs predicted")
        ax.set_xlabel("Observed resistance (%)"); ax.set_ylabel("Predicted resistance (%)")
    paths.append(_finish(fig, out / "15_observed_vs_predicted.png"))

    fig, axes = plt.subplots(2, len(horizons), figsize=(6 * len(horizons), 9), squeeze=False)
    for j, horizon in enumerate(horizons):
        pred = results[horizon].predictions.copy()
        target = f"target_resistance_h{horizon}"
        pred["residual"] = pred[target] - pred["prediction"]
        sns.histplot(data=pred, x="residual", bins=25, kde=True, ax=axes[0, j])
        axes[0, j].axvline(0, color="black", linestyle="--")
        axes[0, j].set_title(f"{horizon}-year residual distribution")
        sns.scatterplot(data=pred, x="prediction", y="residual", alpha=.65, ax=axes[1, j])
        axes[1, j].axhline(0, color="black", linestyle="--")
        axes[1, j].set_title(f"{horizon}-year residuals vs prediction")
        axes[1, j].set_xlabel("Predicted resistance (%)")
        axes[1, j].set_ylabel("Observed − predicted")
    paths.append(_finish(fig, out / "16_residual_diagnostics.png"))

    per_agent = []
    for horizon, result in results.items():
        pred = result.predictions.copy()
        target = f"target_resistance_h{horizon}"
        for antibiotic, group in pred.groupby("antibiotic"):
            if len(group) < 2:
                continue
            per_agent.extend([
                {"Horizon": horizon, "Antibiotic": antibiotic, "Model": "Selected",
                 "RMSE": mean_squared_error(group[target], group["prediction"]) ** .5},
                {"Horizon": horizon, "Antibiotic": antibiotic, "Model": "Persistence",
                 "RMSE": mean_squared_error(group[target], group["persistence_prediction"]) ** .5},
            ])
    per_agent = pd.DataFrame(per_agent)
    if not per_agent.empty:
        for horizon in sorted(per_agent.Horizon.unique()):
            data = per_agent[per_agent.Horizon.eq(horizon)].sort_values("RMSE")
            fig, ax = plt.subplots(figsize=(11, max(6, .48 * data.Antibiotic.nunique())))
            sns.barplot(data=data, y="Antibiotic", x="RMSE", hue="Model", ax=ax)
            ax.set_title(f"{horizon}-year RMSE by antibiotic")
            ax.set_xlabel("RMSE, percentage points"); ax.set_ylabel("")
            paths.append(_finish(fig, out / f"17_rmse_by_antibiotic_h{horizon}.png"))

    fig, axes = plt.subplots(1, len(horizons), figsize=(6 * len(horizons), 5), squeeze=False)
    for ax, horizon in zip(axes[0], horizons):
        calibration = _calibration_table(results[horizon].predictions)
        sns.lineplot(data=calibration, x="mean_prediction", y="mean_observed", marker="o", ax=ax)
        ax.plot([0, 100], [0, 100], "--", color="black", linewidth=1)
        ax.set_xlim(0, 100); ax.set_ylim(0, 100)
        ax.set_title(f"{horizon}-year calibration")
        ax.set_xlabel("Mean predicted resistance (%)")
        ax.set_ylabel("Mean observed resistance (%)")
    paths.append(_finish(fig, out / "18_calibration_by_horizon.png"))

    fig, axes = plt.subplots(1, len(horizons), figsize=(6 * len(horizons), 6), squeeze=False)
    for ax, horizon in zip(axes[0], horizons):
        importance = results[horizon].feature_importance.head(15).sort_values("ImportanceMean")
        ax.barh(importance["Feature"], importance["ImportanceMean"],
                xerr=importance["ImportanceSD"], color="#4C78A8", alpha=.9)
        ax.axvline(0, color="black", linewidth=.8)
        ax.set_title(f"{horizon}-year permutation importance")
        ax.set_xlabel("Increase in RMSE after permutation")
    paths.append(_finish(fig, out / "19_permutation_importance.png"))

    for horizon, result in results.items():
        pred = result.predictions.copy()
        target = f"target_resistance_h{horizon}"
        pred["absolute_error"] = (pred[target] - pred["prediction"]).abs()
        heat = pred.pivot_table(index="country", columns="antibiotic",
                                values="absolute_error", aggfunc="mean")
        if not heat.empty:
            fig, ax = plt.subplots(figsize=(max(11, .8 * heat.shape[1]),
                                            min(20, max(7, .34 * heat.shape[0]))))
            sns.heatmap(heat, cmap="Reds", linewidths=.15, ax=ax,
                        cbar_kws={"label": "Absolute error (percentage points)"})
            ax.set_title(f"{horizon}-year forecast-error heatmap")
            ax.set_xlabel("Antibiotic"); ax.set_ylabel("Country")
            paths.append(_finish(fig, out / f"20_error_heatmap_h{horizon}.png"))
    return paths


def plot_feature_ablation(ablation: pd.DataFrame, output_dir) -> list[Path]:
    """Show whether each pre-specified feature group improves held-out RMSE."""
    out = _folder(output_dir)
    paths = []
    if ablation.empty:
        return paths

    for horizon in sorted(ablation.Horizon.unique()):
        data = ablation[
            ablation.Horizon.eq(horizon) & ~ablation.FeatureSet.eq("Persistence")
        ].copy()
        order = (
            data[data.Split.eq("Validation")].sort_values("RMSE").FeatureSet.tolist()
        )
        fig, ax = plt.subplots(figsize=(11, 6))
        sns.barplot(data=data, y="FeatureSet", x="RMSE", hue="Split", order=order, ax=ax)
        persistence = ablation[
            ablation.Horizon.eq(horizon) & ablation.FeatureSet.eq("Persistence")
            & ablation.Split.eq("Test")
        ]
        if not persistence.empty:
            ax.axvline(float(persistence.RMSE.iloc[0]), linestyle="--", color="black",
                       label="Test persistence")
        ax.set_title(f"{horizon}-year feature-group ablation")
        ax.set_xlabel("RMSE, percentage points"); ax.set_ylabel("")
        ax.legend(title="Evaluation")
        paths.append(_finish(fig, out / f"28_feature_group_ablation_h{horizon}.png"))

    test = ablation[
        ablation.Split.eq("Test")
        & ~ablation.FeatureSet.isin(["Base resistance history", "Persistence"])
    ]
    if not test.empty:
        heat = test.pivot(index="FeatureSet", columns="Horizon",
                          values="RelativeRMSEChange_vs_Base_pct")
        fig, ax = plt.subplots(figsize=(8, max(4.5, .65 * heat.shape[0])))
        sns.heatmap(heat, cmap="RdYlGn_r", center=0, annot=True, fmt=".1f", ax=ax,
                    cbar_kws={"label": "RMSE change versus base (%)"})
        ax.set_title("Does each feature addition help on the untouched test year?")
        ax.set_xlabel("Forecast horizon (years)"); ax.set_ylabel("")
        paths.append(_finish(fig, out / "29_feature_group_test_delta_heatmap.png"))

        decision = (
            test.groupby(["Horizon", "EvidenceLabel"], as_index=False).size()
            .pivot(index="Horizon", columns="EvidenceLabel", values="size").fillna(0)
        )
        fig, ax = plt.subplots(figsize=(9, 5))
        decision.plot(kind="bar", stacked=True, ax=ax, colormap="Set2")
        ax.set_title("Feature-addition evidence labels by horizon")
        ax.set_xlabel("Forecast horizon (years)"); ax.set_ylabel("Number of feature sets")
        ax.legend(title="Evidence label", bbox_to_anchor=(1.02, 1), loc="upper left")
        paths.append(_finish(fig, out / "30_feature_group_decision_summary.png"))
    return paths


def plot_importance_stability(results: dict, output_dir) -> list[Path]:
    """Compare permutation importance across 1-, 3- and 5-year models."""
    out = _folder(output_dir)
    if not results:
        return []
    table = pd.concat([
        result.feature_importance[["Feature", "ImportanceMean"]].assign(Horizon=horizon)
        for horizon, result in results.items()
    ], ignore_index=True)
    pivot = table.pivot(index="Feature", columns="Horizon", values="ImportanceMean").fillna(0)
    keep = pivot.abs().max(axis=1).nlargest(min(20, len(pivot))).index
    pivot = pivot.loc[keep]
    fig, ax = plt.subplots(figsize=(9, max(6, .42 * len(pivot))))
    sns.heatmap(pivot, cmap="vlag", center=0, annot=True, fmt=".2f", ax=ax,
                cbar_kws={"label": "Permutation importance"})
    ax.set_title("Feature importance stability across forecast horizons")
    ax.set_xlabel("Forecast horizon (years)"); ax.set_ylabel("Feature")
    return [_finish(fig, out / "31_importance_stability_across_horizons.png")]


def plot_individual_feature_ablation(ablation: pd.DataFrame, output_dir) -> list[Path]:
    """Show the validation and test impact of adding each feature separately."""
    out = _folder(output_dir)
    paths = []
    additions = ablation[ablation.Feature.ne("Base resistance history")].copy()
    for horizon in sorted(additions.Horizon.unique()):
        data = additions[
            additions.Horizon.eq(horizon) & additions.Split.eq("Validation")
        ].sort_values("RelativeRMSEChange_vs_Base_pct")
        if data.empty:
            continue
        colors = data.EvidenceLabel.map({
            "Improved": "#59A14F", "Worsened": "#E15759",
            "No material change": "#B8B8B8",
        }).fillna("#4E79A7")
        fig, ax = plt.subplots(figsize=(11, max(6, .45 * len(data))))
        ax.barh(data.Feature, data.RelativeRMSEChange_vs_Base_pct, color=colors)
        ax.axvline(0, color="black", linewidth=1)
        ax.axvline(-1, color="#2F7D32", linestyle="--", linewidth=1,
                   label="1% improvement threshold")
        ax.axvline(1, color="#A12A2A", linestyle="--", linewidth=1,
                   label="1% worsening threshold")
        ax.set_title(f"Individual feature additions: {horizon}-year validation")
        ax.set_xlabel("RMSE change versus base model (%) — lower is better")
        ax.set_ylabel(""); ax.legend(loc="best")
        paths.append(_finish(fig, out / f"32_individual_feature_validation_h{horizon}.png"))

    validation = additions[additions.Split.eq("Validation")]
    if not validation.empty:
        heat = validation.pivot(index="Feature", columns="Horizon",
                                values="RelativeRMSEChange_vs_Base_pct")
        order = heat.mean(axis=1).sort_values().index
        heat = heat.loc[order]
        fig, ax = plt.subplots(figsize=(8, max(6, .43 * len(heat))))
        sns.heatmap(heat, cmap="RdYlGn_r", center=0, annot=True, fmt=".1f", ax=ax,
                    cbar_kws={"label": "Validation RMSE change vs base (%)"})
        ax.set_title("Individual feature value across forecast horizons")
        ax.set_xlabel("Forecast horizon (years)"); ax.set_ylabel("Feature")
        paths.append(_finish(fig, out / "33_individual_feature_validation_heatmap.png"))
    return paths


def create_interactive_maps(results: dict, output_dir):
    """Create observed, predicted and absolute-error choropleths as HTML files."""
    try:
        import plotly.express as px
    except ImportError:
        return []
    out = _folder(output_dir)
    saved = []
    for horizon, result in results.items():
        pred = result.predictions.copy()
        target = f"target_resistance_h{horizon}"
        pred["absolute_error"] = (pred[target] - pred["prediction"]).abs()
        summary = pred.groupby(["iso3", "country"], as_index=False).agg(
            observed=(target, "mean"), predicted=("prediction", "mean"),
            absolute_error=("absolute_error", "mean"),
        )
        for variable, title in [
            ("observed", "Observed resistance"),
            ("predicted", "Predicted resistance"),
            ("absolute_error", "Forecast absolute error"),
        ]:
            kwargs = {
                "data_frame": summary,
                "locations": "iso3",
                "color": variable,
                "hover_name": "country",
                "color_continuous_scale": "YlOrRd",
                "title": f"{horizon}-year {title.lower()}: country mean across antibiotics",
                "labels": {variable: title + " (%)"},
            }
            if variable != "absolute_error":
                kwargs["range_color"] = (0, 100)
            fig = px.choropleth(**kwargs)
            path = out / f"21_map_h{horizon}_{variable}.html"
            fig.write_html(path, include_plotlyjs="cdn")
            saved.append(path)
    return saved


def write_visualisation_index(paths, output_dir):
    rows = []
    for path in paths:
        path = Path(path)
        rows.append({"file": path.name, "relative_path": str(path), "format": path.suffix.lstrip(".")})
    index = pd.DataFrame(rows).drop_duplicates("relative_path")
    target = Path(output_dir) / "visualisation_index.csv"
    index.to_csv(target, index=False)
    return target
