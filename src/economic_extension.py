"""Reproducible CEA and BIA bridge for the E. coli BSI forecasting project.

R is the authoritative economic implementation. This Python module is a
formula-equivalent validator and creates reviewable reference outputs.
"""
from __future__ import annotations

import hashlib
import json
import platform
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
FORECAST = ROOT / "outputs/notebook_ml/future_forecasts.csv"
EARS = ROOT / "data/input/ears_net_ecoli_blood.csv"
DEMOGRAPHICS = ROOT / "data/input/demographics.csv"
COVERAGE = ROOT / "data/input/ecdc_earsnet_coverage_2023.csv"
POPULATION = ROOT / "data/input/eurostat_population_projection_2026.csv"
PARAMETERS = ROOT / "economic_model/parameters/economic_parameters.csv"
BIA_PARAMETERS = ROOT / "economic_model/parameters/bia_parameters.csv"
CONFIG = ROOT / "economic_model/model_config.csv"
REFERENCE = ROOT / "outputs/economic_python_reference"


def parameter_values(frame: pd.DataFrame, column: str = "base") -> dict[str, float]:
    return frame.set_index("parameter")[column].astype(float).to_dict()


def load_inputs():
    forecasts = pd.read_csv(FORECAST)
    ears = pd.read_csv(EARS)
    demographics = pd.read_csv(DEMOGRAPHICS)
    coverage = pd.read_csv(COVERAGE)
    population = pd.read_csv(POPULATION)
    parameters = pd.read_csv(PARAMETERS)
    config_frame = pd.read_csv(CONFIG, dtype=str)
    config = dict(zip(config_frame.setting, config_frame.value, strict=True))
    return forecasts, ears, demographics, coverage, population, parameters, config


def prepare_cohort(
    forecasts: pd.DataFrame,
    ears: pd.DataFrame,
    demographics: pd.DataFrame,
    coverage: pd.DataFrame,
    population: pd.DataFrame,
    parameters: pd.DataFrame,
    config: dict[str, str],
) -> pd.DataFrame:
    p = parameter_values(parameters)
    target_year = int(config["forecast_target_year"])
    origin_year = int(config["surveillance_reference_year"])
    cohort = forecasts.loc[
        (forecasts.Horizon == int(config["forecast_horizon_years"]))
        & (forecasts.ForecastTargetYear == target_year)
        & (forecasts.antibiotic == config["antibiotic"])
    ].copy()
    if cohort.empty or cohort.iso3.duplicated().any():
        raise ValueError("Economic cohort must contain one row per country")

    # ECDC estimates national incidence by correcting reported cases for the
    # national population coverage reported by each country. Use the maximum
    # antibiotic-specific denominator as the closest available count of unique
    # reported invasive E. coli episodes; do not use target-antibiotic `tested`
    # as the economic population.
    origin = ears.loc[ears.year.eq(origin_year)].copy()
    total_reported = (
        origin.groupby("iso3", as_index=False).tested.max()
        .rename(columns={"tested": "ReportedEcoliBSIEpisodesOrigin"})
    )
    pop_origin = demographics.loc[
        demographics.year.eq(origin_year), ["iso3", "population"]
    ].rename(columns={"population": "PopulationOrigin"})
    pop_target = population.loc[
        population.year.eq(target_year), ["iso3", "population"]
    ].rename(columns={"population": "ProjectedPopulationTargetYear"})
    cohort = cohort.merge(total_reported, on="iso3", how="left", validate="one_to_one")
    cohort = cohort.merge(coverage, on=["country", "iso3"], how="left", validate="one_to_one")
    cohort = cohort.merge(pop_origin, on="iso3", how="left", validate="one_to_one")
    cohort = cohort.merge(pop_target, on="iso3", how="left", validate="one_to_one")
    valid = (
        cohort.population_coverage_pct.gt(0)
        & cohort.ReportedEcoliBSIEpisodesOrigin.gt(0)
        & cohort.PopulationOrigin.gt(0)
        & cohort.ProjectedPopulationTargetYear.gt(0)
    )
    cohort = cohort.loc[valid].copy()
    cohort["CoverageAdjustedEcoliBSIEpisodesOrigin"] = (
        cohort.ReportedEcoliBSIEpisodesOrigin / (cohort.population_coverage_pct / 100)
    )
    cohort["TotalEcoliBSIIncidencePer100kOrigin"] = (
        cohort.CoverageAdjustedEcoliBSIEpisodesOrigin / cohort.PopulationOrigin * 100_000
    )
    cohort["EconomicCohortEpisodes"] = (
        cohort.TotalEcoliBSIIncidencePer100kOrigin
        * cohort.ProjectedPopulationTargetYear / 100_000
        * p["future_bsi_incidence_multiplier"]
    )

    rep_cols = [
        "geographical_representativeness", "hospital_representativeness",
        "isolate_representativeness",
    ]
    any_low = cohort[rep_cols].eq("Low").any(axis=1)
    any_medium = cohort[rep_cols].eq("Medium").any(axis=1) & ~any_low
    cohort["RepresentativenessCaution"] = np.select(
        [any_low, any_medium], ["Low", "Medium"], default="High"
    )
    cohort["IncidenceMeasurementCV"] = np.sqrt(
        p["coverage_cv_no_caution"] ** 2
        + any_medium.astype(float) * p["coverage_cv_medium_representation"] ** 2
        + any_low.astype(float) * p["coverage_cv_low_representation"] ** 2
    )
    cohort["ForecastIncreasePP"] = (
        cohort.RecommendedForecastPct - cohort.LatestObservedResistancePct
    )
    adequate = cohort.tested.ge(p["minimum_reported_tested"])
    cohort["PersistenceProgrammeTriggered"] = adequate & cohort.PersistenceForecastPct.ge(
        p["resistance_trigger_pct"]
    )
    cohort["ForecastProgrammeTriggered"] = adequate & (
        cohort.RecommendedForecastPct.ge(p["resistance_trigger_pct"])
        | cohort.ForecastIncreasePP.ge(p["increase_trigger_pp"])
    )
    return cohort.sort_values("RecommendedForecastPct", ascending=False).reset_index(drop=True)


def deterministic(cohort: pd.DataFrame, parameters: pd.DataFrame) -> pd.DataFrame:
    p = parameter_values(parameters)
    out = cohort.copy()
    out["ForecastResistantEpisodes"] = (
        out.EconomicCohortEpisodes * out.RecommendedForecastPct / 100
    )
    p0 = p["baseline_bsi_mortality_probability"]
    odds_ratio = p["intervention_mortality_odds_ratio"]
    p1 = odds_ratio * p0 / (1 - p0 + odds_ratio * p0)
    out["UsualCareDeaths"] = out.ForecastResistantEpisodes * p0
    out["PersistenceDeaths"] = np.where(
        out.PersistenceProgrammeTriggered, out.ForecastResistantEpisodes * p1,
        out.UsualCareDeaths,
    )
    out["ForecastDeaths"] = np.where(
        out.ForecastProgrammeTriggered, out.ForecastResistantEpisodes * p1,
        out.UsualCareDeaths,
    )
    programme_cost = (
        p["programme_fixed_cost_per_country"]
        + p["programme_variable_cost_per_bsi_episode"] * out.EconomicCohortEpisodes
    )
    out["PersistenceStrategyCostEUR"] = np.where(
        out.PersistenceProgrammeTriggered, programme_cost, 0
    )
    out["ForecastStrategyCostEUR"] = np.where(
        out.ForecastProgrammeTriggered, programme_cost, 0
    )
    out["IncrementalCostForecastVsUsualEUR"] = out.ForecastStrategyCostEUR
    out["IncrementalCostForecastVsPersistenceEUR"] = (
        out.ForecastStrategyCostEUR - out.PersistenceStrategyCostEUR
    )
    out["QALYGainedForecastVsUsual"] = (
        out.UsualCareDeaths - out.ForecastDeaths
    ) * p["qaly_loss_per_bsi_death"]
    out["QALYGainedForecastVsPersistence"] = (
        out.PersistenceDeaths - out.ForecastDeaths
    ) * p["qaly_loss_per_bsi_death"]
    out["IncrementalNMBForecastVsUsualEUR"] = (
        p["willingness_to_pay_per_qaly"] * out.QALYGainedForecastVsUsual
        - out.IncrementalCostForecastVsUsualEUR
    )
    out["IncrementalNMBForecastVsPersistenceEUR"] = (
        p["willingness_to_pay_per_qaly"] * out.QALYGainedForecastVsPersistence
        - out.IncrementalCostForecastVsPersistenceEUR
    )
    return out


def _sample_parameters(rng, parameters, simulations):
    draws = {}
    for row in parameters.itertuples(index=False):
        if row.distribution == "fixed":
            x = np.full(simulations, float(row.base))
        elif row.distribution == "triangular":
            x = rng.triangular(float(row.low), float(row.base), float(row.high), simulations)
        elif row.distribution == "lognormal_odds_ratio":
            sigma = (np.log(float(row.high)) - np.log(float(row.low))) / (2 * 1.95996398454)
            x = rng.lognormal(np.log(float(row.base)), sigma, simulations)
        else:
            raise ValueError(f"Unsupported distribution: {row.distribution}")
        draws[row.parameter] = x
    return draws


def probabilistic(cohort, parameters, correlation_override=None):
    p = parameter_values(parameters)
    n = int(p["psa_simulations"])
    rng = np.random.default_rng(int(p["random_seed"]))
    d = _sample_parameters(rng, parameters, n)
    rho = p["cross_country_forecast_correlation"] if correlation_override is None else float(correlation_override)
    if not 0 <= rho <= 1:
        raise ValueError("Correlation must be between 0 and 1")
    countries = len(cohort)
    forecast_sd = np.maximum(
        (cohort.Upper80Pct.to_numpy() - cohort.Lower80Pct.to_numpy()) / (2 * 1.2815515655),
        0.01,
    )
    z = np.sqrt(rho) * rng.normal(size=(n, 1)) + np.sqrt(1 - rho) * rng.normal(size=(n, countries))
    sampled_forecast = np.clip(
        cohort.RecommendedForecastPct.to_numpy()[None, :] + z * forecast_sd[None, :], 0, 100
    )
    cv = cohort.IncidenceMeasurementCV.to_numpy()
    sigma = np.sqrt(np.log1p(cv**2))
    incidence_factor = rng.lognormal(-0.5 * sigma**2, sigma, size=(n, countries))
    economic = (
        cohort.EconomicCohortEpisodes.to_numpy()[None, :]
        * d["future_bsi_incidence_multiplier"][:, None]
        * incidence_factor
    )
    adequate = cohort.tested.to_numpy()[None, :] >= d["minimum_reported_tested"][:, None]
    persistence_trigger = adequate & (
        cohort.PersistenceForecastPct.to_numpy()[None, :] >= d["resistance_trigger_pct"][:, None]
    )
    forecast_trigger = adequate & (
        (cohort.RecommendedForecastPct.to_numpy()[None, :] >= d["resistance_trigger_pct"][:, None])
        | (cohort.ForecastIncreasePP.to_numpy()[None, :] >= d["increase_trigger_pp"][:, None])
    )
    resistant = economic * sampled_forecast / 100
    p0 = d["baseline_bsi_mortality_probability"][:, None]
    odds_ratio = d["intervention_mortality_odds_ratio"][:, None]
    p1 = odds_ratio * p0 / (1 - p0 + odds_ratio * p0)
    usual_deaths = resistant * p0
    persistence_deaths = np.where(persistence_trigger, resistant * p1, usual_deaths)
    forecast_deaths = np.where(forecast_trigger, resistant * p1, usual_deaths)
    fixed = d["programme_fixed_cost_per_country"][:, None]
    variable = d["programme_variable_cost_per_bsi_episode"][:, None]
    persistence_cost = np.where(persistence_trigger, fixed + variable * economic, 0).sum(axis=1)
    forecast_cost = np.where(forecast_trigger, fixed + variable * economic, 0).sum(axis=1)
    cost_usual = forecast_cost
    cost_persistence = forecast_cost - persistence_cost
    qaly_usual = (usual_deaths - forecast_deaths).sum(axis=1) * d["qaly_loss_per_bsi_death"]
    qaly_persistence = (persistence_deaths - forecast_deaths).sum(axis=1) * d["qaly_loss_per_bsi_death"]
    nmb_usual = d["willingness_to_pay_per_qaly"] * qaly_usual - cost_usual
    nmb_persistence = d["willingness_to_pay_per_qaly"] * qaly_persistence - cost_persistence
    return pd.DataFrame({
        "Simulation": np.arange(1, n + 1),
        "PersistenceTriggeredCountries": persistence_trigger.sum(axis=1),
        "ForecastTriggeredCountries": forecast_trigger.sum(axis=1),
        "DeathsAvoidedVsUsual": (usual_deaths - forecast_deaths).sum(axis=1),
        "IncrementalCostForecastVsUsualEUR": cost_usual,
        "QALYGainedForecastVsUsual": qaly_usual,
        "IncrementalCostForecastVsPersistenceEUR": cost_persistence,
        "QALYGainedForecastVsPersistence": qaly_persistence,
        "IncrementalNMBForecastVsUsualEUR": nmb_usual,
        "IncrementalNMBForecastVsPersistenceEUR": nmb_persistence,
        "CostEffectiveVsUsual": nmb_usual > 0,
        "CostEffectiveVsPersistence": nmb_persistence > 0,
    })


def psa_summary(psa):
    mapping = {
        "Deaths avoided vs usual care": "DeathsAvoidedVsUsual",
        "Incremental cost: forecast vs usual care (EUR)": "IncrementalCostForecastVsUsualEUR",
        "Incremental cost: forecast vs persistence (EUR)": "IncrementalCostForecastVsPersistenceEUR",
        "QALYs gained: forecast vs usual care": "QALYGainedForecastVsUsual",
        "QALYs gained: forecast vs persistence": "QALYGainedForecastVsPersistence",
        "Incremental NMB: forecast vs usual care (EUR)": "IncrementalNMBForecastVsUsualEUR",
        "Incremental NMB: forecast vs persistence (EUR)": "IncrementalNMBForecastVsPersistenceEUR",
    }
    rows = []
    for label, column in mapping.items():
        x = psa[column]
        rows.append({"Metric": label, "Mean": x.mean(), "Median": x.median(),
                     "Lower95": x.quantile(.025), "Upper95": x.quantile(.975),
                     "MonteCarloSE": np.nan, "MonteCarloLower95": np.nan,
                     "MonteCarloUpper95": np.nan})
    for label, column in [
        ("Probability cost-effective vs usual care", "CostEffectiveVsUsual"),
        ("Probability cost-effective vs persistence planning", "CostEffectiveVsPersistence"),
    ]:
        value = psa[column].mean(); se = np.sqrt(value * (1 - value) / len(psa))
        rows.append({"Metric": label, "Mean": value, "Median": np.nan,
                     "Lower95": np.nan, "Upper95": np.nan, "MonteCarloSE": se,
                     "MonteCarloLower95": max(0, value - 1.959964 * se),
                     "MonteCarloUpper95": min(1, value + 1.959964 * se)})
    return pd.DataFrame(rows)


def correlation_sensitivity(cohort, parameters):
    rows = []
    for rho in (0.0, 0.25, 0.50):
        x = probabilistic(cohort, parameters, rho)
        prob = x.CostEffectiveVsPersistence.mean(); se = np.sqrt(prob * (1 - prob) / len(x))
        rows.append({
            "CrossCountryCorrelation": rho, "Simulations": len(x),
            "MeanIncrementalNMBVsPersistenceEUR": x.IncrementalNMBForecastVsPersistenceEUR.mean(),
            "Lower95IncrementalNMBVsPersistenceEUR": x.IncrementalNMBForecastVsPersistenceEUR.quantile(.025),
            "Upper95IncrementalNMBVsPersistenceEUR": x.IncrementalNMBForecastVsPersistenceEUR.quantile(.975),
            "ProbabilityCostEffectiveVsPersistence": prob, "MonteCarloSE": se,
            "MonteCarloLower95": max(0, prob - 1.959964 * se),
            "MonteCarloUpper95": min(1, prob + 1.959964 * se),
        })
    return pd.DataFrame(rows)


def _scenario_cohort(cohort, base_parameters, scenario_parameters):
    out = cohort.copy()
    base = parameter_values(base_parameters); scenario = parameter_values(scenario_parameters)
    out["EconomicCohortEpisodes"] = (
        out.EconomicCohortEpisodes / base["future_bsi_incidence_multiplier"]
        * scenario["future_bsi_incidence_multiplier"]
    )
    adequate = out.tested.ge(scenario["minimum_reported_tested"])
    out["PersistenceProgrammeTriggered"] = adequate & out.PersistenceForecastPct.ge(
        scenario["resistance_trigger_pct"]
    )
    out["ForecastProgrammeTriggered"] = adequate & (
        out.RecommendedForecastPct.ge(scenario["resistance_trigger_pct"])
        | out.ForecastIncreasePP.ge(scenario["increase_trigger_pp"])
    )
    return out


def one_way_sensitivity(cohort, parameters):
    base_nmb = deterministic(cohort, parameters).IncrementalNMBForecastVsPersistenceEUR.sum()
    rows = []
    uncertain = parameters[(parameters.distribution.ne("fixed")) & (parameters.low.ne(parameters.high))]
    for parameter in uncertain.itertuples(index=False):
        outcomes = {}
        for bound in ("low", "high"):
            scenario = parameters.copy()
            scenario.loc[scenario.parameter.eq(parameter.parameter), "base"] = float(getattr(parameter, bound))
            scenario_cohort = _scenario_cohort(cohort, parameters, scenario)
            outcomes[bound] = deterministic(scenario_cohort, scenario).IncrementalNMBForecastVsPersistenceEUR.sum()
        bounds = [base_nmb, outcomes["low"], outcomes["high"]]
        rows.append({"Parameter": parameter.parameter, "BaseNMBEUR": base_nmb,
                     "LowNMBEUR": outcomes["low"], "HighNMBEUR": outcomes["high"],
                     "MinimumNMBEUR": min(bounds), "MaximumNMBEUR": max(bounds),
                     "NMBRangeEUR": max(bounds) - min(bounds)})
    return pd.DataFrame(rows)


def run_bia(country_results, config):
    p = pd.read_csv(BIA_PARAMETERS).set_index("parameter").base.astype(float)
    years = np.arange(int(config["bia_start_year"]), int(config["bia_end_year"]) + 1)
    uptake = np.linspace(p.uptake_year_1, p.uptake_year_5, len(years))
    rows = []
    for strategy, active in {
        "Persistence-guided planning": country_results.PersistenceProgrammeTriggered,
        "ML forecast-guided planning": country_results.ForecastProgrammeTriggered,
    }.items():
        for i, year in enumerate(years):
            episodes = country_results.EconomicCohortEpisodes * (1 + p.annual_bsi_incidence_growth) ** i
            eligible = episodes[active].sum(); treated = eligible * uptake[i]
            price = (1 + p.price_growth) ** i
            setup = active.sum() * p.one_time_setup_cost_per_country * price if i == 0 else 0
            operating = active.sum() * p.annual_operating_cost_per_country * price
            variable = treated * p.variable_cost_per_treated_episode * price
            rows.append({"Strategy": strategy, "Year": year, "ActivatedCountries": int(active.sum()),
                         "EligibleEpisodes": eligible, "Uptake": uptake[i], "TreatedEpisodes": treated,
                         "SetupCostEUR": setup, "OperatingCostEUR": operating,
                         "VariableCostEUR": variable, "AnnualBudgetImpactEUR": setup + operating + variable})
    out = pd.DataFrame(rows)
    comparator = out[out.Strategy.eq("Persistence-guided planning")][["Year", "AnnualBudgetImpactEUR"]].rename(
        columns={"AnnualBudgetImpactEUR": "PersistenceBudgetEUR"})
    out = out.merge(comparator, on="Year", validate="many_to_one")
    out["IncrementalBudgetVsPersistenceEUR"] = np.where(
        out.Strategy.eq("ML forecast-guided planning"),
        out.AnnualBudgetImpactEUR - out.PersistenceBudgetEUR, 0)
    return out.sort_values(["Strategy", "Year"])


def validate_historical_trigger():
    nested = pd.read_csv(ROOT / "outputs/research_grade/nested_backtest_predictions_h3.csv")
    data = nested[nested.antibiotic.eq("Third-generation cephalosporins")].copy()
    origin_tests = pd.read_csv(EARS)[["iso3", "year", "antibiotic", "tested"]].rename(
        columns={"tested": "origin_tested"})
    data = data.merge(origin_tests, on=["iso3", "year", "antibiotic"], validate="one_to_one")
    eligible = data.origin_tested.ge(100) & data.target_tested.ge(100)
    data = data[eligible].copy()
    data["ActualNeed"] = data.observed.ge(15) | (data.observed - data.resistance_pct).ge(2)
    data["MLTrigger"] = data.prediction.ge(15) | (data.prediction - data.resistance_pct).ge(2)
    data["PersistenceTrigger"] = data.persistence_prediction.ge(15)
    rows = []
    for year in [*sorted(data.target_year.unique()), "Pooled"]:
        frame = data if year == "Pooled" else data[data.target_year.eq(year)]
        for strategy, column in [("ML forecast", "MLTrigger"), ("Persistence", "PersistenceTrigger")]:
            actual = frame.ActualNeed.to_numpy(); pred = frame[column].to_numpy()
            tp = int((actual & pred).sum()); fp = int((~actual & pred).sum())
            fn = int((actual & ~pred).sum()); tn = int((~actual & ~pred).sum())
            rows.append({"TargetYear": year, "Strategy": strategy, "N": len(frame),
                         "TP": tp, "FP": fp, "FN": fn, "TN": tn,
                         "Sensitivity": tp / (tp + fn) if tp + fn else np.nan,
                         "Specificity": tn / (tn + fp) if tn + fp else np.nan,
                         "FalseNegativeRate": fn / (tp + fn) if tp + fn else np.nan,
                         "Accuracy": (tp + tn) / len(frame)})
    return data, pd.DataFrame(rows)


def create_figures(country, psa, correlation, bia, trigger_metrics, one_way, parameters):
    figdir = REFERENCE / "figures"; figdir.mkdir(parents=True, exist_ok=True)
    plt.style.use("seaborn-v0_8-whitegrid")
    ordered = country.sort_values("RecommendedForecastPct")
    fig, ax = plt.subplots(figsize=(10, 9)); y = np.arange(len(ordered))
    ax.errorbar(ordered.RecommendedForecastPct, y,
                xerr=np.vstack([ordered.RecommendedForecastPct-ordered.Lower80Pct,
                                ordered.Upper80Pct-ordered.RecommendedForecastPct]), fmt="none", color="grey")
    ax.scatter(ordered.RecommendedForecastPct, y,
               c=np.where(ordered.ForecastProgrammeTriggered, "#E76F51", "#264653"))
    ax.axvline(parameter_values(parameters)["resistance_trigger_pct"],
               ls="--", color="#E76F51"); ax.set_yticks(y, ordered.country)
    ax.set(xlabel="Forecast resistance (%) with 80% interval", title="2026 forecast and activation rule")
    fig.tight_layout(); fig.savefig(figdir / "01_forecast_trigger_profile.png", dpi=160); plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 6)); sample = psa.iloc[::max(1, len(psa)//8000)]
    ax.scatter(sample.QALYGainedForecastVsPersistence,
               sample.IncrementalCostForecastVsPersistenceEUR/1e6, s=7, alpha=.2)
    ax.axhline(0, color="grey"); ax.axvline(0, color="grey")
    ax.set(xlabel="QALYs gained: ML vs persistence", ylabel="Incremental cost (million EUR)",
           title="CEA probabilistic plane (Python validation)")
    fig.tight_layout(); fig.savefig(figdir / "02_cea_plane.png", dpi=160); plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5)); ax.errorbar(
        correlation.CrossCountryCorrelation, correlation.ProbabilityCostEffectiveVsPersistence,
        yerr=np.vstack([correlation.ProbabilityCostEffectiveVsPersistence-correlation.MonteCarloLower95,
                        correlation.MonteCarloUpper95-correlation.ProbabilityCostEffectiveVsPersistence]),
        marker="o", capsize=5)
    lower = max(0, correlation.MonteCarloLower95.min() - .01)
    upper = min(1, correlation.MonteCarloUpper95.max() + .01)
    ax.set(ylim=(lower, upper), xlabel="Cross-country forecast-error correlation (ρ)",
           ylabel="Probability cost-effective", title="Correlation sensitivity")
    fig.tight_layout(); fig.savefig(figdir / "03_correlation_sensitivity.png", dpi=160); plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 6))
    for name, frame in bia.groupby("Strategy"):
        ax.plot(frame.Year, frame.AnnualBudgetImpactEUR/1e6, marker="o", label=name)
    ax.set(xlabel="Year", ylabel="Annual budget impact (million EUR)",
           title="Separate five-year affordability analysis"); ax.legend()
    fig.tight_layout(); fig.savefig(figdir / "04_five_year_bia.png", dpi=160); plt.close(fig)

    pooled = trigger_metrics[trigger_metrics.TargetYear.astype(str).eq("Pooled")]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, strategy in zip(axes, ["ML forecast", "Persistence"], strict=True):
        row = pooled[pooled.Strategy.eq(strategy)].iloc[0]
        matrix = np.array([[row.TP, row.FN], [row.FP, row.TN]])
        ax.imshow(matrix, cmap="Blues")
        for (i,j), v in np.ndenumerate(matrix): ax.text(j, i, int(v), ha="center", va="center")
        ax.set_xticks([0,1],["Trigger","No trigger"]); ax.set_yticks([0,1],["Actual need","No need"])
        ax.set_title(f"{strategy}\nSensitivity {row.Sensitivity:.1%}")
    fig.tight_layout(); fig.savefig(figdir / "05_trigger_confusion_matrices.png", dpi=160); plt.close(fig)

    labels = {
        "resistance_trigger_pct": "Resistance trigger",
        "increase_trigger_pp": "Forecast-increase trigger",
        "future_bsi_incidence_multiplier": "Future BSI incidence",
        "baseline_bsi_mortality_probability": "Baseline BSI mortality",
        "intervention_mortality_odds_ratio": "RDT + ASP mortality odds ratio",
        "programme_fixed_cost_per_country": "Programme fixed cost",
        "programme_variable_cost_per_bsi_episode": "Variable cost per BSI episode",
        "qaly_loss_per_bsi_death": "QALY loss per BSI death",
        "willingness_to_pay_per_qaly": "Willingness to pay per QALY",
    }
    tornado = one_way.sort_values("NMBRangeEUR")
    fig, ax = plt.subplots(figsize=(11, 7)); y = np.arange(len(tornado))
    ax.hlines(y, tornado.MinimumNMBEUR/1e6, tornado.MaximumNMBEUR/1e6,
              linewidth=8, color="#457B9D")
    ax.scatter(tornado.BaseNMBEUR/1e6, y, marker="D", color="#E76F51", zorder=3)
    ax.axvline(0, color="black", ls="--")
    ax.set_yticks(y, [labels.get(x, x) for x in tornado.Parameter])
    ax.set(xlabel="Incremental NMB: ML vs persistence (million EUR)",
           title="One-way sensitivity analysis")
    fig.tight_layout(); fig.savefig(figdir / "06_one_way_sensitivity_tornado.png", dpi=160); plt.close(fig)


def md5(path):
    return hashlib.md5(path.read_bytes()).hexdigest()


def run():
    forecasts, ears, demographics, coverage, population, parameters, config = load_inputs()
    cohort = prepare_cohort(forecasts, ears, demographics, coverage, population, parameters, config)
    country = deterministic(cohort, parameters)
    psa = probabilistic(cohort, parameters)
    summary = psa_summary(psa)
    corr = correlation_sensitivity(cohort, parameters)
    one_way = one_way_sensitivity(cohort, parameters)
    bia = run_bia(country, config)
    trigger_rows, trigger_metrics = validate_historical_trigger()
    REFERENCE.mkdir(parents=True, exist_ok=True)
    cohort.to_csv(REFERENCE / "economic_cohort.csv", index=False)
    country.to_csv(REFERENCE / "deterministic_country_results.csv", index=False)
    deterministic_summary = pd.DataFrame({"Metric": [
        "Countries with valid coverage-adjusted denominators",
        "Countries triggered by persistence", "Countries triggered by ML forecast",
        "Projected total E. coli BSI episodes", "Forecast resistant episodes",
        "Incremental cost: ML vs persistence (EUR)",
        "QALYs gained: ML vs persistence", "Incremental NMB: ML vs persistence (EUR)"],
        "Value": [len(country), country.PersistenceProgrammeTriggered.sum(),
        country.ForecastProgrammeTriggered.sum(), country.EconomicCohortEpisodes.sum(),
        country.ForecastResistantEpisodes.sum(),
        country.IncrementalCostForecastVsPersistenceEUR.sum(),
        country.QALYGainedForecastVsPersistence.sum(),
        country.IncrementalNMBForecastVsPersistenceEUR.sum()]})
    deterministic_summary.to_csv(REFERENCE / "deterministic_summary.csv", index=False)
    psa.to_csv(REFERENCE / "psa_results.csv", index=False)
    summary.to_csv(REFERENCE / "psa_summary.csv", index=False)
    corr.to_csv(REFERENCE / "correlation_sensitivity.csv", index=False)
    one_way.to_csv(REFERENCE / "one_way_sensitivity.csv", index=False)
    bia.to_csv(REFERENCE / "annual_budget_impact.csv", index=False)
    bia.groupby("Strategy", as_index=False)[["AnnualBudgetImpactEUR", "IncrementalBudgetVsPersistenceEUR"]].sum().to_csv(
        REFERENCE / "five_year_budget_summary.csv", index=False)
    decision_dir = ROOT / "outputs/decision_validation"; decision_dir.mkdir(parents=True, exist_ok=True)
    trigger_rows.to_csv(decision_dir / "historical_trigger_predictions.csv", index=False)
    trigger_metrics.to_csv(decision_dir / "historical_trigger_metrics.csv", index=False)
    create_figures(country, psa, corr, bia, trigger_metrics, one_way, parameters)
    metadata = {"authoritative": False, "engine": "Python_formula_validation",
                "primary_engine": "R", "python_version": platform.python_version(),
                "simulations": len(psa), "seed": int(parameter_values(parameters)["random_seed"]),
                "hash_algorithm": "MD5", "input_hashes": {p.name: md5(p) for p in [FORECAST,EARS,COVERAGE,POPULATION,PARAMETERS]}}
    (REFERENCE / "generation_metadata.json").write_text(json.dumps(metadata, indent=2)+"\n", encoding="utf-8")
    print(summary.tail(2).to_string(index=False))
    print(corr.to_string(index=False))


if __name__ == "__main__":
    run()
