import numpy as np
from src.economic_extension import (
    correlation_sensitivity, deterministic, load_inputs, parameter_values,
    prepare_cohort, probabilistic, run_bia, validate_historical_trigger,
)


def _cohort_and_inputs(draws=100):
    forecasts, ears, demographics, coverage, population, parameters, config = load_inputs()
    parameters = parameters.copy()
    parameters.loc[parameters.parameter.eq("psa_simulations"), ["base", "low", "high"]] = draws
    cohort = prepare_cohort(forecasts, ears, demographics, coverage, population, parameters, config)
    return cohort, parameters, config


def test_coverage_adjusted_cohort_and_deterministic_equations():
    cohort, parameters, _ = _cohort_and_inputs()
    result = deterministic(cohort, parameters)
    assert len(cohort) == 28
    assert cohort.ProjectedPopulationTargetYear.gt(0).all()
    assert cohort.population_coverage_pct.between(1, 100).all()
    assert cohort.IncidenceMeasurementCV.ge(.10).all()
    assert np.allclose(
        cohort.CoverageAdjustedEcoliBSIEpisodesOrigin,
        cohort.ReportedEcoliBSIEpisodesOrigin / (cohort.population_coverage_pct / 100),
    )
    assert np.allclose(
        result.IncrementalCostForecastVsPersistenceEUR,
        result.ForecastStrategyCostEUR - result.PersistenceStrategyCostEUR,
    )


def test_psa_correlation_and_bia_separation():
    cohort, parameters, config = _cohort_and_inputs(80)
    psa = probabilistic(cohort, parameters)
    assert len(psa) == 80 and psa.IncrementalNMBForecastVsPersistenceEUR.notna().all()
    corr = correlation_sensitivity(cohort, parameters)
    assert set(corr.CrossCountryCorrelation) == {0, .25, .5}
    bia = run_bia(deterministic(cohort, parameters), config)
    assert len(bia) == 10 and set(bia.Year) == set(range(2026, 2031))
    assert not any(token in column for column in bia for token in ("QALY", "ICER", "NMB"))


def test_intervention_effect_and_historical_trigger_metrics():
    _, parameters, _ = _cohort_and_inputs()
    p = parameters.set_index("parameter").loc["intervention_mortality_odds_ratio"]
    assert (float(p.base), float(p.low), float(p.high)) == (.78, .63, .96)
    _, metrics = validate_historical_trigger()
    pooled = metrics[metrics.TargetYear.astype(str).eq("Pooled")]
    assert set(pooled.Strategy) == {"ML forecast", "Persistence"}
    assert pooled.Sensitivity.between(0, 1).all()
