from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.research_grade_validation import (  # noqa: E402
    PenalizedCountryInterceptRegressor,
    prequential_conformal_intervals,
    summarize_backtest,
    two_way_cluster_bootstrap,
)


def test_cluster_bootstrap_and_skill_summary():
    rows = []
    for year in [2020, 2021, 2022]:
        for country in ["AAA", "BBB", "CCC"]:
            for antibiotic in ["A", "B"]:
                observed = 20 + year - 2020 + (country == "CCC")
                rows.append({
                    "iso3": country,
                    "target_year": year,
                    "antibiotic": antibiotic,
                    "observed": observed,
                    "prediction": observed + .5,
                    "persistence_prediction": observed + 1.0,
                    "model_error": -.5,
                    "persistence_error": -1.0,
                    "target_tested": 100,
                })
    predictions = pd.DataFrame(rows)
    summary = summarize_backtest(predictions)
    pooled = summary.loc[summary["TargetYear"].eq("Pooled")].iloc[0]
    assert pooled["Skill"] > 0
    distribution, interval = two_way_cluster_bootstrap(
        predictions, n_bootstrap=100, random_state=7
    )
    assert len(distribution) == 100
    assert interval.loc[interval["Metric"].eq("Skill"), "Lower95"].iloc[0] > 0


def test_prequential_intervals_never_use_same_or_future_year():
    frame = pd.DataFrame({
        "iso3": ["AAA"] * 8,
        "antibiotic": ["A"] * 8,
        "target_year": [2019, 2019, 2020, 2020, 2021, 2021, 2022, 2022],
        "observed": [10, 12, 11, 13, 12, 14, 13, 15],
        "prediction": [9, 11, 10, 12, 11, 13, 12, 14],
        "model_error": [1] * 8,
    })
    intervals, summary = prequential_conformal_intervals(
        frame, minimum_calibration_rows=2, antibiotic_specific_minimum=2,
    )
    assert intervals.loc[intervals["target_year"].eq(2019), "lower_80"].isna().all()
    assert intervals.loc[intervals["target_year"].eq(2020), "lower_80"].notna().all()
    assert summary.loc[summary["NominalCoverage"].eq(.8), "EmpiricalCoverage"].iloc[0] == 1


def test_penalized_country_intercepts_shrink_small_groups_and_handle_unseen():
    X = pd.DataFrame({
        "x": [0.0] * 12,
        "country": ["Large"] * 10 + ["Small"] * 2,
    })
    y = pd.Series([10.0] * 10 + [20.0] * 2)
    model = PenalizedCountryInterceptRegressor(Ridge(alpha=1.0), shrinkage=5.0)
    model.fit(X, y)
    large_effect = abs(model.group_effects_["Large"])
    small_effect = abs(model.group_effects_["Small"])
    assert small_effect < 10.0
    assert np.isfinite(large_effect)
    unseen = model.predict(pd.DataFrame({"x": [0.0], "country": ["New"]}))
    base = model.base_estimator_.predict(pd.DataFrame({"x": [0.0]}))
    assert np.allclose(unseen, base)


if __name__ == "__main__":
    test_cluster_bootstrap_and_skill_summary()
    test_prequential_intervals_never_use_same_or_future_year()
    test_penalized_country_intercepts_shrink_small_groups_and_handle_unseen()
    print("All research-grade validation tests passed.")
