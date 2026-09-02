from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.regional_amr import (  # noqa: E402
    breakpoint_audit,
    build_eu_panel,
    validate_consumption,
    validate_demographics,
    validate_ears,
)
from src.iterative_feature_search import engineer_enhanced_features  # noqa: E402


def synthetic_inputs():
    countries = [("AAA", "Alpha"), ("BBB", "Beta"), ("CCC", "Gamma")]
    antibiotics = [("Ciprofloxacin", "Fluoroquinolones"), ("Cefotaxime", "3GC")]
    years = range(2008, 2025)
    ears_rows, consumption_rows, demographic_rows = [], [], []
    for ci, (iso3, country) in enumerate(countries):
        for year in years:
            demographic_rows.append({
                "country": country, "iso3": iso3, "year": year,
                "population": 1_000_000 + ci * 100_000 + 5000 * (year - 2008),
                "age65_pct": 14 + ci + 0.1 * (year - 2008),
                "urban_pct": 70 + ci, "gdp_per_capita": 30_000 + ci * 2000,
                "health_expenditure_pct_gdp": 8 + ci * 0.2,
            })
            for sector in ["community", "hospital"]:
                for _, family in antibiotics:
                    consumption_rows.append({
                        "country": country, "iso3": iso3, "year": year,
                        "sector": sector, "antibiotic_family": family,
                        "ddd_per_1000_per_day": 1 + 0.02 * (year - 2008) + ci * 0.1,
                        "access_pct": 65 + ci, "broad_narrow_ratio": 1.2 + ci * 0.1,
                        "oral_parenteral_ratio": 5 + ci,
                    })
            for ai, (antibiotic, _) in enumerate(antibiotics):
                tested = 500 + 20 * ci + 5 * (year - 2008)
                pct = np.clip(10 + 5 * ai + 2 * ci + 0.45 * (year - 2008), 0, 100)
                ears_rows.append({
                    "region": "EU_EEA", "country": country, "iso3": iso3,
                    "year": year, "pathogen": "Escherichia coli", "specimen": "Blood",
                    "antibiotic": antibiotic, "resistant": round(tested * pct / 100),
                    "tested": tested, "ast_standard": "EUCAST", "ast_version": "15.0",
                })
    return (
        pd.DataFrame(ears_rows), pd.DataFrame(consumption_rows),
        pd.DataFrame(demographic_rows),
        pd.DataFrame(antibiotics, columns=["antibiotic", "antibiotic_family"]),
    )


def test_validation_and_panel():
    ears, consumption, demographics, mapping = synthetic_inputs()
    ears, audit = validate_ears(ears)
    consumption = validate_consumption(consumption)
    demographics = validate_demographics(demographics)
    panel = build_eu_panel(ears, consumption, demographics, mapping)
    assert len(audit) == 9
    assert panel["target_resistance_h1"].notna().any()
    assert panel["target_resistance_h3"].notna().any()
    assert panel["target_resistance_h5"].notna().any()
    assert panel["family_consumption_lag1"].notna().any()
    assert panel["resistance_trend4"].notna().any()
    enhanced = engineer_enhanced_features(panel)
    for feature in [
        "family_consumption_mean3", "ast_tested_per_100k",
        "eu_resistance_excl_country", "other_antibiotic_mean",
        "population_growth1", "trend_x_volatility",
    ]:
        assert feature in enhanced
        assert enhanced[feature].notna().any()


def test_breakpoint_registry():
    registry = pd.read_csv(ROOT / "data/templates/breakpoint_registry.csv")
    summary = breakpoint_audit(registry)
    assert set(summary["region"]) == {"EU_EEA", "USA", "Japan"}


if __name__ == "__main__":
    test_validation_and_panel()
    test_breakpoint_registry()
    print("All pipeline tests passed.")
