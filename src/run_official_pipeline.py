"""Run the complete official-data EU/EEA E. coli AMR analysis outside Jupyter."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.regional_amr import (
    CONSUMPTION_REQUIRED,
    DEMOGRAPHIC_REQUIRED,
    EARS_REQUIRED,
    MAPPING_REQUIRED,
    build_eu_panel,
    evaluate_feature_groups,
    evaluate_individual_feature_additions,
    fit_horizon,
    read_csv,
    validate_consumption,
    validate_demographics,
    validate_ears,
)
from src.visualizations import (
    create_interactive_maps,
    plot_data_overview,
    plot_development_workflow,
    plot_feature_ablation,
    plot_feature_exploration,
    plot_importance_stability,
    plot_individual_feature_ablation,
    plot_model_results,
    plot_predictor_overview,
    write_visualisation_index,
)


ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "data" / "input"
PROCESSED = ROOT / "data" / "processed"
OUTPUT = ROOT / "outputs"


def main():
    PROCESSED.mkdir(parents=True, exist_ok=True)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    ears_raw = read_csv(INPUT / "ears_net_ecoli_blood.csv", EARS_REQUIRED, "EARS-Net")
    consumption_raw = read_csv(
        INPUT / "esac_net_consumption.csv", CONSUMPTION_REQUIRED, "ESAC-Net"
    )
    demographics_raw = read_csv(
        INPUT / "demographics.csv", DEMOGRAPHIC_REQUIRED, "demographics"
    )
    mapping = read_csv(
        INPUT / "antibiotic_mapping.csv", MAPPING_REQUIRED, "antibiotic mapping"
    )
    ears, ears_audit = validate_ears(ears_raw)
    consumption = validate_consumption(consumption_raw)
    demographics = validate_demographics(demographics_raw)
    ears_audit.to_csv(OUTPUT / "ears_input_audit.csv", index=False)

    panel_full = build_eu_panel(ears, consumption, demographics, mapping, horizons=(1, 3, 5))
    model_start_year = int(consumption.year.min())
    panel = panel_full[panel_full.year.ge(model_start_year)].copy()
    panel_full.to_csv(PROCESSED / "eu_ecoli_bsi_full_historical_panel.csv", index=False)
    panel.to_csv(PROCESSED / "eu_ecoli_bsi_forecasting_panel.csv", index=False)
    panel.isna().mean().mul(100).rename("missing_pct").to_csv(
        OUTPUT / "model_panel_missingness.csv"
    )

    visualisations = []
    visualisations.extend(plot_development_workflow(OUTPUT))
    visualisations.extend(plot_data_overview(ears, OUTPUT))
    visualisations.extend(plot_predictor_overview(panel, consumption, OUTPUT))
    for horizon in (1, 3, 5):
        paths, evidence = plot_feature_exploration(panel, OUTPUT, horizon)
        visualisations.extend(paths)
        evidence.to_csv(OUTPUT / f"feature_evidence_h{horizon}.csv", index=False)

    results = {}
    for horizon in (1, 3, 5):
        try:
            result = fit_horizon(panel, horizon, OUTPUT)
        except ValueError as error:
            print(f"Horizon {horizon} skipped: {error}")
            continue
        results[horizon] = result
        result.performance.assign(Horizon=horizon).to_csv(
            OUTPUT / f"performance_h{horizon}.csv", index=False
        )
        result.predictions.to_csv(OUTPUT / f"test_predictions_h{horizon}.csv", index=False)
        result.feature_importance.to_csv(
            OUTPUT / f"permutation_importance_h{horizon}.csv", index=False
        )
        print(f"Horizon {horizon}: selected {result.selected_model_name}")

    if not results:
        raise RuntimeError("No forecast horizon could be fitted.")
    performance = pd.concat(
        [result.performance.assign(Horizon=horizon) for horizon, result in results.items()],
        ignore_index=True,
    )
    performance.to_csv(OUTPUT / "all_horizon_performance.csv", index=False)

    grouped_tables = []
    individual_tables = []
    for horizon, result in results.items():
        grouped = evaluate_feature_groups(panel, horizon, result)
        individual = evaluate_individual_feature_additions(panel, horizon, result)
        grouped.to_csv(OUTPUT / f"feature_group_ablation_h{horizon}.csv", index=False)
        individual.to_csv(OUTPUT / f"individual_feature_ablation_h{horizon}.csv", index=False)
        grouped_tables.append(grouped)
        individual_tables.append(individual)
    grouped_all = pd.concat(grouped_tables, ignore_index=True)
    individual_all = pd.concat(individual_tables, ignore_index=True)
    grouped_all.to_csv(OUTPUT / "feature_group_ablation_all_horizons.csv", index=False)
    individual_all.to_csv(
        OUTPUT / "individual_feature_ablation_all_horizons.csv", index=False
    )

    visualisations.extend(plot_model_results(results, OUTPUT))
    visualisations.extend(plot_feature_ablation(grouped_all, OUTPUT))
    visualisations.extend(plot_individual_feature_ablation(individual_all, OUTPUT))
    visualisations.extend(plot_importance_stability(results, OUTPUT))
    generated_maps = create_interactive_maps(results, OUTPUT)
    if not generated_maps:
        generated_maps = sorted((OUTPUT / "figures").glob("21_map*.html"))
    visualisations.extend(generated_maps)
    write_visualisation_index(visualisations, OUTPUT)

    test_summary = performance[performance.Split.eq("Test")].sort_values(
        ["Horizon", "RMSE"]
    )
    validation_group = grouped_all[
        grouped_all.Split.eq("Validation")
        & ~grouped_all.FeatureSet.isin(["Base resistance history", "Persistence"])
    ].sort_values(["Horizon", "RMSE"])
    print("\nUntouched test results")
    print(test_summary[["Horizon", "Model", "RMSE", "MAE", "R2", "WeightedRMSE"]].to_string(index=False))
    print("\nValidation feature-group results")
    print(validation_group[[
        "Horizon", "FeatureSet", "RMSE", "RelativeRMSEChange_vs_Base_pct",
        "EvidenceLabel",
    ]].to_string(index=False))
    print(f"\nCreated {len(set(map(str, visualisations)))} visualisation files.")


if __name__ == "__main__":
    main()
