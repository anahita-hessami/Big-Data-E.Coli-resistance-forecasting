"""Run and visualise the bounded iterative AMR feature search."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.iterative_feature_search import engineer_enhanced_features, optimize_horizon
from src.render_optimization_results import render


ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
OUTPUT = ROOT / "outputs" / "optimization"
FIGURES = OUTPUT / "figures"


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    panel = pd.read_csv(PROCESSED / "eu_ecoli_bsi_forecasting_panel.csv")
    enhanced = engineer_enhanced_features(panel)
    enhanced.to_csv(PROCESSED / "eu_ecoli_bsi_enhanced_feature_panel.csv", index=False)

    results = {}
    for horizon in (1, 3, 5):
        print(f"\nOptimizing {horizon}-year horizon", flush=True)
        result = optimize_horizon(enhanced, horizon, OUTPUT)
        results[horizon] = result
        result.performance.to_csv(OUTPUT / f"optimized_performance_h{horizon}.csv", index=False)
        result.search_log.to_csv(OUTPUT / f"iterative_search_log_h{horizon}.csv", index=False)
        result.tuning_results.to_csv(OUTPUT / f"model_tuning_h{horizon}.csv", index=False)
        result.predictions.to_csv(OUTPUT / f"optimized_predictions_h{horizon}.csv", index=False)
        result.feature_importance.to_csv(OUTPUT / f"optimized_importance_h{horizon}.csv", index=False)
        print("Selected model:", result.selected_model_name, flush=True)
        print("Selected groups:", result.selected_groups, flush=True)
        print("Features:", result.numeric_features + result.categorical_features, flush=True)
        print(result.performance.to_string(index=False), flush=True)

    performance = pd.concat([result.performance for result in results.values()], ignore_index=True)
    performance.to_csv(OUTPUT / "optimized_performance_all_horizons.csv", index=False)
    feature_rows = []
    for horizon, result in results.items():
        for feature in result.numeric_features:
            feature_rows.append({"Horizon": horizon, "Feature": feature, "Type": "numeric"})
        for feature in result.categorical_features:
            feature_rows.append({"Horizon": horizon, "Feature": feature, "Type": "categorical"})
    selected_features = pd.DataFrame(feature_rows)
    selected_features.to_csv(OUTPUT / "optimized_selected_features.csv", index=False)

    sns.set_theme(style="whitegrid")
    cv = performance[performance.Split.eq("Rolling CV")].copy()
    fig, ax = plt.subplots(figsize=(10, 5.5))
    sns.barplot(data=cv, x="Horizon", y="RMSE", hue="Specification", ax=ax)
    ax.set_title("Rolling-origin validation: persistence, base and optimized features")
    ax.set_ylabel("RMSE (resistance percentage points)")
    fig.tight_layout()
    fig.savefig(FIGURES / "01_rolling_cv_model_comparison.png", dpi=180)
    plt.close(fig)

    confirmation = performance[performance.Split.eq("Latest-year confirmation")].copy()
    fig, ax = plt.subplots(figsize=(9, 5.5))
    sns.barplot(data=confirmation, x="Horizon", y="RMSE", hue="Specification", ax=ax)
    ax.set_title("Latest-year confirmation: optimized model versus persistence")
    ax.set_ylabel("RMSE (resistance percentage points)")
    fig.tight_layout()
    fig.savefig(FIGURES / "02_latest_year_confirmation.png", dpi=180)
    plt.close(fig)

    matrix = selected_features.assign(Selected=1).pivot_table(
        index="Feature", columns="Horizon", values="Selected", fill_value=0
    )
    order = matrix.sum(axis=1).sort_values(ascending=False).index
    fig, ax = plt.subplots(figsize=(7.5, max(7, 0.28 * len(matrix))))
    sns.heatmap(matrix.loc[order], cmap="Blues", cbar=False, linewidths=.4, ax=ax)
    ax.set_title("Features retained by the leakage-safe search")
    fig.tight_layout()
    fig.savefig(FIGURES / "03_selected_feature_heatmap.png", dpi=180)
    plt.close(fig)

    original_path = ROOT / "outputs" / "all_horizon_performance.csv"
    if original_path.exists():
        original = pd.read_csv(original_path)
        original = original[(original.Split == "Test") & (original.Model != "Persistence")]
        original = original[["Horizon", "Model", "RMSE", "MAE", "R2"]].rename(
            columns={"Model": "OriginalModel", "RMSE": "OriginalRMSE",
                     "MAE": "OriginalMAE", "R2": "OriginalR2"}
        )
        optimized = confirmation[confirmation.Specification.eq("Optimized features")][
            ["Horizon", "Model", "RMSE", "MAE", "R2"]
        ].rename(columns={"Model": "OptimizedModel", "RMSE": "OptimizedRMSE",
                          "MAE": "OptimizedMAE", "R2": "OptimizedR2"})
        comparison = original.merge(optimized, on="Horizon", how="outer")
        comparison["RMSEChangePct"] = 100 * (
            comparison.OptimizedRMSE - comparison.OriginalRMSE
        ) / comparison.OriginalRMSE
        comparison.to_csv(OUTPUT / "original_vs_optimized_confirmation.csv", index=False)
        print("\nOriginal versus optimized latest-year confirmation")
        print(comparison.to_string(index=False))

    render()


if __name__ == "__main__":
    main()
