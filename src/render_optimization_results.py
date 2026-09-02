"""Render comparison figures and decision tables from optimization outputs."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "outputs" / "optimization"
FIGURES = OUTPUT / "figures"


def render():
    FIGURES.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")
    performance = pd.read_csv(OUTPUT / "optimized_performance_all_horizons.csv")
    logs = []
    importances = []
    predictions = []
    for horizon in (1, 3, 5):
        logs.append(pd.read_csv(OUTPUT / f"iterative_search_log_h{horizon}.csv").assign(Horizon=horizon))
        importances.append(pd.read_csv(OUTPUT / f"optimized_importance_h{horizon}.csv"))
        predictions.append(pd.read_csv(OUTPUT / f"optimized_predictions_h{horizon}.csv"))
    search_log = pd.concat(logs, ignore_index=True)
    importance = pd.concat(importances, ignore_index=True)
    prediction = pd.concat(predictions, ignore_index=True)
    search_log.to_csv(OUTPUT / "iterative_search_log_all_horizons.csv", index=False)

    accepted = search_log[search_log.Accepted.eq(True)].copy()
    accepted["AcceptedStep"] = accepted.groupby("Horizon").cumcount()
    fig, ax = plt.subplots(figsize=(9, 5.5))
    sns.lineplot(
        data=accepted, x="AcceptedStep", y="CV_RMSE", hue="Horizon",
        marker="o", palette="viridis", ax=ax,
    )
    ax.set_title("RMSE after each accepted search step")
    ax.set_xlabel("Accepted feature-search step")
    ax.set_ylabel("Rolling-validation RMSE")
    fig.tight_layout()
    fig.savefig(FIGURES / "04_accepted_search_trajectory.png", dpi=180)
    plt.close(fig)

    group_log = search_log[search_log.Stage.eq("Group forward selection")]
    group_heat = group_log.pivot_table(
        index="Candidate", columns="Horizon", values="ImprovementPct", aggfunc="max"
    )
    fig, ax = plt.subplots(figsize=(8, max(6, 0.5 * len(group_heat))))
    sns.heatmap(
        group_heat, cmap="RdYlGn", center=0, annot=True, fmt=".1f", ax=ax,
        cbar_kws={"label": "Best incremental CV RMSE improvement (%)"},
    )
    ax.set_title("Candidate feature-group value during forward search")
    fig.tight_layout()
    fig.savefig(FIGURES / "05_candidate_group_improvement_heatmap.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(17, 6), sharex=False)
    for ax, horizon in zip(axes, (1, 3, 5)):
        top = importance[importance.Horizon.eq(horizon)].nlargest(12, "ImportanceMean")
        sns.barplot(data=top, x="ImportanceMean", y="Feature", color="#2a7f9e", ax=ax)
        ax.axvline(0, color="black", linewidth=.8)
        ax.set_title(f"{horizon}-year model")
        ax.set_xlabel("Permutation RMSE importance")
        ax.set_ylabel("")
    fig.suptitle("Most influential retained features on the confirmation year", y=1.02)
    fig.tight_layout()
    fig.savefig(FIGURES / "06_optimized_feature_importance.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    for ax, horizon in zip(axes, (1, 3, 5)):
        frame = prediction[prediction.horizon.eq(horizon)]
        target = f"target_resistance_h{horizon}"
        sns.scatterplot(
            data=frame, x=target, y="prediction", hue="antibiotic",
            alpha=.75, s=36, legend=False, ax=ax,
        )
        ax.plot([0, 100], [0, 100], linestyle="--", color="black", linewidth=1)
        ax.set_xlim(0, 100); ax.set_ylim(0, 100)
        ax.set_title(f"{horizon}-year")
        ax.set_xlabel("Observed resistance (%)"); ax.set_ylabel("Predicted resistance (%)")
    fig.suptitle("Optimized model: observed versus predicted resistance", y=1.02)
    fig.tight_layout()
    fig.savefig(FIGURES / "07_optimized_observed_vs_predicted.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    residual_frames = []
    for horizon in (1, 3, 5):
        frame = prediction[prediction.horizon.eq(horizon)].copy()
        frame["Residual"] = frame[f"target_resistance_h{horizon}"] - frame["prediction"]
        residual_frames.append(frame[["horizon", "Residual"]])
    residual = pd.concat(residual_frames, ignore_index=True)
    fig, ax = plt.subplots(figsize=(9, 5.5))
    sns.kdeplot(data=residual, x="Residual", hue="horizon", fill=False, common_norm=False, ax=ax)
    ax.axvline(0, color="black", linestyle="--")
    ax.set_title("Confirmation-year residual distributions")
    ax.set_xlabel("Observed minus predicted resistance (percentage points)")
    fig.tight_layout()
    fig.savefig(FIGURES / "08_optimized_residual_distribution.png", dpi=180)
    plt.close(fig)

    comparison_path = OUTPUT / "original_vs_optimized_confirmation.csv"
    comparison = pd.read_csv(comparison_path)
    comparison_long = pd.concat([
        comparison[["Horizon", "OriginalRMSE"]].rename(columns={"OriginalRMSE": "RMSE"}).assign(ModelVersion="Previous model"),
        comparison[["Horizon", "OptimizedRMSE"]].rename(columns={"OptimizedRMSE": "RMSE"}).assign(ModelVersion="Optimized model"),
    ], ignore_index=True)
    fig, ax = plt.subplots(figsize=(9, 5.5))
    sns.barplot(data=comparison_long, x="Horizon", y="RMSE", hue="ModelVersion", ax=ax)
    ax.set_title("Previous versus optimized model on the latest confirmation year")
    ax.set_ylabel("RMSE (resistance percentage points)")
    fig.tight_layout()
    fig.savefig(FIGURES / "09_previous_vs_optimized_rmse.png", dpi=180)
    plt.close(fig)

    confirmation = performance[performance.Split.eq("Latest-year confirmation")]
    decisions = []
    for horizon in (1, 3, 5):
        frame = confirmation[confirmation.Horizon.eq(horizon)].set_index("Specification")
        optimized = frame.loc["Optimized features"]
        persistence = frame.loc["Persistence"]
        use_model = optimized.RMSE < persistence.RMSE
        original_row = comparison[comparison.Horizon.eq(horizon)].iloc[0]
        decisions.append({
            "Horizon": horizon,
            "RecommendedForecast": optimized.Model if use_model else "Persistence",
            "OptimizedRMSE": optimized.RMSE,
            "PersistenceRMSE": persistence.RMSE,
            "RMSEImprovementVsPersistencePct": 100 * (persistence.RMSE - optimized.RMSE) / persistence.RMSE,
            "PreviousModelRMSE": original_row.OriginalRMSE,
            "RMSEImprovementVsPreviousModelPct": 100 * (original_row.OriginalRMSE - optimized.RMSE) / original_row.OriginalRMSE,
            "Decision": "Use optimized model" if use_model else "Retain persistence baseline",
        })
    pd.DataFrame(decisions).to_csv(OUTPUT / "forecast_recommendations.csv", index=False)


if __name__ == "__main__":
    render()
