"""Run the research-grade validation extension and save auditable outputs."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.research_grade_validation import (
    frozen_specification_backtest,
    load_frozen_specification,
    multiplicity_controlled_feature_screen,
    nested_pipeline_backtest,
    partial_pooling_backtest,
    prequential_conformal_intervals,
    summarize_backtest,
    two_way_cluster_bootstrap,
)


ROOT = Path(__file__).resolve().parents[1]


def _save(frame: pd.DataFrame, output: Path, name: str) -> None:
    frame.to_csv(output / name, index=False)


def _plot_skill(performance: pd.DataFrame, figure_dir: Path) -> None:
    frame = performance.loc[performance["TargetYear"].ne("Pooled")].copy()
    frame["TargetYear"] = pd.to_numeric(frame["TargetYear"])
    fig, ax = plt.subplots(figsize=(11, 6))
    for (horizon, evaluation), group in frame.groupby(["Horizon", "Evaluation"]):
        ax.plot(
            group["TargetYear"], 100 * group["Skill"], marker="o",
            label=f"H{horizon} — {evaluation}",
        )
    ax.axhline(0, color="black", linewidth=1, linestyle="--")
    ax.set(title="Rolling-origin forecast skill relative to persistence",
           xlabel="Target year", ylabel="RMSE skill (%)")
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(figure_dir / "01_rolling_origin_skill.png", dpi=180)
    plt.close(fig)


def _plot_bootstrap(distributions: pd.DataFrame, figure_dir: Path) -> None:
    horizons = sorted(distributions["Horizon"].unique())
    fig, axes = plt.subplots(1, len(horizons), figsize=(5 * len(horizons), 4), squeeze=False)
    for ax, horizon in zip(axes[0], horizons):
        values = 100 * distributions.loc[distributions["Horizon"].eq(horizon), "Skill"]
        ax.hist(values.dropna(), bins=35, color="#2878B5", alpha=.8)
        ax.axvline(0, color="black", linestyle="--")
        ax.axvline(values.median(), color="#C82423", linewidth=2)
        ax.set(title=f"H{horizon}", xlabel="Bootstrapped RMSE skill (%)", ylabel="Replicates")
    fig.suptitle("Country × target-year cluster bootstrap")
    fig.tight_layout()
    fig.savefig(figure_dir / "02_cluster_bootstrap_skill.png", dpi=180)
    plt.close(fig)


def _plot_conformal(predictions: pd.DataFrame, figure_dir: Path) -> None:
    latest_frames = []
    for horizon, group in predictions.groupby("Horizon"):
        latest = group.loc[group["target_year"].eq(group["target_year"].max())].copy()
        latest = latest.sort_values("observed").reset_index(drop=True)
        latest["Rank"] = np.arange(len(latest))
        latest_frames.append(latest)
    fig, axes = plt.subplots(len(latest_frames), 1, figsize=(12, 4 * len(latest_frames)), squeeze=False)
    for ax, frame in zip(axes[:, 0], latest_frames):
        valid = frame["lower_80"].notna()
        ax.vlines(
            frame.loc[valid, "Rank"], frame.loc[valid, "lower_80"],
            frame.loc[valid, "upper_80"], color="#9CC7E4", alpha=.65,
            label="80% prequential interval",
        )
        ax.scatter(frame["Rank"], frame["prediction"], s=12, color="#2878B5", label="Prediction")
        ax.scatter(frame["Rank"], frame["observed"], s=12, color="#C82423", label="Observed")
        ax.set(title=f"H{int(frame['Horizon'].iloc[0])}, target {int(frame['target_year'].iloc[0])}",
               xlabel="Country–antibiotic rows ordered by outcome", ylabel="Resistance (%)")
        ax.legend(fontsize=8, ncol=3)
    fig.suptitle("Prequential conformal intervals use only earlier-origin errors")
    fig.tight_layout()
    fig.savefig(figure_dir / "03_conformal_intervals.png", dpi=180)
    plt.close(fig)


def _plot_multiplicity(screen: pd.DataFrame, figure_dir: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    for ax, horizon in zip(axes, sorted(screen["Horizon"].unique())):
        group = screen.loc[screen["Horizon"].eq(horizon)].copy()
        color = np.where(group["RetainAfterMultiplicityControl"], "#2CA02C", "#7F7F7F")
        ax.scatter(
            group["ObservedImprovementPct"], group["StabilitySelectionProbability"],
            c=color, s=35,
        )
        ax.axvline(.5, color="black", linestyle="--", linewidth=1)
        ax.axhline(.7, color="black", linestyle="--", linewidth=1)
        for row in group.nlargest(3, "ObservedImprovementPct").itertuples():
            ax.annotate(row.Feature, (row.ObservedImprovementPct, row.StabilitySelectionProbability),
                        fontsize=7, xytext=(3, 3), textcoords="offset points")
        ax.set(title=f"H{horizon}", xlabel="Observed RMSE improvement (%)",
               ylabel="Bootstrap selection stability")
    fig.suptitle("Feature evidence after adaptive-search controls")
    fig.tight_layout()
    fig.savefig(figure_dir / "04_multiplicity_controlled_features.png", dpi=180)
    plt.close(fig)


def run(args: argparse.Namespace) -> None:
    output = ROOT / "outputs/research_grade"
    figures = output / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    panel = pd.read_csv(ROOT / "data/processed/eu_ecoli_bsi_enhanced_feature_panel.csv")

    performance_frames = []
    bootstrap_distributions = []
    bootstrap_summaries = []
    conformal_predictions = []
    conformal_summaries = []
    multiplicity_frames = []
    null_frames = []
    nested_specifications = []
    pooling_tuning = []

    horizons = [int(value) for value in args.horizons.split(",")]
    for horizon in horizons:
        minimum_training_years = 3 if horizon == 5 else 5
        nested_minimum_training_years = 5 if horizon == 5 else minimum_training_years
        model, numeric, categorical = load_frozen_specification(ROOT, horizon)
        frozen_path = output / f"frozen_backtest_predictions_h{horizon}.csv"
        if args.resume and frozen_path.exists():
            frozen_predictions = pd.read_csv(frozen_path)
            frozen_performance = summarize_backtest(frozen_predictions)
        else:
            frozen_predictions, frozen_performance = frozen_specification_backtest(
                panel, horizon, model, numeric, categorical,
                minimum_training_target_years=minimum_training_years,
                max_origins=args.max_frozen_origins,
            )
        _save(frozen_predictions, output, f"frozen_backtest_predictions_h{horizon}.csv")
        frozen_performance["Horizon"] = horizon
        frozen_performance["Evaluation"] = "Frozen specification"
        performance_frames.append(frozen_performance)

        bootstrap, bootstrap_summary = two_way_cluster_bootstrap(
            frozen_predictions, args.bootstrap_repetitions, args.random_state + horizon
        )
        bootstrap["Horizon"] = horizon
        bootstrap_summary["Horizon"] = horizon
        bootstrap_distributions.append(bootstrap)
        bootstrap_summaries.append(bootstrap_summary)

        interval_predictions, interval_summary = prequential_conformal_intervals(
            frozen_predictions
        )
        interval_predictions["Horizon"] = horizon
        interval_summary["Horizon"] = horizon
        conformal_predictions.append(interval_predictions)
        conformal_summaries.append(interval_summary)

        pooling_path = output / f"partial_pooling_predictions_h{horizon}.csv"
        pooling_tuning_path = output / f"partial_pooling_tuning_h{horizon}.csv"
        if args.resume and pooling_path.exists() and pooling_tuning_path.exists():
            pooling_predictions = pd.read_csv(pooling_path)
            pooling_performance = summarize_backtest(pooling_predictions)
            tuning = pd.read_csv(pooling_tuning_path)
        else:
            pooling_predictions, pooling_performance, tuning = partial_pooling_backtest(
                panel, horizon, model, numeric, categorical,
                minimum_training_target_years=minimum_training_years,
                max_origins=args.max_frozen_origins,
            )
        _save(pooling_predictions, output, f"partial_pooling_predictions_h{horizon}.csv")
        pooling_performance["Horizon"] = horizon
        pooling_performance["Evaluation"] = "Partial pooling"
        performance_frames.append(pooling_performance)
        if not tuning.empty:
            _save(tuning, output, f"partial_pooling_tuning_h{horizon}.csv")
            pooling_tuning.append(tuning)

        if not args.skip_nested:
            nested_path = output / f"nested_backtest_predictions_h{horizon}.csv"
            nested_specification_path = output / f"nested_selected_specifications_h{horizon}.csv"
            if args.resume and nested_path.exists() and nested_specification_path.exists():
                nested_predictions = pd.read_csv(nested_path)
                nested_performance = summarize_backtest(nested_predictions)
                specification = pd.read_csv(nested_specification_path)
            else:
                nested_predictions, nested_performance, specification = nested_pipeline_backtest(
                    panel, horizon, max_origins=args.nested_origins,
                    minimum_training_target_years=nested_minimum_training_years,
                    random_state=args.random_state + horizon,
                )
            _save(nested_predictions, output, f"nested_backtest_predictions_h{horizon}.csv")
            nested_performance["Horizon"] = horizon
            nested_performance["Evaluation"] = "Nested pipeline"
            performance_frames.append(nested_performance)
            if not specification.empty:
                _save(specification, output, f"nested_selected_specifications_h{horizon}.csv")
                nested_specifications.append(specification)

        if not args.skip_multiplicity:
            screen, null = multiplicity_controlled_feature_screen(
                panel, horizon, model,
                n_null_repetitions=args.null_repetitions,
                n_stability_bootstrap=args.stability_repetitions,
                random_state=args.random_state + 100 + horizon,
            )
            screen["Horizon"] = horizon
            null["Horizon"] = horizon
            multiplicity_frames.append(screen)
            null_frames.append(null)
            _save(screen, output, f"multiplicity_controlled_feature_screen_h{horizon}.csv")
            _save(null, output, f"null_search_distribution_h{horizon}.csv")

        horizon_performance = pd.concat([
            frame for frame in performance_frames if frame["Horizon"].eq(horizon).all()
        ], ignore_index=True)
        _save(horizon_performance, output, f"rolling_origin_performance_h{horizon}.csv")
        _save(bootstrap_summary, output, f"cluster_bootstrap_summary_h{horizon}.csv")
        _save(interval_summary, output, f"conformal_coverage_summary_h{horizon}.csv")

    performance = pd.concat(performance_frames, ignore_index=True)
    bootstrap_distribution = pd.concat(bootstrap_distributions, ignore_index=True)
    bootstrap_summary = pd.concat(bootstrap_summaries, ignore_index=True)
    conformal_prediction = pd.concat(conformal_predictions, ignore_index=True)
    conformal_summary = pd.concat(conformal_summaries, ignore_index=True)
    _save(performance, output, "rolling_origin_performance_all_horizons.csv")
    _save(bootstrap_distribution, output, "cluster_bootstrap_distribution.csv")
    _save(bootstrap_summary, output, "cluster_bootstrap_summary.csv")
    _save(conformal_prediction, output, "conformal_predictions_all_horizons.csv")
    _save(conformal_summary, output, "conformal_coverage_summary.csv")
    tuning_files = sorted(output.glob("partial_pooling_tuning_h*.csv"))
    if tuning_files:
        _save(pd.concat([pd.read_csv(path) for path in tuning_files], ignore_index=True),
              output, "partial_pooling_tuning.csv")
    specification_files = sorted(output.glob("nested_selected_specifications_h*.csv"))
    if specification_files:
        _save(pd.concat([pd.read_csv(path) for path in specification_files], ignore_index=True),
              output, "nested_selected_specifications.csv")
    if multiplicity_frames:
        screen = pd.concat(multiplicity_frames, ignore_index=True)
        _save(screen, output, "multiplicity_controlled_feature_screen.csv")
        _save(pd.concat(null_frames, ignore_index=True), output, "null_search_distribution.csv")
        _plot_multiplicity(screen, figures)

    _plot_skill(performance, figures)
    _plot_bootstrap(bootstrap_distribution, figures)
    _plot_conformal(conformal_prediction, figures)

    pooled = performance.loc[performance["TargetYear"].eq("Pooled")].copy()
    pooled = pooled[[
        "Horizon", "Evaluation", "N", "ModelRMSE", "PersistenceRMSE",
        "RMSEDifference", "Skill", "ModelWeightedRMSE",
        "PersistenceWeightedRMSE", "WeightedSkill", "ModelWins",
    ]].sort_values(["Horizon", "Evaluation"])
    _save(pooled, output, "research_grade_summary.csv")
    print(pooled.to_string(index=False))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-frozen-origins", type=int, default=8)
    parser.add_argument("--nested-origins", type=int, default=3)
    parser.add_argument("--bootstrap-repetitions", type=int, default=2000)
    parser.add_argument("--null-repetitions", type=int, default=50)
    parser.add_argument("--stability-repetitions", type=int, default=1000)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--horizons", default="1,3,5")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--skip-nested", action="store_true")
    parser.add_argument("--skip-multiplicity", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
