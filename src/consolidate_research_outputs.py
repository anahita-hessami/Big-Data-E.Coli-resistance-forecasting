"""Consolidate completed per-horizon checkpoints into final research outputs.

This is intentionally separate from the expensive model-fitting runner so a
partially resumed run cannot leave the combined tables and figures containing
only the final requested horizon.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.research_grade_validation import (
    prequential_conformal_intervals,
    two_way_cluster_bootstrap,
)
from src.run_research_grade_pipeline import (
    _plot_bootstrap,
    _plot_conformal,
    _plot_multiplicity,
    _plot_skill,
)


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "outputs/research_grade"
FIGURES = OUTPUT / "figures"
HORIZONS = (1, 3, 5)


def _read_horizon_files(stem: str) -> pd.DataFrame:
    frames = []
    for horizon in HORIZONS:
        path = OUTPUT / f"{stem}_h{horizon}.csv"
        if not path.exists():
            raise FileNotFoundError(f"Required checkpoint is missing: {path}")
        frame = pd.read_csv(path)
        if "Horizon" not in frame.columns:
            frame["Horizon"] = horizon
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)

    performance = _read_horizon_files("rolling_origin_performance")
    performance.to_csv(OUTPUT / "rolling_origin_performance_all_horizons.csv", index=False)
    pooled = performance.loc[performance["TargetYear"].eq("Pooled")].copy()
    pooled = pooled[[
        "Horizon", "Evaluation", "N", "ModelRMSE", "PersistenceRMSE",
        "RMSEDifference", "Skill", "ModelWeightedRMSE",
        "PersistenceWeightedRMSE", "WeightedSkill", "ModelWins",
    ]].sort_values(["Horizon", "Evaluation"])
    pooled.to_csv(OUTPUT / "research_grade_summary.csv", index=False)

    bootstrap_frames = []
    bootstrap_summaries = []
    conformal_frames = []
    conformal_summaries = []
    for horizon in HORIZONS:
        predictions = pd.read_csv(OUTPUT / f"frozen_backtest_predictions_h{horizon}.csv")
        bootstrap, bootstrap_summary = two_way_cluster_bootstrap(
            predictions, n_bootstrap=1000, random_state=42 + horizon
        )
        bootstrap["Horizon"] = horizon
        bootstrap_summary["Horizon"] = horizon
        bootstrap_frames.append(bootstrap)
        bootstrap_summaries.append(bootstrap_summary)

        intervals, interval_summary = prequential_conformal_intervals(predictions)
        intervals["Horizon"] = horizon
        interval_summary["Horizon"] = horizon
        conformal_frames.append(intervals)
        conformal_summaries.append(interval_summary)

    bootstrap_distribution = pd.concat(bootstrap_frames, ignore_index=True)
    bootstrap_summary = pd.concat(bootstrap_summaries, ignore_index=True)
    conformal_predictions = pd.concat(conformal_frames, ignore_index=True)
    conformal_summary = pd.concat(conformal_summaries, ignore_index=True)
    bootstrap_distribution.to_csv(OUTPUT / "cluster_bootstrap_distribution.csv", index=False)
    bootstrap_summary.to_csv(OUTPUT / "cluster_bootstrap_summary.csv", index=False)
    conformal_predictions.to_csv(OUTPUT / "conformal_predictions_all_horizons.csv", index=False)
    conformal_summary.to_csv(OUTPUT / "conformal_coverage_summary.csv", index=False)

    screen = _read_horizon_files("multiplicity_controlled_feature_screen")
    screen.to_csv(OUTPUT / "multiplicity_controlled_feature_screen.csv", index=False)
    _read_horizon_files("null_search_distribution").to_csv(
        OUTPUT / "null_search_distribution.csv", index=False
    )
    _read_horizon_files("partial_pooling_tuning").to_csv(
        OUTPUT / "partial_pooling_tuning.csv", index=False
    )
    _read_horizon_files("nested_selected_specifications").to_csv(
        OUTPUT / "nested_selected_specifications.csv", index=False
    )

    _plot_skill(performance, FIGURES)
    _plot_bootstrap(bootstrap_distribution, FIGURES)
    _plot_conformal(conformal_predictions, FIGURES)
    _plot_multiplicity(screen, FIGURES)

    print(pooled.to_string(index=False))


if __name__ == "__main__":
    main()
