"""Fail when economic documentation or authoritative R outputs are stale."""
from __future__ import annotations
import argparse
import hashlib
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
R_OUTPUT = ROOT / "outputs/economic_r"
INPUTS = [
    ROOT / "economic_model/parameters/economic_parameters.csv",
    ROOT / "outputs/notebook_ml/future_forecasts.csv",
    ROOT / "data/input/ears_net_ecoli_blood.csv",
    ROOT / "data/input/ecdc_earsnet_coverage_2023.csv",
    ROOT / "data/input/eurostat_population_projection_2026.csv",
]


def check(require_r=False):
    failures = []
    banned = ("38.4%", "37.3%", "14.1%", "14.08%", "5,000-draw")
    documents = [ROOT / "README.md", *ROOT.glob("docs/*.md"), *ROOT.glob("economic_model/*.md")]
    for path in documents:
        text = path.read_text(encoding="utf-8")
        for value in banned:
            if value in text:
                failures.append(f"stale headline {value} in {path.relative_to(ROOT)}")
    metadata_path = R_OUTPUT / "analysis_metadata.csv"
    if not metadata_path.exists():
        if require_r:
            failures.append("authoritative R output is missing")
    else:
        metadata = pd.read_csv(metadata_path, dtype=str).set_index("Field").Value
        if metadata.get("authoritative") != "true" or metadata.get("engine") != "R_authoritative":
            failures.append("output is not marked as authoritative R")
        if metadata.get("psa_simulations") != "50000":
            failures.append("authoritative PSA is not 50,000 draws")
        psa = pd.read_csv(R_OUTPUT / "psa_results.csv")
        if len(psa) != 50_000:
            failures.append(f"PSA has {len(psa)} rows")
        summary = pd.read_csv(R_OUTPUT / "psa_summary.csv")
        probability = summary.Metric.str.startswith("Probability")
        mc = ["MonteCarloSE", "MonteCarloLower95", "MonteCarloUpper95"]
        if not set(mc).issubset(summary.columns) or summary.loc[probability, mc].isna().any().any():
            failures.append("Monte Carlo uncertainty is missing")
        for path in INPUTS:
            digest = hashlib.md5(path.read_bytes()).hexdigest()
            if metadata.get("md5_" + path.name) != digest:
                failures.append(f"input hash mismatch: {path.name}")
    if failures:
        raise SystemExit("Economic consistency check failed:\n- " + "\n- ".join(failures))
    print("Economic consistency check passed." if metadata_path.exists() else
          "Documentation check passed; CI must generate authoritative R outputs.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("--require-r", action="store_true")
    check(parser.parse_args().require_r)
