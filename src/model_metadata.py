"""Create self-documenting metadata sidecars for saved AMR models."""

from __future__ import annotations

from datetime import datetime, timezone
from hashlib import sha256
from importlib.metadata import PackageNotFoundError, version
import json
from pathlib import Path
import subprocess

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
PACKAGES = [
    "numpy", "pandas", "scipy", "scikit-learn", "joblib", "openpyxl",
    "matplotlib", "seaborn", "plotly",
]


def file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def code_tree_sha256(root: Path = ROOT) -> str:
    digest = sha256()
    code_files = sorted([
        *root.glob("src/*.py"), *root.glob("tests/*.py"),
        *root.glob("build*.py"),
    ])
    for path in code_files:
        digest.update(str(path.relative_to(root)).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def git_commit(root: Path = ROOT) -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=root, check=True,
            capture_output=True, text=True,
        )
        return result.stdout.strip()
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None


def package_versions() -> dict[str, str | None]:
    output = {}
    for package in PACKAGES:
        try:
            output[package] = version(package)
        except PackageNotFoundError:
            output[package] = None
    return output


def input_manifest(root: Path = ROOT) -> list[dict[str, object]]:
    rows = []
    for path in sorted((root / "data/input").glob("*.csv")):
        frame = pd.read_csv(path)
        rows.append({
            "path": str(path.relative_to(root)),
            "sha256": file_sha256(path),
            "rows": len(frame),
            "columns": list(frame.columns),
        })
    return rows


def write_optimized_model_sidecars(root: Path = ROOT) -> list[Path]:
    feature_table = pd.read_csv(
        root / "outputs/optimization/optimized_selected_features.csv"
    )
    performance = pd.read_csv(
        root / "outputs/optimization/optimized_performance_all_horizons.csv"
    )
    sources = root / "data/SOURCE_MANIFEST.csv"
    source_hash = file_sha256(sources) if sources.exists() else None
    inputs = input_manifest(root)
    code_hash = code_tree_sha256(root)
    commit = git_commit(root)
    environment = package_versions()
    created = []
    for model_path in sorted((root / "outputs/optimization").glob(
        "optimized_eu_ecoli_bsi_h*_*.joblib"
    )):
        horizon = int(model_path.name.split("_h", 1)[1].split("_", 1)[0])
        features = feature_table.loc[feature_table["Horizon"].eq(horizon)]
        confirmation = performance.loc[
            performance["Horizon"].eq(horizon)
            & performance["Split"].eq("Latest-year confirmation")
        ].to_dict("records")
        metadata = {
            "schema_version": "1.0",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "intended_use": (
                "Country-level EU/EEA bloodstream/invasive E. coli AMR research "
                "forecasting; not patient-level treatment guidance."
            ),
            "horizon_years": horizon,
            "model_file": str(model_path.relative_to(root)),
            "model_sha256": file_sha256(model_path),
            "numeric_features": features.loc[
                features["Type"].eq("numeric"), "Feature"
            ].tolist(),
            "categorical_features": features.loc[
                features["Type"].eq("categorical"), "Feature"
            ].tolist(),
            "target": f"target_resistance_h{horizon}",
            "outcome_unit": "percentage resistant among tested isolates",
            "specimen_scope": "bloodstream/invasive isolates",
            "breakpoint_policy": (
                "EU/EEA native EARS-Net aggregates; EUCAST-required from 2020; "
                "not pooled directly with CLSI/FDA aggregates."
            ),
            "confirmation_performance": confirmation,
            "source_manifest_sha256": source_hash,
            "input_files": inputs,
            "code_tree_sha256": code_hash,
            "git_commit": commit,
            "environment": environment,
            "random_seed": 42,
            "limitations": [
                "The latest historical period is confirmation, not external prospective validation.",
                "Country-level ecological predictions cannot guide individual treatment.",
                "Performance must be compared with persistence at every horizon.",
            ],
        }
        sidecar = model_path.with_suffix(".metadata.json")
        sidecar.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
        created.append(sidecar)
    return created


if __name__ == "__main__":
    for created_path in write_optimized_model_sidecars():
        print(created_path.relative_to(ROOT))
