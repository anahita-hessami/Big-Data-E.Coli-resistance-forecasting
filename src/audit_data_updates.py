"""Audit current source coverage and guard ingestion of newer ECDC exports."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "outputs" / "research_grade"

OFFICIAL_RELEASES = [
    {
        "Dataset": "EARS-Net AMR",
        "LatestPublicDataYear": 2024,
        "ReleaseDate": "2025-11-18",
        "OfficialURL": (
            "https://www.ecdc.europa.eu/en/publications-data/"
            "antimicrobial-resistance-eueea-ears-net-annual-epidemiological-report-2024"
        ),
        "PreferredMachineReadableSource": "ECDC Surveillance Atlas",
    },
    {
        "Dataset": "ESAC-Net consumption",
        "LatestPublicDataYear": 2024,
        "ReleaseDate": "2025-11-18",
        "OfficialURL": (
            "https://www.ecdc.europa.eu/en/publications-data/"
            "antimicrobial-consumption-eueea-esac-net-annual-epidemiological-report-2024"
        ),
        "PreferredMachineReadableSource": "Current ESAC-Net dashboard export",
    },
    {
        "Dataset": "World Bank demographics",
        "LatestPublicDataYear": 2024,
        "ReleaseDate": "continuously revised",
        "OfficialURL": "https://api.worldbank.org/v2/",
        "PreferredMachineReadableSource": "World Bank Indicators API",
    },
]


def local_coverage(root: Path = ROOT) -> dict[str, tuple[int, int]]:
    mapping = {
        "EARS-Net AMR": root / "data/input/ears_net_ecoli_blood.csv",
        "ESAC-Net consumption": root / "data/input/esac_net_consumption.csv",
        "World Bank demographics": root / "data/input/demographics.csv",
    }
    output = {}
    for dataset, path in mapping.items():
        frame = pd.read_csv(path)
        output[dataset] = (int(frame["year"].min()), int(frame["year"].max()))
    return output


def update_readiness(root: Path = ROOT) -> pd.DataFrame:
    coverage = local_coverage(root)
    rows = []
    for release in OFFICIAL_RELEASES:
        start, end = coverage[release["Dataset"]]
        latest = release["LatestPublicDataYear"]
        rows.append({
            **release,
            "LocalStartYear": start,
            "LocalEndYear": end,
            "YearsBehind": latest - end,
            "UpdateRequired": bool(end < latest),
            "IngestionRule": (
                "Rebuild the complete historical series from the current export; do not append "
                "new years from a different ATC/DDD vintage."
                if release["Dataset"] == "ESAC-Net consumption"
                else "Rebuild from the official source and rerun all schema/leakage tests."
            ),
            "AuditedOn": date.today().isoformat(),
        })
    return pd.DataFrame(rows)


def prescribing_availability(root: Path = ROOT) -> pd.DataFrame:
    data = pd.read_csv(root / "data/input/esac_net_consumption.csv")
    features = [
        "ddd_per_1000_per_day", "access_pct", "broad_narrow_ratio",
        "oral_parenteral_ratio", "reserve_pct",
    ]
    rows = []
    for feature in features:
        available = data[feature].notna() if feature in data else pd.Series(False, index=data.index)
        years = sorted(data.loc[available, "year"].unique())
        rows.append({
            "Feature": feature,
            "AvailableRows": int(available.sum()),
            "TotalRows": len(data),
            "AvailabilityPct": 100 * available.mean(),
            "FirstAvailableYear": int(min(years)) if years else np.nan,
            "LastAvailableYear": int(max(years)) if years else np.nan,
            "ModelEligibleNow": bool(available.sum() > 0),
            "Decision": (
                "Eligible subject to temporal validation"
                if available.sum() else "Do not impute or fabricate; obtain an official export"
            ),
        })
    return pd.DataFrame(rows)


def validate_current_esac_export(path: str | Path) -> pd.DataFrame:
    """Validate a normalized full-history ESAC-Net dashboard export.

    The function intentionally requires a complete historical extract. It
    rejects a file containing only new years because mixing report vintages can
    introduce artificial level changes after ATC/DDD revisions.
    """
    frame = pd.read_csv(path)
    required = {
        "country", "iso3", "year", "sector", "antibiotic_family",
        "ddd_per_1000_per_day", "source_url",
    }
    if missing := required.difference(frame.columns):
        raise ValueError(f"Current ESAC export is missing columns: {sorted(missing)}")
    if int(frame["year"].min()) > 2013:
        raise ValueError(
            "The current ESAC file does not contain the complete model-era history from 2013. "
            "Do not append it to the older workbook."
        )
    duplicates = frame.duplicated([
        "iso3", "year", "sector", "antibiotic_family"
    ])
    if duplicates.any():
        raise ValueError(f"Current ESAC export contains {int(duplicates.sum())} duplicate keys.")
    if not frame["ddd_per_1000_per_day"].dropna().ge(0).all():
        raise ValueError("Consumption contains negative values.")
    for optional in [
        "access_pct", "broad_narrow_ratio", "oral_parenteral_ratio", "reserve_pct"
    ]:
        if optional not in frame:
            frame[optional] = np.nan
    return frame


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    readiness = update_readiness()
    prescribing = prescribing_availability()
    readiness.to_csv(OUTPUT / "data_update_readiness.csv", index=False)
    prescribing.to_csv(OUTPUT / "prescribing_feature_availability.csv", index=False)
    print("Data update readiness")
    print(readiness.to_string(index=False))
    print("\nPrescribing feature availability")
    print(prescribing.to_string(index=False))


if __name__ == "__main__":
    main()
