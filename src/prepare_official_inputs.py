"""Download and harmonise public EU/EEA inputs for the E. coli AMR model.

The script uses only official public sources:

* ECDC Surveillance Atlas REST service for EARS-Net resistance counts;
* ECDC ESAC-Net 2022 downloadable workbook for 2013-2022 consumption;
* World Bank Indicators API for country-year demographic covariates.

The ECDC Atlas exposes antibiotic-group outcomes rather than individual agents.
Pre-2020 Atlas aggregates are not relabelled as EUCAST because the exact national
breakpoint standard/version is not exposed by the aggregate download.
"""

from __future__ import annotations

from datetime import date
import json
from pathlib import Path
import re
import urllib.parse
import urllib.request

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
INPUT_DIR = ROOT / "data" / "input"

EARS_ENDPOINT = (
    "https://atlas.ecdc.europa.eu/public/AtlasService/rest/post/"
    "GetMeasureResultsForTimePeriodAndGeoLevel"
)
EARS_SOURCE = "https://atlas.ecdc.europa.eu/public/"
ESAC_FILE_URL = (
    "https://www.ecdc.europa.eu/sites/default/files/documents/"
    "antimicrobial-resistance-ESAC-Net-report-downloadable-tables-2022.xlsx"
)
ESAC_SOURCE = (
    "https://www.ecdc.europa.eu/en/publications-data/"
    "downloadable-tables-antimicrobial-consumption-annual-epidemiological-report-2022"
)
WORLD_BANK_SOURCE = "https://api.worldbank.org/v2/"
EARS_END_YEAR_EXCLUSIVE = "2025"  # includes publicly released 2024 data


COUNTRIES = {
    "AT": ("AUT", "Austria"),
    "BE": ("BEL", "Belgium"),
    "BG": ("BGR", "Bulgaria"),
    "HR": ("HRV", "Croatia"),
    "CY": ("CYP", "Cyprus"),
    "CZ": ("CZE", "Czechia"),
    "DK": ("DNK", "Denmark"),
    "EE": ("EST", "Estonia"),
    "FI": ("FIN", "Finland"),
    "FR": ("FRA", "France"),
    "DE": ("DEU", "Germany"),
    "EL": ("GRC", "Greece"),
    "GR": ("GRC", "Greece"),
    "HU": ("HUN", "Hungary"),
    "IS": ("ISL", "Iceland"),
    "IE": ("IRL", "Ireland"),
    "IT": ("ITA", "Italy"),
    "LV": ("LVA", "Latvia"),
    "LI": ("LIE", "Liechtenstein"),
    "LT": ("LTU", "Lithuania"),
    "LU": ("LUX", "Luxembourg"),
    "MT": ("MLT", "Malta"),
    "NL": ("NLD", "Netherlands"),
    "NO": ("NOR", "Norway"),
    "PL": ("POL", "Poland"),
    "PT": ("PRT", "Portugal"),
    "RO": ("ROU", "Romania"),
    "SK": ("SVK", "Slovakia"),
    "SI": ("SVN", "Slovenia"),
    "ES": ("ESP", "Spain"),
    "SE": ("SWE", "Sweden"),
}
ISO3_TO_COUNTRY = {iso3: country for iso3, country in COUNTRIES.values()}
COUNTRY_TO_ISO3 = {country: iso3 for iso3, country in COUNTRIES.values()}


EARS_MEASURES = {
    "Aminopenicillins": {"tested": 893853, "resistant": 893854},
    "Fluoroquinolones": {"tested": 893859, "resistant": 893860},
    "Third-generation cephalosporins": {"tested": 893866, "resistant": 893867},
    "Aminoglycosides": {"tested": 893873, "resistant": 893874},
    "Carbapenems": {"tested": 893880, "resistant": 893881},
}


ESAC_SHEETS = {
    ("community", "J01C_penicillins"): "D2_J01C_AC",
    ("community", "J01D_other_beta_lactams"): "D3_J01D_AC",
    ("community", "J01M_quinolones"): "D6_J01M_AC",
    ("hospital", "J01C_penicillins"): "D9_J01C_HC",
    ("hospital", "J01D_other_beta_lactams"): "D10_J01D_HC",
    ("hospital", "J01DH_carbapenems"): "D11_J01DH_HC",
    ("hospital", "J01M_quinolones"): "D14_J01M_HC",
}


ANTIBIOTIC_MAPPING = {
    "Aminopenicillins": "J01C_penicillins",
    "Fluoroquinolones": "J01M_quinolones",
    "Third-generation cephalosporins": "J01D_other_beta_lactams",
    "Aminoglycosides": "J01G_aminoglycosides",
    "Carbapenems": "J01DH_carbapenems",
}


WORLD_BANK_INDICATORS = {
    "population": "SP.POP.TOTL",
    "age65_pct": "SP.POP.65UP.TO.ZS",
    "urban_pct": "SP.URB.TOTL.IN.ZS",
    "gdp_per_capita": "NY.GDP.PCAP.CD",
    "health_expenditure_pct_gdp": "SH.XPD.CHEX.GD.ZS",
}


def _json_request(url: str, payload: dict | None = None, timeout: int = 180):
    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    headers = {"Content-Type": "application/json"} if payload is not None else {}
    request = urllib.request.Request(url, data=data, headers=headers)
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.load(response)


def prepare_ears() -> pd.DataFrame:
    measure_ids = [
        str(measure_id)
        for pair in EARS_MEASURES.values()
        for measure_id in pair.values()
    ]
    payload = {
        "measureIds": ",".join(measure_ids),
        "timeCodes": None,
        "startTimeCode": "2000",
        "endTimeCodeExcl": EARS_END_YEAR_EXCLUSIVE,
        "geoLevel": "2",
    }
    response = _json_request(EARS_ENDPOINT, payload)
    rows = response["GetMeasureResultsForTimePeriodAndGeoLevelResult"]["MeasureResults"]
    raw = pd.DataFrame(rows)
    raw = raw[raw["GeoCountry"].isin(COUNTRIES)].copy()
    raw["year"] = pd.to_numeric(raw["TimeCode"], errors="raise").astype(int)
    raw["value"] = pd.to_numeric(raw["YValue"], errors="coerce")

    output = []
    for antibiotic, measures in EARS_MEASURES.items():
        tested = raw.loc[raw.MeasureId.eq(measures["tested"]), ["GeoCountry", "year", "value"]]
        tested = tested.rename(columns={"value": "tested"})
        resistant = raw.loc[
            raw.MeasureId.eq(measures["resistant"]), ["GeoCountry", "year", "value"]
        ].rename(columns={"value": "resistant"})
        frame = tested.merge(resistant, on=["GeoCountry", "year"], how="left")
        frame = frame[frame.tested.gt(0)].copy()
        frame["resistant"] = frame["resistant"].fillna(0)
        frame["tested"] = frame["tested"].round().astype(int)
        frame["resistant"] = frame["resistant"].round().astype(int)
        frame["antibiotic"] = antibiotic
        output.append(frame)

    ears = pd.concat(output, ignore_index=True)
    ears["iso3"] = ears.GeoCountry.map(lambda code: COUNTRIES[code][0])
    ears["country"] = ears.GeoCountry.map(lambda code: COUNTRIES[code][1])
    ears["region"] = "EU_EEA"
    ears["pathogen"] = "Escherichia coli"
    ears["specimen"] = "Blood/invasive"
    ears["ast_standard"] = np.where(
        ears.year.ge(2020), "EUCAST", "EARS-Net mixed/not reported"
    )
    ears["ast_version"] = np.where(
        ears.year.ge(2020),
        "EUCAST required; exact version unavailable in Atlas aggregate",
        "Not reported in Atlas aggregate",
    )
    ears["test_method"] = "ECDC Surveillance Atlas aggregate"
    ears["source_url"] = EARS_SOURCE
    columns = [
        "region", "country", "iso3", "year", "pathogen", "specimen",
        "antibiotic", "resistant", "tested", "ast_standard", "ast_version",
        "test_method", "source_url",
    ]
    ears = ears[columns].sort_values(["iso3", "antibiotic", "year"])
    return ears.reset_index(drop=True)


def _clean_country_name(value) -> str:
    value = str(value).replace("\u00a0", " ").strip()
    value = re.sub(r"\s*\([a-z0-9]+\).*?$", "", value, flags=re.IGNORECASE)
    aliases = {
        "Czech Republic": "Czechia",
        "Slovak Republic": "Slovakia",
        "The Netherlands": "Netherlands",
    }
    return aliases.get(value, value)


def _read_esac_trend_sheet(path: Path, sheet: str, sector: str, family: str) -> pd.DataFrame:
    table = pd.read_excel(path, sheet_name=sheet, header=1)
    country_column = table.columns[0]
    year_columns = [column for column in table.columns if str(column).split(".")[0].isdigit()]
    keep = [country_column] + year_columns
    table = table[keep].rename(columns={country_column: "country"})
    table["country"] = table.country.map(_clean_country_name)
    table = table[table.country.isin(COUNTRY_TO_ISO3)].copy()
    long = table.melt(id_vars="country", var_name="year", value_name="ddd_per_1000_per_day")
    long["year"] = long.year.map(lambda value: int(float(value)))
    long["ddd_per_1000_per_day"] = pd.to_numeric(
        long.ddd_per_1000_per_day, errors="coerce"
    )
    long = long.dropna(subset=["ddd_per_1000_per_day"])
    long["iso3"] = long.country.map(COUNTRY_TO_ISO3)
    long["sector"] = sector
    long["antibiotic_family"] = family
    return long


def _read_reserve_percentage(path: Path) -> pd.DataFrame:
    table = pd.read_excel(path, sheet_name="D24_Reserve_HC", header=1)
    country_column = table.columns[0]
    year_columns = [column for column in table.columns if str(column).split(".")[0].isdigit()]
    table = table[[country_column] + year_columns].rename(columns={country_column: "country"})
    table["country"] = table.country.map(_clean_country_name)
    table = table[table.country.isin(COUNTRY_TO_ISO3)].copy()
    long = table.melt(id_vars="country", var_name="year", value_name="reserve_pct")
    long["year"] = long.year.map(lambda value: int(float(value)))
    long["reserve_pct"] = pd.to_numeric(long.reserve_pct, errors="coerce")
    if long.reserve_pct.max(skipna=True) <= 1:
        long["reserve_pct"] *= 100
    return long.dropna(subset=["reserve_pct"])


def prepare_esac() -> pd.DataFrame:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    workbook = RAW_DIR / "esac_net_aer_2022_tables.xlsx"
    if not workbook.exists():
        urllib.request.urlretrieve(ESAC_FILE_URL, workbook)
    frames = [
        _read_esac_trend_sheet(workbook, sheet, sector, family)
        for (sector, family), sheet in ESAC_SHEETS.items()
    ]
    consumption = pd.concat(frames, ignore_index=True)
    reserve = _read_reserve_percentage(workbook)
    consumption = consumption.merge(reserve, on=["country", "year"], how="left")
    consumption["access_pct"] = np.nan
    consumption["broad_narrow_ratio"] = np.nan
    consumption["oral_parenteral_ratio"] = np.nan
    consumption["source_url"] = ESAC_SOURCE
    columns = [
        "country", "iso3", "year", "sector", "antibiotic_family",
        "ddd_per_1000_per_day", "access_pct", "broad_narrow_ratio",
        "oral_parenteral_ratio", "reserve_pct", "source_url",
    ]
    return consumption[columns].sort_values(
        ["iso3", "year", "sector", "antibiotic_family"]
    ).reset_index(drop=True)


def _world_bank_indicator(indicator: str, start_year: int, end_year: int) -> pd.DataFrame:
    url = (
        f"https://api.worldbank.org/v2/country/all/indicator/{indicator}"
        f"?format=json&per_page=20000&date={start_year}:{end_year}"
    )
    response = _json_request(url)
    records = response[1] if response and len(response) > 1 else []
    rows = [
        {"iso3": row.get("countryiso3code"), "year": int(row["date"]), "value": row["value"]}
        for row in records
        if row.get("countryiso3code") in ISO3_TO_COUNTRY
    ]
    return pd.DataFrame(rows)


def prepare_demographics(start_year: int = 2000, end_year: int = 2024) -> pd.DataFrame:
    base = pd.MultiIndex.from_product(
        [sorted(ISO3_TO_COUNTRY), range(start_year, end_year + 1)],
        names=["iso3", "year"],
    ).to_frame(index=False)
    for column, indicator in WORLD_BANK_INDICATORS.items():
        values = _world_bank_indicator(indicator, start_year, end_year)
        values = values.rename(columns={"value": column})
        base = base.merge(values, on=["iso3", "year"], how="left")
    base["country"] = base.iso3.map(ISO3_TO_COUNTRY)
    base["source_url"] = WORLD_BANK_SOURCE
    columns = [
        "country", "iso3", "year", "population", "age65_pct", "urban_pct",
        "gdp_per_capita", "health_expenditure_pct_gdp", "source_url",
    ]
    return base[columns].sort_values(["iso3", "year"]).reset_index(drop=True)


def prepare_mapping() -> pd.DataFrame:
    return pd.DataFrame(
        ANTIBIOTIC_MAPPING.items(), columns=["antibiotic", "antibiotic_family"]
    )


def write_manifest(outputs: dict[str, pd.DataFrame]) -> pd.DataFrame:
    records = [
        {
            "file": "ears_net_ecoli_blood.csv",
            "source": "ECDC Surveillance Atlas / EARS-Net",
            "source_url": EARS_SOURCE,
            "years": f"{outputs['ears'].year.min()}-{outputs['ears'].year.max()}",
            "rows": len(outputs["ears"]),
            "retrieved": date.today().isoformat(),
            "limitations": (
                "Antibiotic-group aggregates; exact pre-2020 AST standard and exact "
                "EUCAST version are unavailable in the Atlas aggregate."
            ),
        },
        {
            "file": "esac_net_consumption.csv",
            "source": "ECDC ESAC-Net AER 2022 downloadable tables",
            "source_url": ESAC_SOURCE,
            "years": f"{outputs['consumption'].year.min()}-{outputs['consumption'].year.max()}",
            "rows": len(outputs["consumption"]),
            "retrieved": date.today().isoformat(),
            "limitations": (
                "ATC-group aggregates. Historical Access, broad/narrow and oral/parenteral "
                "indicators are unavailable in this workbook; Reserve percentage is retained."
            ),
        },
        {
            "file": "demographics.csv",
            "source": "World Bank Indicators API",
            "source_url": WORLD_BANK_SOURCE,
            "years": f"{outputs['demographics'].year.min()}-{outputs['demographics'].year.max()}",
            "rows": len(outputs["demographics"]),
            "retrieved": date.today().isoformat(),
            "limitations": "Some indicators have delayed publication or country-year missingness.",
        },
        {
            "file": "antibiotic_mapping.csv",
            "source": "Project mapping of EARS-Net groups to ESAC-Net ATC groups",
            "source_url": ESAC_SOURCE,
            "years": "Not applicable",
            "rows": len(outputs["mapping"]),
            "retrieved": date.today().isoformat(),
            "limitations": (
                "J01C and J01D are broader consumption groups than aminopenicillins and "
                "third-generation cephalosporins; J01G consumption is unavailable in the workbook."
            ),
        },
    ]
    manifest = pd.DataFrame(records)
    manifest.to_csv(ROOT / "data" / "SOURCE_MANIFEST.csv", index=False)
    return manifest


def main():
    INPUT_DIR.mkdir(parents=True, exist_ok=True)
    outputs = {
        "ears": prepare_ears(),
        "consumption": prepare_esac(),
        "demographics": prepare_demographics(),
        "mapping": prepare_mapping(),
    }
    paths = {
        "ears": INPUT_DIR / "ears_net_ecoli_blood.csv",
        "consumption": INPUT_DIR / "esac_net_consumption.csv",
        "demographics": INPUT_DIR / "demographics.csv",
        "mapping": INPUT_DIR / "antibiotic_mapping.csv",
    }
    for name, frame in outputs.items():
        frame.to_csv(paths[name], index=False)
        print(f"{paths[name]}: {len(frame):,} rows")
    manifest = write_manifest(outputs)
    print("\nSource manifest")
    print(manifest.to_string(index=False))


if __name__ == "__main__":
    main()
