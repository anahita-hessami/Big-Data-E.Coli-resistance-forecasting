# Phase 1 data-source checklist

Checked: 31 August 2026

## A. EARS-Net resistance outcome — completed

Source: [ECDC Surveillance Atlas](https://www.ecdc.europa.eu/en/surveillance-atlas-infectious-diseases)

The ECDC Surveillance Atlas REST service was queried with the following restrictions:

- health topic: antimicrobial resistance;
- pathogen: *Escherichia coli*;
- specimen: blood or the Atlas invasive-isolate definition;
- geography: EU/EEA country;
- frequency: annual;
- include each available systemic antibiotic/antibiotic group;
- retain resistant isolates and tested/interpretable isolates, not only percentages;
- retain the complete historical period available under documented definitions.

`data/input/ears_net_ecoli_blood.csv` contains 3,192 rows from 30 EU/EEA countries and 2000–2023. `ast_standard=EUCAST` is used only from 2020 onward. Earlier years remain `EARS-Net mixed/not reported`, and unavailable exact versions are stated rather than inferred.

## B. ESAC-Net consumption and prescribing proxies — completed with limitations

Source: [ECDC ESAC-Net](https://www.ecdc.europa.eu/en/about-us/partnerships-and-networks/disease-and-laboratory-networks/esac-net)

The 2022 Annual Epidemiological Report workbook supplied annual country-level consumption for:

- community sector;
- hospital sector;
- total systemic antibacterials;
- antibiotic classes matching the selected EARS-Net outcomes;
- DDD per 1,000 inhabitants per day;
- Reserve-antibiotic percentage in the hospital sector.

`data/input/esac_net_consumption.csv` contains 1,770 rows covering 2013–2022. Historical Access percentage, broad-to-narrow ratio and oral-to-parenteral ratio were not available in this workbook and are deliberately left missing. DDD means **defined daily dose**, a standardised technical unit used to compare medicine consumption; it is not necessarily the exact prescribed dose for an individual patient.

## C. Demographics — completed

Use one consistent public source across all EU/EEA countries, preferably Eurostat or the World Bank. Required fields are country, ISO3 code, year, population and percentage aged 65+. Optional fields are urbanisation, GDP per capita and health expenditure as a percentage of GDP.

`data/input/demographics.csv` contains 720 World Bank country-year rows covering 2000–2023. No future demographic values were fabricated.

## D. Antibiotic-family mapping — completed with documented approximations

`data/input/antibiotic_mapping.csv` contains five mappings. J01C and J01D are broader than aminopenicillin and third-generation-cephalosporin resistance respectively; this is documented. J01G aminoglycoside consumption is unavailable rather than substituted with an unrelated total.

## Acceptance criteria applied before training

- No urinary or mixed-specimen observations.
- One row per country–year–antibiotic outcome.
- Resistant count does not exceed tested count.
- Country codes match across all input files.
- Every antibiotic has one consumption-family mapping.
- Breakpoint standard and version are recorded or explicitly marked unavailable.
- At least six distinct target years exist for a proposed horizon.
- Five-year modelling is withheld if historical coverage is insufficient.

## Later regional data

US source: [CDC NHSN Antimicrobial Use and Resistance](https://www.cdc.gov/nhsn/psc/aur/index.html).
Japan source: [Japanese national AMR data](https://id-info.jihs.go.jp/en/relevant-information/antimicrobial-resistant/20101112/janis-glass-excel-en.html).

These sources will be processed in separate regional pipelines. Their aggregate CLSI/FDA-defined percentages will not be appended directly to the EUCAST-defined EU outcome table.
