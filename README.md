# Regional bloodstream *E. coli* AMR forecasting

This project begins the scientifically revised global model. It replaces naïve pooling of EUCAST and CLSI/FDA resistance percentages with a staged regional design:

1. develop and test the primary EU/EEA model using EARS-Net outcomes;
2. add ESAC-Net antibiotic consumption, prescribing-behaviour proxies and demographic features;
3. build separate US and Japanese models under their native laboratory standards;
4. compare transportability and later connect the regional models hierarchically.

The project predicts country-level surveillance outcomes. It is not a patient-level susceptibility model or a treatment-recommendation tool.

## Key results at a glance

- **5-year forecast**: Extra Trees beats persistence by **43% lower RMSE** (1.977 vs 3.472), with a 95% cluster-bootstrap interval that stays clear of zero (32.3%–57.6% skill).
- **3-year forecast**: Extra Trees beats persistence by **27% lower RMSE**, and — unlike the 5-year result — this holds up under a *fully nested* backtest that re-runs feature and model selection at every historical cutoff rather than reapplying a frozen specification in hindsight.
- **1-year forecast**: simple persistence (`next value = latest value`) wins. The apparent 1-year ML edge in the frozen backtest reverses once evaluated with the stricter nested design, and that reversal is reported rather than smoothed over.
- **Checked against real subsequent data**: the 2024 forecast — built entirely from data through 2023 — was compared against ECDC's actual published 2024 EARS-Net report (released November 2025, after this project's inputs were collected). Mean absolute error across the five antibiotic classes: **0.47 percentage points**.
- **Multiplicity-controlled feature search**: of ~40 candidate engineered features, only **1** survives Benjamini–Hochberg plus max-null permutation correction — reported plainly rather than presenting single-fold "significant" results uncorrected.

Full tables and methodology are below; see `RESEARCH_GRADE_RESULTS_SUMMARY.md` and `docs/` for the complete validation writeup.

## Primary research question

> To what extent do recent resistance trends, antibiotic consumption, prescribing behaviour and demographic changes improve 1-, 3- and 5-year forecasts of bloodstream *E. coli* resistance across EU/EEA countries?

## Why EU comes first

EARS-Net provides invasive-isolate resistance surveillance and requires EUCAST for participation from 2020 onward. ESAC-Net provides country-level community and hospital antimicrobial-consumption indicators. Keeping the primary development region internally coherent prevents the model from learning artificial differences caused by EUCAST–CLSI/FDA breakpoint definitions.

Official starting points:

- [EARS-Net data and protocol](https://www.ecdc.europa.eu/en/about-us/networks/disease-networks-and-laboratory-networks/ears-net-data)
- [ECDC Surveillance Atlas](https://www.ecdc.europa.eu/en/surveillance-atlas-infectious-diseases)
- [ESAC-Net](https://www.ecdc.europa.eu/en/about-us/partnerships-and-networks/disease-and-laboratory-networks/esac-net)
- [EUCAST breakpoint tables](https://www.eucast.org/bacteria/clinical-breakpoints-and-interpretation/clinical-breakpoint-tables/)
- [CDC NHSN AUR](https://www.cdc.gov/nhsn/psc/aur/index.html)
- [Japanese national AMR data](https://id-info.jihs.go.jp/en/relevant-information/antimicrobial-resistant/20101112/janis-glass-excel-en.html)

## Phase 1 data now included

The four required inputs have been assembled from official public sources and are
included under `data/input/`:

- EARS-Net bloodstream/invasive *E. coli* counts: 3,192 country-year-antibiotic-group
  rows, 30 EU/EEA countries, 2000–2023;
- ESAC-Net consumption and Reserve-use indicator: 1,770 rows, 2013–2022;
- World Bank demographics: 750 country-year rows, 2000–2024;
- a five-row EARS-Net-to-ESAC-Net antibiotic-family mapping.

`data/SOURCE_MANIFEST.csv` records source URLs, retrieval dates, row counts and
limitations. The inputs can be rebuilt from the original public sources with:

```bash
python src/prepare_official_inputs.py
```

The Atlas supplies antibiotic-group outcomes, not every individual antibiotic.
The analysis therefore forecasts aminopenicillin, fluoroquinolone,
third-generation-cephalosporin, aminoglycoside and carbapenem resistance groups.

## Getting started

The notebook to run is `notebooks/Regional_Ecoli_BSI_AMR_Complete_ML.ipynb`. It trains all
candidate models itself and never trains on invented or synthetic observations.

**Locally:**

```bash
pip install -r requirements.txt
jupyter notebook notebooks/Regional_Ecoli_BSI_AMR_Complete_ML.ipynb
```

**In Google Colab:**

1. Upload the complete project ZIP to Colab and unzip it into `/content/`.
2. Open `regional_ecoli_amr/notebooks/Regional_Ecoli_BSI_AMR_Complete_ML.ipynb`.
3. Run cells from top to bottom. The four populated input CSVs are already included, and the
   notebook locates the project root automatically regardless of which folder it's opened from.

To rerun everything from the command line instead, use:

```bash
pip install -r requirements.txt
python -m src.run_official_pipeline
python -m src.run_iterative_optimization
```

Trained model files (`*.joblib`) are not committed to this repository — they're large
(tens of MB) and fully regenerable offline from the committed `data/input/*.csv` files with
the commands above, plus the notebook itself, which trains and saves its own copies under
`outputs/notebook_ml/`. Everything downstream of `data/input/` is deterministic (`random_state=42`
throughout), so a fresh run reproduces the same numbers reported in this README.

Prefer to see the results without running anything? Open
`notebooks/Regional_Ecoli_BSI_AMR_Complete_ML_EXECUTED.ipynb` — the same notebook, pre-run with
every table and chart already rendered.

## Repository layout

```
notebooks/                 Complete_ML.ipynb (run this) and its pre-executed copy
notebooks/archive/         earlier Phase 1 notebook, kept for methodology history
src/                        pipeline code (data prep, modelling, research-grade validation)
data/input/                 the four official source extracts used for modelling
data/processed/             engineered feature panels built from data/input/
outputs/                    result tables, figures, and (after regeneration) trained models
outputs/optimization/       iterative feature/model search results
outputs/research_grade/     nested backtest, bootstrap, conformal, and multiplicity-screen results
docs/                       TRIPOD+AI checklist, PROBAST+AI self-assessment, model card, validation protocol
tests/                      pytest suite covering leakage-safety and validation logic
```

### If charts do not appear

The revised notebook forces Matplotlib inline mode and explicitly displays every saved PNG inside the notebook. It also searches for the four input CSVs in `data/input/`, beside the notebook, in the current working directory and in `/content/` for direct Colab uploads.

After running the dataset-status cell, confirm that all four rows show `Found = True`. If any row is `False`, only the model-development workflow can be displayed; data-driven resistance, feature and model charts cannot be calculated until all four official inputs are available. Every displayed chart is also saved under `outputs/figures/`.

## Visualisations

When the official inputs are present, the notebook automatically creates a detailed explanatory visual suite (up to roughly fifty outputs, depending on data coverage), including:

- country–year reporting coverage and antibiotic-coverage charts;
- testing-volume and breakpoint-version diagnostics;
- resistance distributions, longitudinal trends and country–antibiotic heatmaps;
- consumption and available Reserve-antibiotic-use trends;
- consumption–resistance and demographic–resistance scatter plots;
- a predictor correlation heatmap;
- feature missingness, distributions and availability-over-time diagnostics;
- Pearson, Spearman and mutual-information feature evidence for every horizon;
- binned feature–outcome relationship plots;
- validation/test feature-group ablation for consumption, prescribing and demographics;
- one-at-a-time validation ablation for every added numeric feature;
- cross-horizon permutation-importance stability;
- model and horizon RMSE comparisons;
- observed-versus-predicted, residual and calibration diagnostics;
- antibiotic-specific performance and country–antibiotic error heatmaps;
- permutation feature importance;
- interactive observed, predicted and forecast-error maps for every horizon.

All plots are saved under `outputs/figures/`, with a machine-readable `outputs/visualisation_index.csv`.

The supporting feature tables are saved as:

- `outputs/feature_evidence_h1.csv`, `feature_evidence_h3.csv` and `feature_evidence_h5.csv`;
- `outputs/feature_group_ablation_h1.csv`, `feature_group_ablation_h3.csv` and `feature_group_ablation_h5.csv`;
- `outputs/feature_group_ablation_all_horizons.csv`.
- `outputs/individual_feature_ablation_h1.csv`, `individual_feature_ablation_h3.csv` and `individual_feature_ablation_h5.csv`;
- `outputs/individual_feature_ablation_all_horizons.csv`.

Feature retention is based primarily on chronological validation RMSE, not correlation alone. The notebook holds the selected algorithm and hyperparameters constant during ablation, then labels a feature-group addition as improved, worsened or showing no material change relative to the resistance-history base model.

## Methodological safeguards

- EU, US and Japanese resistance labels remain region-specific.
- Every AMR row records the AST standard and version.
- EUCAST I is not automatically combined with R.
- Input rows must be *E. coli* and blood/invasive specimens.
- Duplicated country–year–antibiotic rows fail validation.
- All forecast features are known by the prediction origin year.
- Model selection uses chronological validation; the final year remains untouched for testing.
- Persistence, meaning “next value equals the latest value,” is the mandatory baseline.
- The 1-, 3- and 5-year horizons are fitted as separate direct models.
- ANN is included as a candidate, but is selected only if chronological validation supports it.
- Feature additions are assessed separately against the same resistance-history base model.
- The last target year is used only for final confirmation, not for deciding which features to retain.

## Breakpoint policy

The public surveillance sources usually contain aggregated resistant/tested counts rather than raw MIC measurements. These aggregates cannot be reliably converted from CLSI/FDA to EUCAST. `breakpoint_registry.csv` records the definition applied to every regional source and year. An antibiotic may enter a later pooled sensitivity analysis only when comparability is documented; otherwise it stays in its regional model.

## Completed Phase 1 results

The real-data pipeline, visual analysis, chronological model selection, final testing,
feature ablation and saved models are complete. The algorithms compared were Ridge,
Random Forest, Gradient Boosting and a multilayer-perceptron ANN, with persistence as
the mandatory baseline.

| Horizon | Selected model | Test RMSE | Test MAE | Test R2 | Persistence RMSE | Interpretation |
|---:|---|---:|---:|---:|---:|---|
| 1 year | Ridge | 2.975 | 1.644 | 0.976 | 2.461 | Did not beat persistence |
| 3 years | Gradient Boosting | 2.295 | 1.571 | 0.986 | 2.331 | Only a 1.5% RMSE gain; weighted RMSE was worse |
| 5 years | Gradient Boosting | 2.672 | 1.859 | 0.981 | 3.472 | Clear 23.0% RMSE improvement |

High R2 values are expected because resistance levels are strongly persistent and
must not be interpreted alone. RMSE and MAE against the persistence baseline are the
main evidence. The five-year model is the strongest research result; the one-year
model should not replace persistence, and the three-year result is marginal.

On chronological validation, demographics helped the one-year resistance-history
model, the available prescribing proxy helped the three-year model, and the complete
feature set helped most at five years. These are predictive associations, not causal
effects. Exact results are in `REAL_DATA_RESULTS_SUMMARY.md` and `outputs/`.

## Iterative feature and model optimisation

The second-stage search added 41 candidate variables and compared six model families
using expanding-year validation. A feature was retained only when it reduced pooled
validation RMSE by at least 0.5% and improved at least 60% of validation folds. Run it
reproducibly with:

```bash
python -m src.run_iterative_optimization
```

| Horizon | Recommended forecast | Rolling-CV RMSE | Confirmation RMSE | Persistence RMSE | Decision |
|---:|---|---:|---:|---:|---|
| 1 year | Persistence | 2.101 for optimized Ridge | 3.034 | 2.461 | Reject optimized model |
| 3 years | Extra Trees | 2.325 | 2.088 | 2.331 | Use optimized model |
| 5 years | Extra Trees | 2.393 | 1.977 | 3.472 | Use optimized model |

Relative to the previous models, confirmation RMSE improved by 9.0% at three years
and 26.0% at five years. The five-year optimized model reduced RMSE by 43.1% relative
to persistence. The one-year model remains unsuitable because its rolling-validation
improvement did not generalize to the confirmation year.

Optimization outputs, experiment logs, saved models and figures are under
`outputs/optimization/`. The latest year had already been inspected during earlier
development, so it is a confirmation period rather than a pristine external test.

## Research-grade validation extension

The project now addresses the main weaknesses of the original confirmation analysis:

- frozen-specification and fully nested rolling-origin backtests at historical cutoffs;
- a country × target-year cluster bootstrap against persistence;
- multiplicity-controlled feature screening with Benjamini–Hochberg, a maximum-null
  adjustment and stability selection;
- a penalized country random-intercept benchmark for partial pooling;
- prequential 80% and 95% conformal prediction intervals;
- exact dependency pins, a lockfile, automated tests and GitHub Actions CI;
- model metadata sidecars, source provenance and data-update audits;
- TRIPOD+AI and PROBAST+AI reporting/self-assessment documents.

Run the extension with:

```bash
python -m src.run_research_grade_pipeline \
  --max-frozen-origins 8 --nested-origins 3 \
  --bootstrap-repetitions 2000 --null-repetitions 200 \
  --stability-repetitions 2000
python -m src.consolidate_research_outputs
```

The included engineering run used 20 null-search repetitions; this is sufficient to
verify the workflow but produces coarse adjusted p-values. Use at least 200 before a
publication claim.

| Horizon | Frozen RMSE | Persistence RMSE | Frozen skill | Cluster-bootstrap skill (95% interval) | Nested result |
|---:|---:|---:|---:|---:|---|
| 1 year | 2.329 | 2.343 | 0.6% | 0.9% (−21.8% to 14.6%) | Does not beat persistence |
| 3 years | 2.186 | 3.028 | 27.8% | 27.0% (10.1% to 38.1%) | 27.3% skill across 3 cutoffs |
| 5 years | 2.191 | 3.962 | 44.7% | 44.8% (32.3% to 57.6%) | 45.2% skill, but only 1 valid nested cutoff |

Accordingly, persistence remains the recommended one-year forecast. The three-year
model has the strongest combination of repeated fully nested evidence and clustered
uncertainty. The five-year model is highly promising in the frozen analysis, but the
short data history permits only one fully nested cutoff, so it needs prospective
confirmation before operational use. Partial pooling did not improve RMSE in the
current same-country temporal evaluation, although it remains useful as an explicit
shrinkage benchmark for small or previously unseen countries.

Full tables and interpretation are in `RESEARCH_GRADE_RESULTS_SUMMARY.md` and
`outputs/research_grade/`. The complete executable notebook is
`notebooks/Regional_Ecoli_BSI_AMR_Complete_ML.ipynb`.

Unlike the earlier reporting-only notebook, the complete notebook visibly:

- constructs Ridge, Gradient Boosting, Random Forest, Extra Trees and ANN models;
- tunes every model with chronological GridSearchCV;
- selects a model independently at each 1-, 3- and 5-year horizon;
- evaluates the selected estimator on the latest confirmation year;
- refits the frozen winning pipeline on all labelled observations;
- saves three fitted joblib models;
- generates country–antibiotic forecasts for 2024, 2026 and 2028;
- reports both the ML and persistence forecast with 80% and 95% uncertainty bands.

The notebook contains 56 cells and has model training enabled by default.
Two versions are included under `notebooks/`:

- `Regional_Ecoli_BSI_AMR_Complete_ML.ipynb`: clean rerunnable notebook;
- `Regional_Ecoli_BSI_AMR_Complete_ML_EXECUTED.ipynb`: the same notebook with
  all 27 code cells executed and 43 saved outputs, including tables and charts.

## Principal limitations

- The ECDC Atlas aggregates cannot be reinterpreted under a common breakpoint table.
  Pre-2020 standards are therefore recorded as mixed/not reported; 2020 onward is
  marked EUCAST-required with the exact version unavailable.
- ESAC consumption begins in 2013 and ends in 2022, limiting target-year coverage.
- The public workbook lacks the requested historical Access, broad-to-narrow and
  oral-to-parenteral series. Reserve percentage is the only available prescribing-mix
  proxy in this build.
- J01C and J01D consumption groups are broader than the corresponding EARS-Net
  resistance groups, and no J01G aminoglycoside consumption series was available.
- Country-level ecological predictions cannot be applied to individual patients or
  used as treatment recommendations.
- The public 2024 EARS-Net report is available, but the checked Atlas export still
  returned data only through 2023. ESAC-Net machine-readable history in this package
  still ends in 2022. New years must be rebuilt from coherent official exports rather
  than appended from incompatible tables.

## Health-economic extension

`economic_model/` and `R/` extend the forecast into a decision question: is
activating enhanced stewardship using the 3-year resistance forecast
economically preferable to usual care or to a rule based on the latest
observed resistance alone? The cost-effectiveness analysis (CEA) and the
2026–2030 budget-impact analysis (BIA) are kept as separate models with
separate parameter files.

R is the authoritative stochastic engine (50,000-draw PSA with Monte Carlo
confidence intervals); a Python implementation (`src/economic_extension.py`)
is an independent formula check, not the primary result. Start with
`economic_model/README.md` for how to run both, `VERSION_9_CHANGELOG.md` for
what changed and why, and `docs/ECONOMIC_VALIDATION_V9.md` for the full
scientific interpretation, assumptions, and remaining limitations.

The intervention cost, baseline mortality, QALY loss, and BIA uptake inputs
are still illustrative pending jurisdiction-specific evidence — this is a
research scenario, not a policy recommendation.

## License

Released under the [MIT License](LICENSE).
