# AMR Version 9: Important Updated Files

This compact bundle contains the principal files changed or added for the Version 9 scientific and economic validation work. The complete project remains in `Regional_Ecoli_BSI_AMR_Decision_Validated_v9.zip`.

## Start here

1. `VERSION_9_CHANGELOG.md` — concise record of the Version 9 changes.
2. `docs/ECONOMIC_VALIDATION_V9.md` — scientific interpretation, assumptions, limitations, and result status.
3. `economic_model/README.md` — instructions for running the R cost-effectiveness and budget-impact analyses.
4. `README.md` — overall forecasting-project instructions and structure.

## Authoritative R analysis

- `R/economic_model.R` — cost-effectiveness model, probabilistic sensitivity analysis, correlated uncertainty, and figures.
- `R/run_economic_analysis.R` — command-line runner for the primary economic analysis.
- `R/budget_impact_model.R` — separate five-year budget-impact analysis.
- `R/run_budget_impact_analysis.R` — command-line runner for the budget-impact analysis.
- `economic_model/AMR_Economic_Analysis.qmd` — reproducible cost-effectiveness report.
- `economic_model/AMR_Budget_Impact_Analysis.qmd` — reproducible budget-impact report.

The R implementation is the designated authoritative economic analysis. R was not available in the development runtime, so the final R outputs must be generated locally, in Colab with R support, or through the included GitHub Actions workflow.

## Parameters and evidence

- `economic_model/model_config.csv` — analysis configuration, file locations, seed, and simulation count.
- `economic_model/parameters/economic_parameters.csv` — CEA assumptions and uncertainty distributions.
- `economic_model/parameters/bia_parameters.csv` — five-year BIA assumptions.
- `economic_model/parameters/clinical_effect_evidence.csv` — intervention effect evidence and provenance fields.
- `data/input/ecdc_earsnet_coverage_2023.csv` — surveillance-coverage input.
- `data/input/eurostat_population_projection_2026.csv` — population input used to replace `tested` as the burden denominator.
- `data/SOURCE_MANIFEST.csv` — input-source and provenance register.

## Python formula validation

- `src/economic_extension.py` — independent Python implementation used to validate formulas and generate diagnostic outputs.
- `src/check_economic_consistency.py` — automated stale-output and cross-file consistency checks.

Python economic results are validation outputs, not the authoritative final economic result.

## Tests and automation

- `tests/test_economic_extension.py` — Python economic-model tests.
- `tests/test_economic_model.R` — R unit and consistency tests.
- `.github/workflows/ci.yml` — automated Python and R checks and report generation.

## Key review outputs

- `outputs/decision_validation/historical_trigger_metrics.csv` — rolling-origin trigger sensitivity, specificity, and false-negative metrics.
- `outputs/decision_validation/historical_trigger_predictions.csv` — observation-level historical trigger predictions.
- `outputs/economic_python_reference/deterministic_summary.csv` — deterministic validation summary.
- `outputs/economic_python_reference/psa_summary.csv` — 50,000-draw PSA summary and Monte Carlo interval.
- `outputs/economic_python_reference/correlation_sensitivity.csv` — comparison of cross-country correlation assumptions.
- `outputs/economic_python_reference/five_year_budget_summary.csv` — five-year budget-impact validation summary.
- `outputs/economic_python_reference/one_way_sensitivity.csv` — tornado-analysis results.
- `outputs/economic_python_reference/figures/` — six diagnostic and decision-analysis figures.

The large observation-level file `psa_results.csv` is intentionally excluded from this compact bundle. It is available in the complete Version 9 archive and can be regenerated from the scripts and fixed configuration.
