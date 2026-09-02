# Model card: regional bloodstream *E. coli* AMR forecasts

## Intended use

Research forecasting of country-level bloodstream/invasive *E. coli* resistance percentages across EU/EEA countries at 1-, 3- and 5-year horizons. The model may support surveillance-method research and hypothesis generation.

## Prohibited use

- selecting treatment for an individual patient;
- interpreting predictions as infection incidence;
- directly combining EUCAST outcomes with CLSI/FDA aggregates;
- ranking healthcare providers or countries without surveillance-quality context;
- automated policy decisions without expert review and prospective validation.

## Inputs

Resistance history, testing volume, four-period trend and volatility, mapped antimicrobial consumption, available prescribing-mix measures, demographics, European AMR context and cross-antibiotic context. Exact features vary by horizon and are stored in model metadata sidecars.

## Outputs

- resistance percentage point forecasts;
- persistence comparator;
- 80% and 95% prequential conformal intervals when enough earlier calibration errors exist;
- forecast horizon and target year;
- validation and provenance metadata.

## Evaluation

The primary development evaluation is nested rolling-origin backtesting. RMSE skill relative to persistence is the main comparative statistic. Country × target-year clustered bootstrap intervals quantify uncertainty. Frozen-specification results are labelled as retrospective robustness analyses.

## Fairness and representativeness

Country effects can be unstable when surveillance volume is small. A penalized country random-intercept benchmark shrinks small-country estimates toward the European mean. Performance must still be reported by country, antibiotic group and testing-volume strata.

## Known limitations

- ecological rather than patient-level model;
- incomplete prescribing-behaviour series;
- lagged publication and revisions of surveillance data;
- historical breakpoint-version uncertainty;
- limited independent target years for long horizons;
- no prospective deployment evaluation.

## Monitoring requirements

On every data release, compare schema, missingness, testing volume, feature distributions, breakpoint metadata, error and interval coverage with the development era. A model should be retrained only after a versioned data audit and should be rolled back if prospective performance is worse than persistence.
