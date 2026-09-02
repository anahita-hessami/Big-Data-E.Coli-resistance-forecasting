# Locked validation protocol for research-grade extension

Version: 2.0
Scope: country-level EU/EEA bloodstream/invasive *E. coli* resistance forecasting

## Intended estimand

For each antibiotic group and forecast horizon, estimate resistance percentage in a future target year using information available by the forecast origin year. Performance is evaluated for the same surveillance population, not for individual patients.

## Outcomes and horizons

- Outcome: percentage resistant among tested bloodstream/invasive *E. coli* isolates.
- Direct forecast horizons: 1, 3 and 5 years.
- Unit of prediction: country–antibiotic–target-year.
- Primary comparator: persistence, defined as the resistance percentage at the forecast origin.

## Primary evaluation

Nested rolling-origin backtesting is the primary internal evaluation. At every outer target year, feature selection and model tuning are repeated using earlier target years only. The outer target year is not used for imputation fitting, feature decisions, hyperparameter selection, conformal calibration or model fitting.

Frozen-specification backtesting is a secondary robustness analysis because the specification was originally selected using the full development era.

## Metrics

Primary metrics are RMSE and forecast skill relative to persistence:

`skill = 1 - RMSE(model) / RMSE(persistence)`

Secondary metrics are MAE, isolate-weighted RMSE, weighted skill and R². R² is not used alone to claim superiority because resistance is strongly persistent.

## Statistical uncertainty

The model-minus-persistence RMSE difference and skill receive 95% intervals from a two-way country × target-year cluster bootstrap. A standard row bootstrap is prohibited. Diebold–Mariano testing is withheld until enough independent forecast origins exist for a credible time-series loss comparison.

## Adaptive feature search

The complete feature-selection pipeline is repeated inside each outer split. Supplementary feature evidence uses:

- temporal fold win rate;
- bootstrap selection stability;
- circular residual null experiments;
- Benjamini–Hochberg q-values; and
- the null distribution of the maximum apparent feature improvement.

An added feature is not retained solely because of correlation or one favorable fold.

## Partial pooling

The hierarchical benchmark uses a penalized country random intercept. Country effects are shrunk toward the EU/EEA mean according to available observations. Unseen countries receive the pooled intercept of zero. This is compared with the existing fixed-country tree model and is not used to combine EUCAST with CLSI/FDA outcomes.

## Prediction intervals

Prequential conformal intervals use only errors from earlier rolling origins. Coverage and width are reported at 80% and 95% nominal levels. Antibiotic-specific calibration is used only when enough earlier errors are available; otherwise calibration is pooled.

## Claims policy

- A positive point estimate of skill is described as an observed improvement, not proof of generalization.
- If the 95% uncertainty interval crosses zero, the advantage is described as uncertain.
- Deployment readiness requires later prospective temporal validation and decision-impact assessment.
- No output may be presented as patient-level treatment guidance.
