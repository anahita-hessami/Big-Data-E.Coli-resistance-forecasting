# Real-data model results and decisions

## Scope

This model forecasts annual country-level bloodstream/invasive *Escherichia coli*
resistance percentages across EU/EEA surveillance systems. It covers five EARS-Net
antibiotic groups and direct 1-, 3- and 5-year horizons. It is an ecological
surveillance model, not a patient-level susceptibility or treatment model.

## Data used

| Input | Coverage | Records | Role |
|---|---|---:|---|
| EARS-Net | 30 countries, 2000–2023 | 3,192 | Resistance outcome and test denominators |
| ESAC-Net | 2013–2022 | 1,770 | Community/hospital consumption and Reserve-use proxy |
| World Bank | 30 countries, 2000–2023 | 720 | Population, age, urbanisation, GDP and health expenditure |
| Antibiotic mapping | Five resistance groups | 5 | Links EARS-Net groups to ESAC-Net ATC families |

Model origin years begin in 2013 because this is the start of the available ESAC-Net
series. There were 1,438 non-missing one-year targets, 1,146 three-year targets and
858 five-year targets.

## Model selection and untouched test performance

Models were selected using chronological validation, then evaluated on the latest
held-out target year. Lower RMSE and MAE are better. R2 describes explained variation,
but it is secondary because a strongly persistent time series can have a high R2 even
when a simple persistence forecast performs better.

| Horizon | Selected algorithm | RMSE | MAE | R2 | Weighted RMSE | Persistence RMSE | Decision |
|---:|---|---:|---:|---:|---:|---:|---|
| 1 year | Ridge | 2.975 | 1.644 | 0.976 | 1.518 | 2.461 | Reject as an operational improvement |
| 3 years | Gradient Boosting | 2.295 | 1.571 | 0.986 | 1.740 | 2.331 | Marginal and not robust across metrics |
| 5 years | Gradient Boosting | 2.672 | 1.859 | 0.981 | 2.081 | 3.472 | Strongest result; 23.0% lower RMSE |

For the five-year horizon, the selected model also reduced MAE by 23.1% and
tested-isolate-weighted RMSE by 30.4% relative to persistence. At three years the
unweighted RMSE improvement was only about 1.5%, while MAE and weighted RMSE were
worse. At one year, persistence was clearly superior.

## Do the added feature groups help?

Feature groups were added to the same resistance-history base and assessed with
chronological validation while holding the selected algorithm and hyperparameters
constant.

| Horizon | Best validation addition | RMSE change vs resistance-history base | Interpretation |
|---:|---|---:|---|
| 1 year | Demographics | -3.8% | Modest validation improvement, not enough for final model to beat persistence |
| 3 years | Available prescribing proxy | -7.7% | Useful validation signal, but not robust on the test period |
| 5 years | Full feature set | -18.5% | Consumption, demographics and Reserve-use proxy jointly added useful medium-term signal |

Negative RMSE change means improvement. These experiments support predictive utility,
not causality. Correlations and feature importance do not show that antibiotic use or
demography causes the observed resistance changes.

## Scientific interpretation

The latest resistance level and recent resistance trajectory contain most of the
short-term information. This explains why persistence is difficult to beat at one
year. Over five years, nonlinear interactions between resistance history, consumption,
demography and reporting context appear more useful, which is consistent with the
stronger Gradient Boosting result.

The results do not yet justify a live five-year public forecast. External temporal
validation on newer EARS-Net/ESAC-Net releases, prediction intervals, drift checks and
prospective monitoring are required first.

## Limitations that must accompany any result

1. Public ECDC aggregates do not provide raw MICs, so older mixed standards cannot be
   converted retrospectively to EUCAST. EUCAST is recorded only where participation
   rules support it from 2020 onward.
2. Consumption series end in 2022 and use broader ATC groups than some resistance
   outcomes. Aminoglycoside-family consumption is unavailable in the selected workbook.
3. Historical Access, broad-to-narrow and oral-to-parenteral indicators are unavailable;
   Reserve percentage is the only prescribing-mix proxy used here.
4. Country-level ecological associations cannot be transferred to individual patients.
5. The latest held-out period is only one temporal test. Results may change after
   surveillance revisions or a new external release.

## Recommended next stage

Freeze this EU/EEA model as Phase 1. When newer official releases become available,
run external temporal validation before retraining. Then add uncertainty intervals and
drift monitoring. Separate US and Japanese regional models should be built under their
native CLSI/FDA definitions; resistance percentages should not be naively pooled with
the EU model.

## Subsequent iterative optimisation

A bounded second-stage search engineered 41 additional candidates and compared Ridge,
Random Forest, Extra Trees, Gradient Boosting, Histogram Gradient Boosting and ANN.
Feature decisions used up to five expanding-year validation folds. Additions required
at least a 0.5% pooled RMSE improvement and wins in at least 60% of folds.

| Horizon | Final recommendation | Rolling-CV RMSE | Latest-year confirmation RMSE | Persistence RMSE |
|---:|---|---:|---:|---:|
| 1 year | Persistence | 2.101 for optimized Ridge | 3.034 | 2.461 |
| 3 years | Optimized Extra Trees | 2.325 | 2.088 | 2.331 |
| 5 years | Optimized Extra Trees | 2.393 | 1.977 | 3.472 |

The three-year model retained demographic change, country identity and three lagged
consumption variables. The five-year model retained country identity, three lagged
consumption variables, testing/reporting-burden measures and origin year. The one-year
model's validation gain failed confirmation, so persistence remains the recommended
forecast.

The optimized three- and five-year confirmation RMSE values were 9.0% and 26.0% lower
than the earlier selected models. The five-year result was 43.1% lower than persistence.
Country identity improves prediction for represented countries but reduces
transportability. Reported resistant isolates per 100,000 is a surveillance indicator,
not coverage-corrected disease incidence. Because the latest year had been examined in
earlier development, these results still require external validation on a newer release.
