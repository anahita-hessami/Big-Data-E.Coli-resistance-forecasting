# Research-grade validation results

## Purpose

This extension tests whether the earlier single-confirmation-year improvements were
repeatable. It evaluates country-level forecasts of bloodstream/invasive *E. coli*
resistance across EU/EEA countries at 1-, 3- and 5-year horizons. Persistence—the
latest observed resistance percentage—is the mandatory comparator.

The frozen-specification backtest repeatedly applies the previously chosen model and
feature set at earlier historical cutoffs. The stricter nested backtest repeats feature
search and model tuning using only information available inside each cutoff. Neither
design should be interpreted as prospective external validation.

## Main performance results

| Horizon | Evaluation | Forecast rows | Model RMSE | Persistence RMSE | RMSE skill | Tested-count weighted skill |
|---:|---|---:|---:|---:|---:|---:|
| 1 year | Frozen specification | 720 | 2.329 | 2.343 | 0.6% | −0.4% |
| 1 year | Fully nested pipeline | 432 | 2.390 | 2.227 | −7.3% | −10.9% |
| 3 years | Frozen specification | 427 | 2.186 | 3.028 | 27.8% | 40.0% |
| 3 years | Fully nested pipeline | 427 | 2.203 | 3.028 | 27.3% | 38.2% |
| 5 years | Frozen specification | 427 | 2.191 | 3.962 | 44.7% | 54.9% |
| 5 years | Fully nested pipeline | 139 | 1.902 | 3.472 | 45.2% | 57.7% |

The one-year apparent frozen advantage is negligible and reverses in the nested
analysis. Persistence should therefore remain the one-year forecast. The three-year
result is the most defensible: model selection was repeated at three historical
cutoffs and the model retained a substantial advantage. The five-year frozen result
also repeated across three origins, but only one cutoff had enough earlier target
years to support the fully nested feature-and-model search. It is promising rather
than operationally confirmed.

## Clustered uncertainty

The two-way bootstrap resamples countries and target years, avoiding the unrealistic
assumption that every country–antibiotic row is independent.

| Horizon | Bootstrapped RMSE skill | 95% interval | Probability model is worse |
|---:|---:|---:|---:|
| 1 year | 0.9% | −21.8% to 14.6% | 38.1% |
| 3 years | 27.0% | 10.1% to 38.1% | 0.7% |
| 5 years | 44.8% | 32.3% to 57.6% | <0.1% in 1,000 replicates |

These intervals support an advantage at three and five years in the frozen analysis,
but not at one year. The probability column is a bootstrap diagnostic, not a formal
causal probability or a guarantee of future performance.

## Prediction intervals

Prequential conformal intervals use only errors from earlier target years. Empirical
coverage was 80.7%, 84.8% and 88.3% for nominal 80% intervals at horizons 1, 3 and 5,
respectively. Nominal 95% intervals covered 97.2%, 96.8% and 97.5%. Their average full
widths were approximately 4.35–5.44 resistance-percentage points at 80% and
9.86–10.24 points at 95%. Coverage at the longer horizons is based on only two
evaluable target years and must be updated prospectively.

## Feature-search correction

The adaptive screen compares each candidate with the resistance-history base,
applies Benjamini–Hochberg correction, compares it with the maximum improvement under
a circular-residual null, and requires bootstrap selection stability. In the included
20-null-repetition engineering run, only `origin_year_index` and
`reserve_pct_lag2` at the three-year horizon passed all configured rules. No candidate
passed at one or five years.

Twenty null repetitions give a minimum attainable raw p-value of 1/21 and are not
enough for a publication-grade multiplicity claim. The workflow is implemented and
tested, but a final report should rerun at least 200 null repetitions and 2,000
stability resamples. Candidate retention may change after that run.

## Country partial pooling

The penalized random-intercept benchmark shrinks country effects toward the EU/EEA
mean and assigns the pooled effect to a country not seen during training. It did not
outperform the frozen tree/Ridge specifications in the current temporal evaluation:
its RMSE was 2.334, 2.284 and 2.393 at 1, 3 and 5 years. This negative comparison is
informative. Partial pooling should remain a benchmark for small-country and external
transportability analyses, not be presented as the best current predictor.

## Data currency

World Bank demographic indicators have been rebuilt through 2024. ECDC published
2024 EARS-Net and ESAC-Net reports on 18 November 2025, but the checked Atlas query
still returned the AMR series only through 2023, and a coherent machine-readable
ESAC-Net full history through 2024 was not obtained. The local AMR outcome therefore
ends in 2023 and consumption in 2022. The project deliberately does not splice newer
summary tables into older ATC/DDD series or invent missing prescribing variables.

## Defensible conclusion

This project is a surveillance forecasting study, not a clinical treatment system.
The three-year model is the strongest current result. The five-year model warrants
prospective confirmation when later official data become machine-readable. The
one-year model should remain persistence. Any policy use requires external temporal
validation, drift monitoring, expert review and presentation of prediction intervals.
