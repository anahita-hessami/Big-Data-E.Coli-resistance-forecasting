# PROBAST+AI development self-assessment

This is a preliminary self-assessment adapted from [PROBAST+AI](https://www.bmj.com/content/388/bmj-2024-082505). It is not an independent formal appraisal.

| Domain | Current judgement | Evidence and remaining risk |
|---|---|---|
| Participants and data sources | Some concerns | Routine country-level surveillance varies in coverage, testing volume and case mix. Small countries and missing years remain important. |
| Predictors | Some concerns | Predictors are restricted to origin-time information, but consumption begins later than AMR data and several prescribing indicators remain unavailable. |
| Outcome | Some concerns | EARS-Net aggregates are appropriate for EU/EEA forecasting. Exact historical breakpoint versions are not always exposed, and EUCAST/CLSI aggregates cannot be directly pooled. |
| Analysis | Improving; still some concerns | Nested temporal validation, persistence comparison, multiplicity controls, partial pooling and uncertainty intervals are implemented. The number of independent target years remains limited. |
| Applicability | Limited to stated use | Outputs apply to EU/EEA country-level surveillance and cannot be transferred to individual treatment decisions or automatically to US/Japanese systems. |

## Primary safeguards

- direct 1-, 3- and 5-year models;
- time-ordered splitting by target year;
- preprocessing fitted within training data;
- persistence benchmark at every horizon;
- nested selection inside outer target-year splits;
- country × target-year cluster uncertainty;
- prequential conformal calibration;
- breakpoint-standard separation;
- explicit model provenance and feature order.

## Residual risks

1. Few independent forecast origins, particularly for five-year models with consumption features.
2. Ecological associations cannot establish causal effects of prescribing or demographics.
3. Surveillance practice and breakpoint changes can cause dataset shift.
4. Some ESAC-Net antibiotic groups are broader than EARS-Net outcome groups.
5. No prospective 2025 target-year evaluation is yet available.
6. No independent external dataset under the same outcome definition has been evaluated.

Overall, the model should be described as research-stage internal development with enhanced temporal validation, not as an operational clinical or policy system.
