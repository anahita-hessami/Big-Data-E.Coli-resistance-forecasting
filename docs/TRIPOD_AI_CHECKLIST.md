# TRIPOD+AI reporting map

This project adapts the 27-item [TRIPOD+AI statement](https://www.bmj.com/content/385/bmj-2023-078378) to a country-level ecological surveillance forecast. TRIPOD+AI was written for clinical prediction research, so non-applicable individual-participant items are explicitly labelled rather than silently omitted.

| Reporting area | Project location or required action | Status |
|---|---|---|
| Title and abstract identify model development/evaluation | README and future manuscript abstract | Partial |
| Background and intended use | README; `docs/MODEL_CARD.md` | Complete |
| Target population and setting | EU/EEA EARS-Net country-level surveillance | Complete |
| Data sources and dates | `data/SOURCE_MANIFEST.csv` | Complete for current build |
| Eligibility criteria | Bloodstream/invasive *E. coli* validation rules in `src/regional_amr.py` | Complete |
| Outcome definition and timing | `docs/VALIDATION_PROTOCOL.md` | Complete |
| Predictor definitions and availability | Feature code and selected-feature tables | Complete |
| Sample size rationale | Describe surveillance coverage and feasible target years; no conventional patient-level calculation | Partial |
| Missing data | Median/mode imputation fitted inside training pipelines; missingness outputs | Complete |
| Data preprocessing | Source code and notebook | Complete |
| Model specification | Saved models, hyperparameters and metadata sidecars | Complete |
| Hyperparameter tuning | Temporal inner validation | Complete |
| Internal validation | Nested rolling-origin backtesting | Implemented |
| Performance measures | RMSE, MAE, weighted RMSE, skill and R² | Complete |
| Comparator | Persistence for every horizon | Complete |
| Uncertainty | Cluster bootstrap and prequential conformal intervals | Implemented |
| Model output availability | Joblib files plus JSON metadata | Complete |
| Full model presentation | Feature order and hyperparameters in sidecars; tree object in joblib | Complete |
| Calibration | Continuous observed-versus-predicted figures; interval coverage | Complete |
| Subgroup performance | Country and antibiotic outputs; expand in manuscript | Partial |
| Interpretation | README and research-grade result summary | Partial until final run |
| Limitations | README, model card and PROBAST+AI assessment | Complete |
| Registration/protocol | `docs/VALIDATION_PROTOCOL.md`; external preregistration not yet completed | Partial |
| Data/code availability | Repository-ready project; redistribution terms must be confirmed | Partial |
| Funding/conflicts | Must be completed by the author | Required in manuscript |
| Patient/public involvement | Not applicable to current coursework development; justify | Not applicable |
| Environmental/clinical workflow impact | Not yet evaluated | Future work |

The checklist improves reporting transparency; it does not itself establish low risk of bias or operational validity.
