#!/usr/bin/env Rscript
source("R/economic_model.R"); source("R/budget_impact_model.R")
parameters <- read_parameters("economic_model/parameters/economic_parameters.csv")
config <- read_model_config("economic_model/model_config.csv")
cohort <- prepare_economic_cohort("outputs/notebook_ml/future_forecasts.csv",
  "data/input/ears_net_ecoli_blood.csv","data/input/demographics.csv",
  "data/input/ecdc_earsnet_coverage_2023.csv","data/input/eurostat_population_projection_2026.csv",
  config,parameters)
stopifnot(nrow(cohort)==28L,all(cohort$population_coverage_pct>0),
  all(cohort$IncidenceMeasurementCV>=.10),all(cohort$ProjectedPopulationTargetYear>0))
country <- deterministic_cea(cohort,parameters)
stopifnot(all(country$ForecastStrategyCostEUR>=0),all(is.finite(country$IncrementalNMBForecastVsPersistenceEUR)))
small <- parameters; small$base[small$parameter=="psa_simulations"] <- 50
psa <- run_psa(cohort,small); stopifnot(nrow(psa)==50L)
summary <- summarise_psa(psa); probability <- grepl("Probability",summary$Metric)
stopifnot(all(is.finite(summary$MonteCarloSE[probability])))
bia <- run_budget_impact(country,config,read_bia_parameters("economic_model/parameters/bia_parameters.csv"))
stopifnot(nrow(bia)==10L,all(bia$Year%in%2026:2030),
  !any(c("QALY","ICER","NMB")%in%names(bia)))
cat("All R economic tests passed.\n")
