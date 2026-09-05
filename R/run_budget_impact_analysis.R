#!/usr/bin/env Rscript
root <- normalizePath(if(length(commandArgs(trailingOnly=TRUE))) commandArgs(trailingOnly=TRUE)[1] else ".")
source(file.path(root,"R/economic_model.R")); source(file.path(root,"R/budget_impact_model.R"))
parameters <- read_parameters(file.path(root,"economic_model/parameters/economic_parameters.csv"))
config <- read_model_config(file.path(root,"economic_model/model_config.csv"))
cohort <- prepare_economic_cohort(file.path(root,"outputs/notebook_ml/future_forecasts.csv"),
  file.path(root,"data/input/ears_net_ecoli_blood.csv"),file.path(root,"data/input/demographics.csv"),
  file.path(root,"data/input/ecdc_earsnet_coverage_2023.csv"),
  file.path(root,"data/input/eurostat_population_projection_2026.csv"),config,parameters)
country <- deterministic_cea(cohort,parameters)
bia_parameters <- read_bia_parameters(file.path(root,"economic_model/parameters/bia_parameters.csv"))
bia <- run_budget_impact(country,config,bia_parameters)
output <- file.path(root,"outputs/budget_impact_r"); dir.create(file.path(output,"figures"),recursive=TRUE,showWarnings=FALSE)
write.csv(bia,file.path(output,"annual_budget_impact.csv"),row.names=FALSE)
summary <- aggregate(cbind(AnnualBudgetImpactEUR,IncrementalBudgetVsPersistenceEUR)~Strategy,data=bia,sum)
write.csv(summary,file.path(output,"five_year_budget_summary.csv"),row.names=FALSE)
create_bia_figure(bia,file.path(output,"figures/five_year_budget_impact.png"))
cat("Separate R budget-impact analysis complete.\n")
