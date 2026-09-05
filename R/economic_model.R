# Dependency-free health-economic model. R is the authoritative implementation.

read_parameters <- function(path) read.csv(path, stringsAsFactors = FALSE, check.names = FALSE)

parameter_value <- function(parameters, name, column = "base") {
  row <- parameters[parameters$parameter == name, , drop = FALSE]
  if (nrow(row) != 1L) stop("Expected one parameter: ", name)
  value <- suppressWarnings(as.numeric(row[[column]]))
  if (!is.finite(value)) stop("Non-numeric parameter: ", name)
  value
}

read_model_config <- function(path) {
  x <- read.csv(path, stringsAsFactors = FALSE)
  setNames(as.list(x$value), x$setting)
}

prepare_economic_cohort <- function(
  forecast_path, ears_path, demographics_path, coverage_path, population_path,
  config, parameters
) {
  forecasts <- read.csv(forecast_path, stringsAsFactors = FALSE, check.names = FALSE)
  horizon <- as.integer(config$forecast_horizon_years)
  target_year <- as.integer(config$forecast_target_year)
  origin_year <- as.integer(config$surveillance_reference_year)
  cohort <- forecasts[
    forecasts$Horizon == horizon & forecasts$ForecastTargetYear == target_year &
      forecasts$antibiotic == config$antibiotic, , drop = FALSE
  ]
  if (nrow(cohort) == 0L || anyDuplicated(cohort$iso3)) stop("Invalid forecast cohort.")

  ears <- read.csv(ears_path, stringsAsFactors = FALSE, check.names = FALSE)
  origin <- ears[ears$year == origin_year, c("iso3", "tested"), drop = FALSE]
  reported <- aggregate(tested ~ iso3, data = origin, FUN = max)
  names(reported)[2] <- "ReportedEcoliBSIEpisodesOrigin"
  coverage <- read.csv(coverage_path, stringsAsFactors = FALSE, check.names = FALSE)
  demographics <- read.csv(demographics_path, stringsAsFactors = FALSE, check.names = FALSE)
  origin_population <- demographics[demographics$year == origin_year, c("iso3", "population")]
  names(origin_population)[2] <- "PopulationOrigin"
  target_population <- read.csv(population_path, stringsAsFactors = FALSE, check.names = FALSE)
  target_population <- target_population[target_population$year == target_year, c("iso3", "population")]
  names(target_population)[2] <- "ProjectedPopulationTargetYear"
  cohort <- merge(cohort, reported, by = "iso3", all.x = TRUE, sort = FALSE)
  cohort <- merge(cohort, coverage, by = c("iso3", "country"), all.x = TRUE, sort = FALSE)
  cohort <- merge(cohort, origin_population, by = "iso3", all.x = TRUE, sort = FALSE)
  cohort <- merge(cohort, target_population, by = "iso3", all.x = TRUE, sort = FALSE)
  valid <- is.finite(cohort$population_coverage_pct) & cohort$population_coverage_pct > 0 &
    is.finite(cohort$ReportedEcoliBSIEpisodesOrigin) & cohort$ReportedEcoliBSIEpisodesOrigin > 0 &
    is.finite(cohort$PopulationOrigin) & cohort$PopulationOrigin > 0 &
    is.finite(cohort$ProjectedPopulationTargetYear) & cohort$ProjectedPopulationTargetYear > 0
  cohort <- cohort[valid, , drop = FALSE]
  if (nrow(cohort) == 0L) stop("No country has a valid coverage-adjusted denominator.")

  cohort$CoverageAdjustedEcoliBSIEpisodesOrigin <-
    cohort$ReportedEcoliBSIEpisodesOrigin / (cohort$population_coverage_pct / 100)
  cohort$TotalEcoliBSIIncidencePer100kOrigin <-
    cohort$CoverageAdjustedEcoliBSIEpisodesOrigin / cohort$PopulationOrigin * 100000
  cohort$EconomicCohortEpisodes <- cohort$TotalEcoliBSIIncidencePer100kOrigin *
    cohort$ProjectedPopulationTargetYear / 100000 *
    parameter_value(parameters, "future_bsi_incidence_multiplier")

  rep_columns <- c("geographical_representativeness", "hospital_representativeness", "isolate_representativeness")
  any_low <- apply(cohort[rep_columns] == "Low", 1, any)
  any_medium <- apply(cohort[rep_columns] == "Medium", 1, any) & !any_low
  cohort$RepresentativenessCaution <- ifelse(any_low, "Low", ifelse(any_medium, "Medium", "High"))
  cohort$IncidenceMeasurementCV <- sqrt(
    parameter_value(parameters, "coverage_cv_no_caution")^2 +
      any_medium * parameter_value(parameters, "coverage_cv_medium_representation")^2 +
      any_low * parameter_value(parameters, "coverage_cv_low_representation")^2
  )
  cohort$ForecastIncreasePP <- cohort$RecommendedForecastPct - cohort$LatestObservedResistancePct
  adequate <- cohort$tested >= parameter_value(parameters, "minimum_reported_tested")
  cohort$PersistenceProgrammeTriggered <- adequate &
    cohort$PersistenceForecastPct >= parameter_value(parameters, "resistance_trigger_pct")
  cohort$ForecastProgrammeTriggered <- adequate & (
    cohort$RecommendedForecastPct >= parameter_value(parameters, "resistance_trigger_pct") |
      cohort$ForecastIncreasePP >= parameter_value(parameters, "increase_trigger_pp")
  )
  cohort[order(cohort$RecommendedForecastPct, decreasing = TRUE), , drop = FALSE]
}

deterministic_cea <- function(cohort, parameters) {
  out <- cohort
  out$ForecastResistantEpisodes <- out$EconomicCohortEpisodes * out$RecommendedForecastPct / 100
  p0 <- parameter_value(parameters, "baseline_bsi_mortality_probability")
  odds_ratio <- parameter_value(parameters, "intervention_mortality_odds_ratio")
  p1 <- odds_ratio * p0 / (1 - p0 + odds_ratio * p0)
  out$UsualCareDeaths <- out$ForecastResistantEpisodes * p0
  out$PersistenceDeaths <- ifelse(out$PersistenceProgrammeTriggered,
                                  out$ForecastResistantEpisodes * p1, out$UsualCareDeaths)
  out$ForecastDeaths <- ifelse(out$ForecastProgrammeTriggered,
                               out$ForecastResistantEpisodes * p1, out$UsualCareDeaths)
  programme_cost <- parameter_value(parameters, "programme_fixed_cost_per_country") +
    parameter_value(parameters, "programme_variable_cost_per_bsi_episode") * out$EconomicCohortEpisodes
  out$PersistenceStrategyCostEUR <- ifelse(out$PersistenceProgrammeTriggered, programme_cost, 0)
  out$ForecastStrategyCostEUR <- ifelse(out$ForecastProgrammeTriggered, programme_cost, 0)
  out$IncrementalCostForecastVsUsualEUR <- out$ForecastStrategyCostEUR
  out$IncrementalCostForecastVsPersistenceEUR <- out$ForecastStrategyCostEUR - out$PersistenceStrategyCostEUR
  qaly <- parameter_value(parameters, "qaly_loss_per_bsi_death")
  out$QALYGainedForecastVsUsual <- (out$UsualCareDeaths - out$ForecastDeaths) * qaly
  out$QALYGainedForecastVsPersistence <- (out$PersistenceDeaths - out$ForecastDeaths) * qaly
  wtp <- parameter_value(parameters, "willingness_to_pay_per_qaly")
  out$IncrementalNMBForecastVsUsualEUR <- wtp * out$QALYGainedForecastVsUsual - out$IncrementalCostForecastVsUsualEUR
  out$IncrementalNMBForecastVsPersistenceEUR <- wtp * out$QALYGainedForecastVsPersistence - out$IncrementalCostForecastVsPersistenceEUR
  out
}

rtriangular <- function(n, low, mode, high) {
  if (low == high) return(rep(low, n))
  u <- runif(n); cut <- (mode - low) / (high - low)
  ifelse(u < cut, low + sqrt(u * (high-low) * (mode-low)),
         high - sqrt((1-u) * (high-low) * (high-mode)))
}

sample_parameter <- function(parameters, name, n) {
  row <- parameters[parameters$parameter == name, , drop = FALSE]
  low <- as.numeric(row$low); base <- as.numeric(row$base); high <- as.numeric(row$high)
  if (row$distribution == "fixed") return(rep(base, n))
  if (row$distribution == "triangular") return(rtriangular(n, low, base, high))
  if (row$distribution == "lognormal_odds_ratio") {
    sigma <- (log(high) - log(low)) / (2 * qnorm(.975))
    return(rlnorm(n, log(base), sigma))
  }
  stop("Unsupported distribution: ", row$distribution)
}

run_psa <- function(cohort, parameters, correlation_override = NULL) {
  n <- as.integer(parameter_value(parameters, "psa_simulations"))
  set.seed(as.integer(parameter_value(parameters, "random_seed")))
  rho <- if (is.null(correlation_override)) parameter_value(parameters, "cross_country_forecast_correlation") else as.numeric(correlation_override)
  if (!is.finite(rho) || rho < 0 || rho > 1) stop("Correlation outside [0,1].")
  names_to_sample <- c("resistance_trigger_pct", "increase_trigger_pp", "minimum_reported_tested",
    "future_bsi_incidence_multiplier", "baseline_bsi_mortality_probability",
    "intervention_mortality_odds_ratio", "programme_fixed_cost_per_country",
    "programme_variable_cost_per_bsi_episode", "qaly_loss_per_bsi_death",
    "willingness_to_pay_per_qaly")
  d <- lapply(names_to_sample, function(x) sample_parameter(parameters, x, n)); names(d) <- names_to_sample
  forecast_sd <- pmax((cohort$Upper80Pct - cohort$Lower80Pct) / (2*qnorm(.90)), .01)
  output <- matrix(NA_real_, nrow = n, ncol = 10)
  colnames(output) <- c("PersistenceTriggeredCountries", "ForecastTriggeredCountries", "DeathsAvoidedVsUsual",
    "IncrementalCostForecastVsUsualEUR", "QALYGainedForecastVsUsual",
    "IncrementalCostForecastVsPersistenceEUR", "QALYGainedForecastVsPersistence",
    "IncrementalNMBForecastVsUsualEUR", "IncrementalNMBForecastVsPersistenceEUR", "CostEffectiveVsPersistence")
  incidence_sigma <- sqrt(log1p(cohort$IncidenceMeasurementCV^2))
  for (i in seq_len(n)) {
    error <- sqrt(rho)*rnorm(1) + sqrt(1-rho)*rnorm(nrow(cohort))
    forecast <- pmin(100, pmax(0, cohort$RecommendedForecastPct + forecast_sd*error))
    incidence_factor <- rlnorm(nrow(cohort), -.5*incidence_sigma^2, incidence_sigma)
    economic <- cohort$EconomicCohortEpisodes * d$future_bsi_incidence_multiplier[i] * incidence_factor
    adequate <- cohort$tested >= d$minimum_reported_tested[i]
    persistence <- adequate & cohort$PersistenceForecastPct >= d$resistance_trigger_pct[i]
    ml <- adequate & (cohort$RecommendedForecastPct >= d$resistance_trigger_pct[i] |
                        cohort$ForecastIncreasePP >= d$increase_trigger_pp[i])
    resistant <- economic * forecast/100
    p0 <- d$baseline_bsi_mortality_probability[i]
    odds_ratio <- d$intervention_mortality_odds_ratio[i]
    p1 <- odds_ratio*p0/(1-p0+odds_ratio*p0)
    usual_deaths <- resistant*p0
    persistence_deaths <- ifelse(persistence, resistant*p1, usual_deaths)
    ml_deaths <- ifelse(ml, resistant*p1, usual_deaths)
    persistence_cost <- sum(ifelse(persistence, d$programme_fixed_cost_per_country[i] + d$programme_variable_cost_per_bsi_episode[i]*economic, 0))
    ml_cost <- sum(ifelse(ml, d$programme_fixed_cost_per_country[i] + d$programme_variable_cost_per_bsi_episode[i]*economic, 0))
    cost_usual <- ml_cost; cost_persistence <- ml_cost - persistence_cost
    qaly_usual <- sum(usual_deaths-ml_deaths)*d$qaly_loss_per_bsi_death[i]
    qaly_persistence <- sum(persistence_deaths-ml_deaths)*d$qaly_loss_per_bsi_death[i]
    nmb_usual <- d$willingness_to_pay_per_qaly[i]*qaly_usual-cost_usual
    nmb_persistence <- d$willingness_to_pay_per_qaly[i]*qaly_persistence-cost_persistence
    output[i,] <- c(sum(persistence), sum(ml), sum(usual_deaths-ml_deaths), cost_usual,
      qaly_usual, cost_persistence, qaly_persistence, nmb_usual, nmb_persistence,
      as.numeric(nmb_persistence > 0))
  }
  out <- as.data.frame(output)
  out$Simulation <- seq_len(n)
  out$CostEffectiveVsUsual <- out$IncrementalNMBForecastVsUsualEUR > 0
  out$CostEffectiveVsPersistence <- out$CostEffectiveVsPersistence > 0
  out
}

summarise_psa <- function(psa) {
  variables <- c("DeathsAvoidedVsUsual", "IncrementalCostForecastVsUsualEUR",
    "IncrementalCostForecastVsPersistenceEUR", "QALYGainedForecastVsUsual",
    "QALYGainedForecastVsPersistence", "IncrementalNMBForecastVsUsualEUR",
    "IncrementalNMBForecastVsPersistenceEUR")
  rows <- lapply(variables, function(v) data.frame(Metric=v, Mean=mean(psa[[v]]),
    Median=median(psa[[v]]), Lower95=quantile(psa[[v]],.025), Upper95=quantile(psa[[v]],.975),
    MonteCarloSE=NA_real_, MonteCarloLower95=NA_real_, MonteCarloUpper95=NA_real_))
  for (entry in list(c("Probability cost-effective vs usual care","CostEffectiveVsUsual"),
                     c("Probability cost-effective vs persistence planning","CostEffectiveVsPersistence"))) {
    p <- mean(psa[[entry[2]]]); se <- sqrt(p*(1-p)/nrow(psa))
    rows[[length(rows)+1]] <- data.frame(Metric=entry[1], Mean=p, Median=NA, Lower95=NA, Upper95=NA,
      MonteCarloSE=se, MonteCarloLower95=max(0,p-qnorm(.975)*se), MonteCarloUpper95=min(1,p+qnorm(.975)*se))
  }
  do.call(rbind, rows)
}

run_correlation_sensitivity <- function(cohort, parameters) {
  do.call(rbind, lapply(c(0,.25,.50), function(rho) {
    psa <- run_psa(cohort, parameters, rho); p <- mean(psa$CostEffectiveVsPersistence)
    se <- sqrt(p*(1-p)/nrow(psa)); x <- psa$IncrementalNMBForecastVsPersistenceEUR
    data.frame(CrossCountryCorrelation=rho, Simulations=nrow(psa),
      MeanIncrementalNMBVsPersistenceEUR=mean(x), Lower95IncrementalNMBVsPersistenceEUR=quantile(x,.025),
      Upper95IncrementalNMBVsPersistenceEUR=quantile(x,.975), ProbabilityCostEffectiveVsPersistence=p,
      MonteCarloSE=se, MonteCarloLower95=max(0,p-qnorm(.975)*se), MonteCarloUpper95=min(1,p+qnorm(.975)*se))
  }))
}

one_way_sensitivity <- function(cohort, parameters) {
  base_nmb <- sum(deterministic_cea(cohort,parameters)$IncrementalNMBForecastVsPersistenceEUR)
  uncertain <- parameters[parameters$distribution!="fixed" & parameters$low!=parameters$high,]
  rows <- lapply(seq_len(nrow(uncertain)),function(i) {
    name <- uncertain$parameter[i]
    calculate <- function(bound) {
      scenario <- parameters
      scenario$base[scenario$parameter==name] <- uncertain[[bound]][i]
      current <- cohort
      current$EconomicCohortEpisodes <- current$EconomicCohortEpisodes /
        parameter_value(parameters,"future_bsi_incidence_multiplier") *
        parameter_value(scenario,"future_bsi_incidence_multiplier")
      adequate <- current$tested>=parameter_value(scenario,"minimum_reported_tested")
      current$PersistenceProgrammeTriggered <- adequate & current$PersistenceForecastPct>=parameter_value(scenario,"resistance_trigger_pct")
      current$ForecastProgrammeTriggered <- adequate & (current$RecommendedForecastPct>=parameter_value(scenario,"resistance_trigger_pct") |
        current$ForecastIncreasePP>=parameter_value(scenario,"increase_trigger_pp"))
      sum(deterministic_cea(current,scenario)$IncrementalNMBForecastVsPersistenceEUR)
    }
    low <- calculate("low"); high <- calculate("high")
    bounds <- c(base_nmb,low,high)
    data.frame(Parameter=name,BaseNMBEUR=base_nmb,LowNMBEUR=low,HighNMBEUR=high,
      MinimumNMBEUR=min(bounds),MaximumNMBEUR=max(bounds),NMBRangeEUR=max(bounds)-min(bounds))
  })
  do.call(rbind,rows)
}

parameter_display_label <- function(x) {
  labels <- c(resistance_trigger_pct="Resistance trigger",increase_trigger_pp="Forecast-increase trigger",
    future_bsi_incidence_multiplier="Future BSI incidence",baseline_bsi_mortality_probability="Baseline BSI mortality",
    intervention_mortality_odds_ratio="RDT + ASP mortality odds ratio",programme_fixed_cost_per_country="Programme fixed cost",
    programme_variable_cost_per_bsi_episode="Variable cost per BSI episode",qaly_loss_per_bsi_death="QALY loss per BSI death",
    willingness_to_pay_per_qaly="Willingness to pay per QALY")
  result <- unname(labels[x]); result[is.na(result)] <- x[is.na(result)]; result
}

create_cea_figures <- function(country, psa, correlation, one_way, output_directory, parameters) {
  dir.create(output_directory, recursive=TRUE, showWarnings=FALSE)
  ordered <- country[order(country$RecommendedForecastPct),]; y <- seq_len(nrow(ordered))
  png(file.path(output_directory,"01_forecast_trigger_profile.png"),1400,1000,res=150)
  par(mar=c(5,11,4,2)); plot(ordered$RecommendedForecastPct,y,yaxt="n",pch=19,
    xlim=range(c(ordered$Lower80Pct,ordered$Upper80Pct)),xlab="Forecast resistance (%)",ylab="",
    main="2026 forecast and activation rule")
  segments(ordered$Lower80Pct,y,ordered$Upper80Pct,y,col="grey55",lwd=2)
  axis(2,at=y,labels=ordered$country,las=1,cex.axis=.8)
  abline(v=parameter_value(parameters,"resistance_trigger_pct"),lty=2,col="#E76F51"); dev.off()
  png(file.path(output_directory,"02_cea_plane.png"),1200,800,res=150)
  plot(psa$QALYGainedForecastVsPersistence,psa$IncrementalCostForecastVsPersistenceEUR/1e6,
    pch=16,cex=.35,col=rgb(38/255,70/255,83/255,.2),xlab="QALYs gained: ML vs persistence",
    ylab="Incremental cost (million EUR)",main="Authoritative R probabilistic CEA")
  abline(h=0,v=0,lty=2); dev.off()
  png(file.path(output_directory,"03_correlation_sensitivity.png"),1200,800,res=150)
  ylimits <- c(max(0,min(correlation$MonteCarloLower95)-.01),min(1,max(correlation$MonteCarloUpper95)+.01))
  plot(correlation$CrossCountryCorrelation,correlation$ProbabilityCostEffectiveVsPersistence,type="o",pch=19,lwd=3,
    ylim=ylimits,xlab="Cross-country forecast-error correlation",ylab="Probability cost-effective",
    main="Correlation sensitivity"); arrows(correlation$CrossCountryCorrelation,correlation$MonteCarloLower95,
    correlation$CrossCountryCorrelation,correlation$MonteCarloUpper95,angle=90,code=3,length=.05); dev.off()
  tornado <- one_way[order(one_way$NMBRangeEUR),]; y <- seq_len(nrow(tornado))
  png(file.path(output_directory,"06_one_way_sensitivity_tornado.png"),1500,900,res=150)
  par(mar=c(5,16,4,2)); plot(tornado$BaseNMBEUR/1e6,y,type="n",yaxt="n",
    xlim=range(c(tornado$MinimumNMBEUR,tornado$MaximumNMBEUR))/1e6,
    xlab="Incremental NMB: ML vs persistence (million EUR)",ylab="",main="One-way sensitivity analysis")
  segments(tornado$MinimumNMBEUR/1e6,y,tornado$MaximumNMBEUR/1e6,y,lwd=8,col="#457B9D")
  points(tornado$BaseNMBEUR/1e6,y,pch=18,col="#E76F51")
  axis(2,at=y,labels=parameter_display_label(tornado$Parameter),las=1,cex.axis=.8)
  abline(v=0,lty=2); dev.off()
}
