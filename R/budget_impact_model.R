read_bia_parameters <- function(path) read.csv(path, stringsAsFactors=FALSE, check.names=FALSE)
bia_value <- function(parameters,name) as.numeric(parameters$base[parameters$parameter==name])

run_budget_impact <- function(country, config, parameters) {
  years <- seq(as.integer(config$bia_start_year),as.integer(config$bia_end_year))
  uptake <- seq(bia_value(parameters,"uptake_year_1"),bia_value(parameters,"uptake_year_5"),length.out=length(years))
  rows <- list(); k <- 1L
  strategies <- list("Persistence-guided planning"=country$PersistenceProgrammeTriggered,
                     "ML forecast-guided planning"=country$ForecastProgrammeTriggered)
  for (strategy in names(strategies)) for (i in seq_along(years)) {
    active <- strategies[[strategy]]
    episodes <- country$EconomicCohortEpisodes*(1+bia_value(parameters,"annual_bsi_incidence_growth"))^(i-1)
    eligible <- sum(episodes[active]); treated <- eligible*uptake[i]
    price <- (1+bia_value(parameters,"price_growth"))^(i-1)
    setup <- if(i==1) sum(active)*bia_value(parameters,"one_time_setup_cost_per_country")*price else 0
    operating <- sum(active)*bia_value(parameters,"annual_operating_cost_per_country")*price
    variable <- treated*bia_value(parameters,"variable_cost_per_treated_episode")*price
    rows[[k]] <- data.frame(Strategy=strategy,Year=years[i],ActivatedCountries=sum(active),
      EligibleEpisodes=eligible,Uptake=uptake[i],TreatedEpisodes=treated,SetupCostEUR=setup,
      OperatingCostEUR=operating,VariableCostEUR=variable,AnnualBudgetImpactEUR=setup+operating+variable)
    k <- k+1L
  }
  out <- do.call(rbind,rows)
  comparator <- out[out$Strategy=="Persistence-guided planning",c("Year","AnnualBudgetImpactEUR")]
  names(comparator)[2] <- "PersistenceBudgetEUR"
  out <- merge(out,comparator,by="Year",all.x=TRUE,sort=FALSE)
  out$IncrementalBudgetVsPersistenceEUR <- ifelse(out$Strategy=="ML forecast-guided planning",
    out$AnnualBudgetImpactEUR-out$PersistenceBudgetEUR,0)
  out[order(out$Strategy,out$Year),]
}

create_bia_figure <- function(bia,path) {
  strategies <- unique(bia$Strategy); years <- sort(unique(bia$Year))
  values <- sapply(strategies,function(x) bia$AnnualBudgetImpactEUR[bia$Strategy==x]/1e6)
  png(path,1200,800,res=150); matplot(years,values,type="o",pch=19,lwd=3,lty=1,
    col=c("#6C757D","#2A9D8F"),xlab="Budget year",ylab="Annual budget impact (million EUR)",
    main="Five-year affordability of alternative activation rules")
  legend("topleft",legend=strategies,col=c("#6C757D","#2A9D8F"),lwd=3,pch=19,bty="n"); dev.off()
}
