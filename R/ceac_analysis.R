#!/usr/bin/env Rscript
# Adds two standard HEOR deliverables that the base pipeline computes NMB for
# but does not report explicitly: base-case ICERs, and a cost-effectiveness
# acceptability curve (CEAC) across a grid of willingness-to-pay thresholds.
# Reads the authoritative PSA draws written by R/run_economic_analysis.R;
# run that script first (or via run_economic_analysis.R) so psa_results.csv exists.

root <- normalizePath(if (length(commandArgs(trailingOnly = TRUE))) commandArgs(trailingOnly = TRUE)[1] else ".")
psa_path <- file.path(root, "outputs/economic_r/psa_results.csv")
if (!file.exists(psa_path)) {
  stop("outputs/economic_r/psa_results.csv not found. Run R/run_economic_analysis.R first.")
}
psa <- read.csv(psa_path, stringsAsFactors = FALSE, check.names = FALSE)

output <- file.path(root, "outputs/economic_r")
figures <- file.path(output, "figures")
dir.create(figures, recursive = TRUE, showWarnings = FALSE)

## ---- Base-case ICER (mean incremental cost / mean incremental QALY) -------
## Reported per the Second Panel on Cost-Effectiveness convention: if the
## intervention costs less AND gains QALYs it dominates (ICER not meaningful
## as a ratio to compare against a WTP threshold); if it costs more and gains
## fewer QALYs it is dominated. Only compute a ratio in the two ambiguous
## quadrants (more costly + more effective, or less costly + less effective).
icer_row <- function(cost, qaly, label) {
  mean_cost <- mean(cost); mean_qaly <- mean(qaly)
  if (mean_cost <= 0 && mean_qaly >= 0) {
    verdict <- "Dominant (cheaper and more effective)"
    icer <- NA_real_
  } else if (mean_cost >= 0 && mean_qaly <= 0) {
    verdict <- "Dominated (costlier and less effective)"
    icer <- NA_real_
  } else {
    icer <- mean_cost / mean_qaly
    verdict <- if (icer >= 0) "Trade-off: compare to willingness-to-pay threshold" else "Sign-crossing ratio: interpret with the CEAC, not the ICER alone"
  }
  data.frame(
    Comparison = label,
    MeanIncrementalCostEUR = mean_cost,
    MeanIncrementalQALY = mean_qaly,
    ICEREURPerQALY = icer,
    Verdict = verdict
  )
}

icer_table <- rbind(
  icer_row(psa$IncrementalCostForecastVsUsualEUR, psa$QALYGainedForecastVsUsual, "ML forecast-guided vs usual care"),
  icer_row(psa$IncrementalCostForecastVsPersistenceEUR, psa$QALYGainedForecastVsPersistence, "ML forecast-guided vs persistence-guided")
)
write.csv(icer_table, file.path(output, "icer_summary.csv"), row.names = FALSE)

## ---- Cost-effectiveness acceptability curve (CEAC) -------------------------
## For each WTP value on the grid, the proportion of PSA draws where the
## intervention's net monetary benefit is positive at that threshold.
wtp_grid <- seq(0, 100000, by = 2500)

ceac_point <- function(cost, qaly, wtp) mean(wtp * qaly - cost > 0)

ceac <- data.frame(
  WillingnessToPayEURPerQALY = wtp_grid,
  ProbabilityCostEffectiveVsUsual = sapply(wtp_grid, function(w) ceac_point(psa$IncrementalCostForecastVsUsualEUR, psa$QALYGainedForecastVsUsual, w)),
  ProbabilityCostEffectiveVsPersistence = sapply(wtp_grid, function(w) ceac_point(psa$IncrementalCostForecastVsPersistenceEUR, psa$QALYGainedForecastVsPersistence, w))
)
write.csv(ceac, file.path(output, "ceac.csv"), row.names = FALSE)

png(file.path(figures, "07_ceac.png"), 1300, 850, res = 150)
plot(ceac$WillingnessToPayEURPerQALY, ceac$ProbabilityCostEffectiveVsUsual, type = "l", lwd = 3,
     col = "#2A9D8F", ylim = c(0, 1), xlab = "Willingness to pay (EUR per QALY)",
     ylab = "Probability cost-effective", main = "Cost-effectiveness acceptability curve")
lines(ceac$WillingnessToPayEURPerQALY, ceac$ProbabilityCostEffectiveVsPersistence, lwd = 3, col = "#E76F51")
abline(v = 30000, lty = 2, col = "grey40")
legend("bottomright", legend = c("vs usual care", "vs persistence-guided planning", "Base-case WTP (EUR 30,000)"),
       col = c("#2A9D8F", "#E76F51", "grey40"), lwd = c(3, 3, 1), lty = c(1, 1, 2), bty = "n", cex = 0.85)
dev.off()

cat("ICER and CEAC outputs written to outputs/economic_r/.\n")
print(icer_table)
