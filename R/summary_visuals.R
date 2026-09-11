#!/usr/bin/env Rscript
# Additional result visualisations built on top of the authoritative PSA draws
# and the ICER/CEAC outputs. Dependency-free (base graphics only), consistent
# with R/economic_model.R. Run R/run_economic_analysis.R and R/ceac_analysis.R
# first so their outputs exist.

root <- normalizePath(if (length(commandArgs(trailingOnly = TRUE))) commandArgs(trailingOnly = TRUE)[1] else ".")
output <- file.path(root, "outputs/economic_r")
figures <- file.path(output, "figures")
dir.create(figures, recursive = TRUE, showWarnings = FALSE)

psa <- read.csv(file.path(output, "psa_results.csv"), stringsAsFactors = FALSE, check.names = FALSE)
icer <- read.csv(file.path(output, "icer_summary.csv"), stringsAsFactors = FALSE)
bia <- read.csv(file.path(root, "outputs/budget_impact_r/five_year_budget_summary.csv"), stringsAsFactors = FALSE)
psa_summary <- read.csv(file.path(output, "psa_summary.csv"), stringsAsFactors = FALSE)
metric_value <- function(name) psa_summary$Mean[psa_summary$Metric == name]

## ---- 08: NMB uncertainty distributions -------------------------------------
png(file.path(figures, "08_nmb_distributions.png"), 1500, 800, res = 150)
par(mfrow = c(1, 2), mar = c(5, 4, 4, 1))
h1 <- hist(psa$IncrementalNMBForecastVsUsualEUR / 1e6, breaks = 60, plot = FALSE)
plot(h1, col = "#2A9D8F", border = "white", main = "vs usual care",
     xlab = "Incremental NMB (million EUR)", ylab = "PSA draws")
abline(v = 0, lty = 2, lwd = 2)
h2 <- hist(psa$IncrementalNMBForecastVsPersistenceEUR / 1e6, breaks = 60, plot = FALSE)
plot(h2, col = "#E76F51", border = "white", main = "vs persistence-guided planning",
     xlab = "Incremental NMB (million EUR)", ylab = "PSA draws")
abline(v = 0, lty = 2, lwd = 2)
mtext("Distribution of incremental net monetary benefit across 50,000 PSA draws (WTP = EUR 30,000/QALY)",
      side = 3, line = -1.5, outer = TRUE, cex = 0.85)
dev.off()

## ---- 09: Annotated cost-effectiveness plane (vs persistence) ---------------
## The existing 02_cea_plane.png shows the vs-persistence comparison unlabelled;
## this version adds quadrant labels and the WTP threshold line, which is what
## makes a CE plane readable to someone outside health economics.
wtp <- 30000
png(file.path(figures, "09_ce_plane_annotated.png"), 1300, 1000, res = 150)
x <- psa$QALYGainedForecastVsPersistence
y <- psa$IncrementalCostForecastVsPersistenceEUR / 1e6
## Trim axis limits to the 0.5th-99.5th percentile of each PSA quantity so a
## handful of extreme tail draws (heavy-tailed by construction here, since
## costs/QALYs are summed across 30 countries with correlated forecast error)
## don't compress the WTP threshold line into a near-vertical segment.
xr <- quantile(x, c(0.005, 0.995)); yr <- quantile(y, c(0.005, 0.995))
xr <- xr + c(-1, 1) * diff(xr) * 0.08; yr <- yr + c(-1, 1) * diff(yr) * 0.08
plot(x, y, pch = 16, cex = 0.35, col = rgb(38/255, 70/255, 83/255, 0.15),
     xlim = xr, ylim = yr, xlab = "Incremental QALYs: ML forecast-guided vs persistence-guided",
     ylab = "Incremental cost (million EUR)",
     main = "Cost-effectiveness plane: ML forecast-guided vs persistence-guided",
     sub = "Axes trimmed to the 0.5th-99.5th percentile of PSA draws; extreme tail draws excluded from view only")
abline(h = 0, v = 0, lty = 1, col = "grey60")
abline(a = 0, b = wtp / 1e6, lty = 2, col = "#E76F51", lwd = 2)
legend("bottomleft", legend = sprintf("WTP threshold line (EUR %s/QALY)", format(wtp, big.mark = ",")),
       col = "#E76F51", lty = 2, lwd = 2, bty = "n", cex = 0.8)
text(xr[2]*0.5, yr[2]*0.8, "Costlier, more effective\n(compare to WTP line)", cex = 0.75, col = "grey30")
text(xr[1]*0.5, yr[2]*0.8, "Costlier, less effective\n(dominated)", cex = 0.75, col = "grey30")
text(xr[2]*0.5, yr[1]*0.8, "Cheaper, more effective\n(dominant)", cex = 0.75, col = "grey30")
text(xr[1]*0.5, yr[1]*0.8, "Cheaper, less effective\n(compare to WTP line)", cex = 0.75, col = "grey30")
dev.off()

## ---- 10: Headline results summary panel -------------------------------------
png(file.path(figures, "10_headline_summary.png"), 1500, 950, res = 150)
par(mar = c(0, 0, 2, 0))
plot(0, 0, type = "n", xlim = c(0, 10), ylim = c(0, 10), axes = FALSE, xlab = "", ylab = "",
     main = "Headline economic results: ML forecast-guided antimicrobial stewardship activation")

box_metric <- function(x, y, w, h, title, value, colour) {
  rect(x, y, x + w, y + h, col = colour, border = NA)
  text(x + w/2, y + h*0.62, value, cex = 1.5, font = 2, col = "white")
  text(x + w/2, y + h*0.22, title, cex = 0.75, col = "white")
}
box_metric(0.2, 5.6, 2.9, 3.6, "Deaths avoided/yr vs usual care\n(median, EU/EEA)",
           round(metric_value("DeathsAvoidedVsUsual")), "#264653")
box_metric(3.4, 5.6, 2.9, 3.6, "ICER vs usual care\n(EUR per QALY)",
           paste0("\u20ac", format(round(icer$ICEREURPerQALY[icer$Comparison == "ML forecast-guided vs usual care"]), big.mark=",")),
           "#2A9D8F")
box_metric(6.6, 5.6, 3.2, 3.6, "Probability cost-effective\nvs usual care (WTP \u20ac30k)",
           paste0(round(metric_value("Probability cost-effective vs usual care")*100, 1), "%"), "#2A9D8F")

box_metric(0.2, 1.2, 2.9, 3.6, "Probability cost-effective\nvs persistence-guided rule",
           paste0(round(metric_value("Probability cost-effective vs persistence planning")*100, 1), "%"), "#E76F51")
bia_diff <- bia$IncrementalBudgetVsPersistenceEUR[bia$Strategy == "ML forecast-guided planning"] / 1e6
box_metric(3.4, 1.2, 2.9, 3.6, "5-yr budget impact:\nML vs persistence-guided",
           paste0("\u20ac", format(round(abs(bia_diff), 1)), "M ", ifelse(bia_diff < 0, "saved", "added")),
           "#457B9D")
box_metric(6.6, 1.2, 3.2, 3.6, "PSA simulations\n(fixed seed, reproducible)",
           "50,000", "#457B9D")
dev.off()

cat("Additional visualisations written to outputs/economic_r/figures/.\n")
