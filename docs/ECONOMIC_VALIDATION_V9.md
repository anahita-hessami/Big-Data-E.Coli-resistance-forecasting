# Economic validation extension — Version 9

## What changed

### 1. Coverage-adjusted national incidence denominator

The economic model no longer uses the target-antibiotic `tested` count as if it
were a national population. For each country, it takes the largest reported 2023
invasive *E. coli* episode count across antibiotic groups, divides by ECDC's reported
national population coverage, calculates a 2023 incidence rate, and applies that
rate to Eurostat's official 2026 baseline population projection.

This is substantially more defensible, but it remains an ECDC-method surveillance
estimate rather than a complete national disease registry. Coverage and
representativeness are country-reported. The PSA therefore adds mean-preserving
lognormal measurement uncertainty: CV 0.10 for High, approximately 0.22 for Medium
and approximately 0.36 for Low representativeness. [ECDC explains the coverage
adjustment and its cautions](https://www.ecdc.europa.eu/sites/default/files/documents/antimicrobial-resistance-annual-epidemiological-report-EARS-Net-2023.pdf).

### 2. Intervention and effect size

The model intervention is molecular rapid diagnostic testing on a positive blood
culture plus active, real-time antimicrobial-stewardship review and treatment
recommendation. The comparator is conventional blood culture plus active
stewardship. The mortality odds ratio is 0.78 (95% CI 0.63–0.96) and is sampled on
the log scale. No generic 25% reduction in discordant treatment is assumed. [Peri et
al., 2024](https://pmc.ncbi.nlm.nih.gov/articles/PMC11327801/).

### 3. Correlated European uncertainty

Country forecast errors are generated as

$$e_i=\sqrt{\rho}Z_{common}+\sqrt{1-\rho}Z_i.$$

The base case uses $\rho=0.25$; structured analyses use 0, 0.25 and 0.50. In the
Python formula-validation run, the probability of cost-effectiveness versus
persistence remains about 13.9% in all three scenarios. This stability indicates
that, within the tested range, effect, threshold and activation assumptions matter
more than the common-error correlation. Only the R result may be used as the
authoritative headline.

### 4. CEA and BIA are separate

The CEA reports incremental cost, QALYs and net monetary benefit for 2026 and is
reported using CHEERS 2022/CHEERS-AI principles. The distinct BIA reports annual
programme spending for 2026–2030 in constant euros, including uptake, setup,
operating and variable costs. It contains no QALY, ICER or willingness-to-pay
decision rule. [CHEERS reporting guidance](https://www.equator-network.org/reporting-guidelines/cheers/)
and [ISPOR BIA guidance](https://www.ispor.org/heor-resources/good-practices/article/principles-of-good-practice-for-budget-impact-analysis-ii)
are complementary, not interchangeable.

### 5. One authoritative stochastic result

`R/run_economic_analysis.R` is authoritative. It uses 50,000 simulations and saves:

- the seed and R version;
- MD5 hashes for the forecast, EARS-Net, ECDC coverage, Eurostat population and
  parameter files;
- Monte Carlo standard errors and 95% confidence intervals for decision
  probabilities; and
- the three correlation scenarios.

`src/check_economic_consistency.py --require-r` fails if an output has the wrong
engine label, the wrong draw count, missing Monte Carlo uncertainty, stale hashes or
old headline percentages. The Python result is explicitly labelled non-authoritative.

## Historical decision-rule validation

Rolling-origin trigger validation uses the nested three-year forecasts for target
years 2021–2023. The pooled ML trigger has sensitivity 86.5%, specificity 93.8% and
false-negative rate 13.5%; persistence has sensitivity 78.4%, specificity 87.5% and
false-negative rate 21.6%. However, ML sensitivity is lower in the latest 2023
cutoff (66.7% versus 73.3%). The confusion matrix must therefore accompany economic
results.

## Remaining limitations

- The coverage-adjusted denominator is an estimate, not an independent national
  incidence registry.
- Programme costs, baseline mortality, QALY loss and uptake remain illustrative.
- The pooled clinical effect is not specific to every included country.
- Only three historical target years support decision-rule validation.
- These ecological country-level results must not guide individual treatment.
