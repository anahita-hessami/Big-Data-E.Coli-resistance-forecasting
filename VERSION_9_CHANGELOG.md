# Version 9 change log

- Replaced the laboratory-volume economic population with an ECDC-method,
  coverage-adjusted national *E. coli* BSI incidence estimate.
- Added official Eurostat EUROPOP2023 baseline population for 2026.
- Added representativeness-dependent incidence uncertainty.
- Defined the intervention as molecular RDT plus active real-time stewardship and
  used the matched mortality odds ratio from Peri et al. (2024).
- Added common European forecast errors and ρ = 0, 0.25 and 0.50 scenarios.
- Separated the one-year CEA from the 2026–2030 BIA in code, inputs and reports.
- Made R the only authoritative stochastic engine; Python is formula validation.
- Increased the R PSA to 50,000 draws with Monte Carlo confidence intervals.
- Added R version, seed and input-file hashes to generated metadata.
- Added a consistency checker that rejects stale headline values or stale hashes.
- Added historical trigger confusion matrices using all three nested cutoffs.
- Fixed the parameter-driven trigger reference line and unclipped tornado labels.
- Added GitHub Actions generation and upload of authoritative R outputs.

The intervention cost, baseline mortality, QALY loss and BIA uptake remain
illustrative. This is a research scenario, not a policy recommendation.
