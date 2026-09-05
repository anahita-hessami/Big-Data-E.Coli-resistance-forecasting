# R economic extension

`R/run_economic_analysis.R` runs the authoritative 50,000-draw CEA and writes to
`outputs/economic_r/`. `R/run_budget_impact_analysis.R` runs the separate 2026–2030
BIA and writes to `outputs/budget_impact_r/`.

The CEA compares ML-guided activation with persistence-guided activation and usual
care. The BIA is an affordability calculation only. Both are scenario analyses,
not policy recommendations, until jurisdiction-specific costs, mortality, QALY and
uptake data replace the clearly labelled illustrative inputs.

Python formula-validation outputs are regenerated with:

```bash
python -m src.economic_extension
```

They are stored in `outputs/economic_python_reference/` and are not authoritative.
