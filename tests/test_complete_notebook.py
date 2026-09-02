"""Structural tests for the user-facing complete ML notebook."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks" / "Regional_Ecoli_BSI_AMR_Complete_ML.ipynb"


def test_complete_notebook_contains_executable_ml_and_forecasting() -> None:
    notebook = json.loads(NOTEBOOK.read_text())
    source = "\n".join(
        "".join(cell.get("source", []))
        for cell in notebook["cells"]
        if cell.get("cell_type") == "code"
    )
    assert len(notebook["cells"]) >= 50
    assert "RUN_MODEL_TRAINING = True" in source
    assert "GridSearchCV(" in source
    assert "search.fit(" in source
    assert "fitted_full_models" in source
    assert "future_forecasts.to_csv" in source
    assert "joblib.dump(" in source


if __name__ == "__main__":
    test_complete_notebook_contains_executable_ml_and_forecasting()
    print("Complete notebook test passed.")
