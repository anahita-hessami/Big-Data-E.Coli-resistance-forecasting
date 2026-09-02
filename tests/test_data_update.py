from pathlib import Path
import sys
import tempfile

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.audit_data_updates import validate_current_esac_export  # noqa: E402


def test_esac_guard_rejects_new_year_append_only_file():
    frame = pd.DataFrame({
        "country": ["Alpha"], "iso3": ["AAA"], "year": [2024],
        "sector": ["community"], "antibiotic_family": ["J01C_penicillins"],
        "ddd_per_1000_per_day": [2.0], "source_url": ["https://official.example"],
    })
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "candidate.csv"
        frame.to_csv(path, index=False)
        try:
            validate_current_esac_export(path)
        except ValueError as error:
            assert "complete model-era history" in str(error)
        else:
            raise AssertionError("A partial new-year export must be rejected.")


if __name__ == "__main__":
    test_esac_guard_rejects_new_year_append_only_file()
    print("All data-update tests passed.")
