from pathlib import Path
from zipfile import is_zipfile

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"


def test_streamlit_input_files_exist_and_are_populated() -> None:
    filenames = (
        "baseline_predictions.csv",
        "arima_predictions.csv",
        "auto_arima_predictions.csv",
        "lstm_predictions.csv",
        "prophet_predictions.csv",
        "model_comparison.csv",
    )
    for filename in filenames:
        path = PROCESSED_DIR / filename
        assert path.is_file(), f"Missing Streamlit input: {filename}"
        assert path.stat().st_size > 0, f"Empty Streamlit input: {filename}"


def test_cleaned_stock_data_is_valid() -> None:
    frame = pd.read_csv(PROCESSED_DIR / "tcs_stock_data_cleaned.csv")
    expected_columns = {"Date", "Open", "High", "Low", "Close", "Volume"}

    assert expected_columns.issubset(frame.columns)
    assert not frame.empty
    assert not frame[list(expected_columns)].isna().any().any()
    assert not frame["Date"].duplicated().any()


def test_saved_keras_model_is_present() -> None:
    model_path = PROJECT_ROOT / "models" / "lstm_model.keras"
    assert model_path.is_file()
    assert is_zipfile(model_path)
