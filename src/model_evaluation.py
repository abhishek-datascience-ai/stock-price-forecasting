"""Generate aligned benchmark predictions and evaluate model artifacts."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
TEST_FRACTION = 0.20


@dataclass(frozen=True)
class PredictionSpec:
    name: str
    filename: str
    prediction_column: str


PREDICTION_SPECS = (
    PredictionSpec("Baseline", "baseline_predictions.csv", "Prediction"),
    PredictionSpec("ARIMA", "arima_predictions.csv", "ARIMA_Prediction"),
    PredictionSpec("Auto ARIMA", "auto_arima_predictions.csv", "Auto_ARIMA_Prediction"),
    PredictionSpec("LSTM", "lstm_predictions.csv", "LSTM_Prediction"),
    PredictionSpec("Prophet", "prophet_predictions.csv", "Prophet_Prediction"),
)


def load_close_prices() -> pd.DataFrame:
    source = PROCESSED_DIR / "tcs_stock_data_cleaned.csv"
    frame = pd.read_csv(source, parse_dates=["Date"])
    required = {"Date", "Close"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Missing source columns: {sorted(missing)}")

    frame = frame.loc[:, ["Date", "Close"]].dropna().sort_values("Date")
    if frame["Date"].duplicated().any():
        raise ValueError("Source data contains duplicate dates.")
    return frame.reset_index(drop=True)


def split_close_prices(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    split_index = int(len(frame) * (1 - TEST_FRACTION))
    if split_index < 2 or split_index >= len(frame):
        raise ValueError("Source data is too small for an 80/20 chronological split.")
    return frame.iloc[:split_index].copy(), frame.iloc[split_index:].copy()


def generate_baseline_predictions(train: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame:
    forecast = np.repeat(float(train["Close"].iloc[-1]), len(test))
    return pd.DataFrame(
        {
            "Date": test["Date"].dt.strftime("%Y-%m-%d"),
            "Actual": test["Close"].to_numpy(),
            "Prediction": forecast,
        }
    )


def generate_arima_predictions(train: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame:
    from statsmodels.tsa.arima.model import ARIMA

    model = ARIMA(train["Close"].to_numpy(dtype=float), order=(5, 1, 0))
    fitted_model = model.fit()
    forecast = np.asarray(fitted_model.forecast(steps=len(test)), dtype=float)

    return pd.DataFrame(
        {
            "Date": test["Date"].dt.strftime("%Y-%m-%d"),
            "Actual": test["Close"].to_numpy(),
            "ARIMA_Prediction": forecast,
        }
    )


def load_prediction(spec: PredictionSpec) -> pd.DataFrame:
    path = PROCESSED_DIR / spec.filename
    frame = pd.read_csv(path)
    if "Date" not in frame.columns and "Unnamed: 0" in frame.columns:
        frame = frame.rename(columns={"Unnamed: 0": "Date"})

    required = {"Date", "Actual", spec.prediction_column}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"{spec.filename} is missing columns: {sorted(missing)}")

    frame = frame.loc[:, ["Date", "Actual", spec.prediction_column]].copy()
    frame["Date"] = pd.to_datetime(frame["Date"], errors="coerce")
    frame["Actual"] = pd.to_numeric(frame["Actual"], errors="coerce")
    frame[spec.prediction_column] = pd.to_numeric(
        frame[spec.prediction_column], errors="coerce"
    )
    if frame.isna().any().any():
        raise ValueError(f"{spec.filename} contains missing or invalid evaluation values.")
    if frame["Date"].duplicated().any():
        raise ValueError(f"{spec.filename} contains duplicate dates.")
    return frame.sort_values("Date").reset_index(drop=True)


def load_aligned_predictions() -> dict[str, pd.DataFrame]:
    predictions = {spec.name: load_prediction(spec) for spec in PREDICTION_SPECS}
    reference = predictions[PREDICTION_SPECS[0].name]

    for spec in PREDICTION_SPECS[1:]:
        candidate = predictions[spec.name]
        if not reference["Date"].equals(candidate["Date"]):
            raise ValueError(f"{spec.filename} does not use the shared evaluation dates.")
        if not np.allclose(reference["Actual"], candidate["Actual"], rtol=0, atol=1e-6):
            raise ValueError(f"{spec.filename} does not use the shared actual values.")

    return predictions


def calculate_metrics(actual: pd.Series, predicted: pd.Series) -> dict[str, float]:
    errors = actual - predicted
    nonzero = actual.ne(0)
    if not nonzero.any():
        raise ValueError("MAPE cannot be calculated when every actual value is zero.")

    return {
        "MAE": float(errors.abs().mean()),
        "RMSE": float(np.sqrt(np.mean(np.square(errors)))),
        "MAPE (%)": float((errors[nonzero].abs() / actual[nonzero]).mean() * 100),
    }


def evaluate_predictions(predictions: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for spec in PREDICTION_SPECS:
        frame = predictions[spec.name]
        rows.append(
            {
                "Model": spec.name,
                **calculate_metrics(frame["Actual"], frame[spec.prediction_column]),
            }
        )
    return pd.DataFrame(rows)


def save_evaluation_summary(comparison: pd.DataFrame, reference: pd.DataFrame) -> None:
    comparison.to_csv(PROCESSED_DIR / "model_comparison.csv", index=False)
    summary = {
        "evaluation_rows": int(len(reference)),
        "start_date": reference["Date"].min().strftime("%Y-%m-%d"),
        "end_date": reference["Date"].max().strftime("%Y-%m-%d"),
        "split_method": "chronological 80/20",
        "forecast_protocol": "fixed-origin multi-step without test-period updates",
        "metrics": comparison.round(6).to_dict(orient="records"),
    }
    with (PROCESSED_DIR / "model_evaluation_summary.json").open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)
        file.write("\n")


def main() -> None:
    source = load_close_prices()
    train, test = split_close_prices(source)

    generate_baseline_predictions(train, test).to_csv(
        PROCESSED_DIR / "baseline_predictions.csv", index=False
    )
    generate_arima_predictions(train, test).to_csv(
        PROCESSED_DIR / "arima_predictions.csv", index=False
    )

    predictions = load_aligned_predictions()
    comparison = evaluate_predictions(predictions)
    save_evaluation_summary(comparison, predictions[PREDICTION_SPECS[0].name])
    print(comparison.to_string(index=False))


if __name__ == "__main__":
    main()
