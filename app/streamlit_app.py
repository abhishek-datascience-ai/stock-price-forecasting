from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

MODEL_FILES = {
    "Baseline": PROCESSED_DIR / "baseline_predictions.csv",
    "ARIMA": PROCESSED_DIR / "arima_predictions.csv",
    "Auto ARIMA": PROCESSED_DIR / "auto_arima_predictions.csv",
    "LSTM": PROCESSED_DIR / "lstm_predictions.csv",
    "Prophet": PROCESSED_DIR / "prophet_predictions.csv",
}

MODEL_PREDICTION_COLUMNS = {
    "Baseline": "Prediction",
    "ARIMA": "ARIMA_Prediction",
    "Auto ARIMA": "Auto_ARIMA_Prediction",
    "LSTM": "LSTM_Prediction",
    "Prophet": "Prophet_Prediction",
}


@st.cache_data
def load_csv(file_path: Path) -> pd.DataFrame:
    if not file_path.is_file():
        raise FileNotFoundError(f"File not found: {file_path}")
    return pd.read_csv(file_path)


def calculate_metrics(
    actual: pd.Series, predicted: pd.Series
) -> tuple[float, float, float]:
    evaluation = pd.DataFrame(
        {
            "Actual": pd.to_numeric(actual, errors="coerce"),
            "Predicted": pd.to_numeric(predicted, errors="coerce"),
        }
    ).dropna()
    if evaluation.empty:
        raise ValueError("The selected prediction file has no valid evaluation rows.")

    errors = evaluation["Actual"] - evaluation["Predicted"]
    nonzero = evaluation["Actual"].ne(0)
    if not nonzero.any():
        raise ValueError("MAPE cannot be calculated when every actual value is zero.")

    mae = errors.abs().mean()
    rmse = errors.pow(2).mean() ** 0.5
    mape = (
        errors[nonzero].abs().div(evaluation.loc[nonzero, "Actual"]).mean() * 100
    )
    return round(mae, 4), round(rmse, 4), round(mape, 4)


def plot_prediction(
    frame: pd.DataFrame, actual_column: str, prediction_column: str, model_name: str
) -> None:
    fig, ax = plt.subplots(figsize=(12, 6))
    x_values = pd.to_datetime(frame["Date"], errors="coerce")
    ax.plot(x_values, frame[actual_column], label="Actual")
    ax.plot(x_values, frame[prediction_column], label=model_name)
    ax.set_title(f"{model_name} Forecast vs Actual")
    ax.set_xlabel("Date")
    ax.set_ylabel("Closing Price (INR)")
    ax.legend()
    fig.autofmt_xdate()
    st.pyplot(fig)
    plt.close(fig)


def render_app() -> None:
    st.set_page_config(
        page_title="Stock Price Forecasting",
        page_icon="📈",
        layout="wide",
    )
    st.title("Stock Price Forecasting")
    st.caption("Historical benchmark results for educational use—not investment advice.")

    selected_model = st.sidebar.selectbox("Select model", list(MODEL_FILES))
    show_data = st.sidebar.checkbox("Show prediction data", value=True)

    try:
        prediction_column = MODEL_PREDICTION_COLUMNS[selected_model]
        frame = load_csv(MODEL_FILES[selected_model])
        required_columns = {"Date", "Actual", prediction_column}
        missing_columns = required_columns.difference(frame.columns)
        if missing_columns:
            raise ValueError(
                f"The selected file is missing columns: {sorted(missing_columns)}"
            )

        st.subheader(f"{selected_model} results")
        mae, rmse, mape = calculate_metrics(
            frame["Actual"], frame[prediction_column]
        )
        metric_columns = st.columns(3)
        metric_columns[0].metric("MAE", mae)
        metric_columns[1].metric("RMSE", rmse)
        metric_columns[2].metric("MAPE (%)", mape)

        plot_prediction(frame, "Actual", prediction_column, selected_model)
        if show_data:
            st.subheader("Prediction data")
            st.dataframe(frame.head(20), width="stretch")

        comparison_file = PROCESSED_DIR / "model_comparison.csv"
        if comparison_file.is_file():
            st.subheader("Model comparison")
            comparison = load_csv(comparison_file)
            st.dataframe(comparison, width="stretch")

            fig, ax = plt.subplots(figsize=(10, 5))
            ax.bar(comparison["Model"], comparison["RMSE"])
            ax.set_title("RMSE comparison")
            ax.set_xlabel("Model")
            ax.set_ylabel("RMSE")
            ax.tick_params(axis="x", rotation=20)
            st.pyplot(fig)
            plt.close(fig)
    except (FileNotFoundError, ValueError, pd.errors.ParserError) as error:
        st.error(str(error))
        st.info("Verify the generated files in data/processed and try again.")


if __name__ == "__main__":
    render_app()
