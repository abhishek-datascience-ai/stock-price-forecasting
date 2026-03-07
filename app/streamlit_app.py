import os
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st


# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="Stock Price Forecasting",
    page_icon="📈",
    layout="wide"
)

st.title("Stock Price Forecasting (Time Series)")
st.markdown("Compare actual stock prices with forecasting model predictions.")


# =========================
# FILE PATHS
# =========================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data", "processed")

MODEL_FILES = {
    "Baseline": os.path.join(DATA_DIR, "baseline_predictions.csv"),
    "ARIMA": os.path.join(DATA_DIR, "arima_predictions.csv"),
    "Auto ARIMA": os.path.join(DATA_DIR, "auto_arima_predictions.csv"),
    "LSTM": os.path.join(DATA_DIR, "lstm_predictions.csv"),
    "Prophet": os.path.join(DATA_DIR, "prophet_predictions.csv"),
}

MODEL_PREDICTION_COLUMNS = {
    "Baseline": "Prediction",
    "ARIMA": "ARIMA_Prediction",
    "Auto ARIMA": "Auto_ARIMA_Prediction",
    "LSTM": "LSTM_Prediction",
    "Prophet": "Prophet_Prediction",
}


# =========================
# HELPER FUNCTIONS
# =========================
@st.cache_data
def load_csv(file_path: str) -> pd.DataFrame:
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    return pd.read_csv(file_path)


def calculate_metrics(actual: pd.Series, predicted: pd.Series):
    actual = pd.to_numeric(actual, errors="coerce")
    predicted = pd.to_numeric(predicted, errors="coerce")

    df_eval = pd.DataFrame({"Actual": actual, "Predicted": predicted}).dropna()

    mae = (df_eval["Actual"] - df_eval["Predicted"]).abs().mean()
    rmse = ((df_eval["Actual"] - df_eval["Predicted"]) ** 2).mean() ** 0.5
    mape = ((df_eval["Actual"] - df_eval["Predicted"]).abs() / df_eval["Actual"]).mean() * 100

    return round(mae, 4), round(rmse, 4), round(mape, 4)


def plot_prediction(df: pd.DataFrame, actual_col: str, pred_col: str, model_name: str):
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(df[actual_col].values, label="Actual")
    ax.plot(df[pred_col].values, label=model_name)
    ax.set_title(f"{model_name} Forecast vs Actual")
    ax.set_xlabel("Time Step")
    ax.set_ylabel("Close Price")
    ax.legend()
    st.pyplot(fig)


# =========================
# SIDEBAR
# =========================
st.sidebar.header("Controls")
selected_model = st.sidebar.selectbox(
    "Select Model",
    list(MODEL_FILES.keys())
)

show_data = st.sidebar.checkbox("Show prediction data", value=True)


# =========================
# LOAD DATA
# =========================
try:
    file_path = MODEL_FILES[selected_model]
    pred_col = MODEL_PREDICTION_COLUMNS[selected_model]

    df = load_csv(file_path)

    if "Actual" not in df.columns:
        st.error("The selected file does not contain 'Actual' column.")
        st.stop()

    if pred_col not in df.columns:
        st.error(f"The selected file does not contain '{pred_col}' column.")
        st.stop()

    st.subheader(f"{selected_model} Results")

    mae, rmse, mape = calculate_metrics(df["Actual"], df[pred_col])

    col1, col2, col3 = st.columns(3)
    col1.metric("MAE", mae)
    col2.metric("RMSE", rmse)
    col3.metric("MAPE (%)", mape)

    plot_prediction(df, "Actual", pred_col, selected_model)

    if show_data:
        st.subheader("Prediction Data")
        st.dataframe(df.head(20), use_container_width=True)

except FileNotFoundError as e:
    st.error(str(e))
    st.info("Make sure all prediction CSV files are available inside data/processed/")
except Exception as e:
    st.error(f"Unexpected error: {e}")


# =========================
# OPTIONAL MODEL COMPARISON
# =========================
comparison_file = os.path.join(DATA_DIR, "model_comparison.csv")

if os.path.exists(comparison_file):
    st.subheader("Model Comparison")
    comparison_df = pd.read_csv(comparison_file)
    st.dataframe(comparison_df, use_container_width=True)

    if "RMSE" in comparison_df.columns:
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.bar(comparison_df["Model"], comparison_df["RMSE"])
        ax.set_title("RMSE Comparison")
        ax.set_xlabel("Model")
        ax.set_ylabel("RMSE")
        plt.xticks(rotation=20)
        st.pyplot(fig)