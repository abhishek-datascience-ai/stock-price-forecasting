# Stock Price Forecasting Using Machine Learning

![Python](https://img.shields.io/badge/Python-3.12-3776AB)
![Time Series](https://img.shields.io/badge/Time%20Series-Forecasting-4C78A8)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-F7931E)
![TensorFlow](https://img.shields.io/badge/TensorFlow-Keras-FF6F00)
![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

A reproducible time-series project that benchmarks statistical, machine-learning, and deep-learning approaches for forecasting the closing price of Tata Consultancy Services (`TCS.NS`). A Streamlit app presents the stored predictions and evaluation metrics.

## Features

- Historical stock-data collection from Yahoo Finance
- Data cleaning, exploratory analysis, and feature engineering
- Chronological train/test separation
- Baseline, ARIMA, Auto ARIMA, LSTM, and Prophet forecasts
- Aligned MAE, RMSE, and MAPE evaluation
- Interactive Streamlit result comparison
- Automated data and evaluation tests

## Dataset

The committed dataset contains `TCS.NS` daily market observations from January 2018 through December 2025, including date, open, high, low, close, and volume fields.

Yahoo Finance data is intended for research and educational use. Review the provider's current terms before using it in another context.

## Evaluation

All prediction files use the same 395 observations from `2024-05-30` through `2025-12-30`. Models forecast the full test window from a single training cutoff without incorporating test-period observations.

| Model | MAE | RMSE | MAPE |
| --- | ---: | ---: | ---: |
| Baseline | 459.86 | 510.43 | 13.08% |
| ARIMA | 459.78 | 510.21 | 13.07% |
| Auto ARIMA | 643.53 | 784.31 | 19.22% |
| LSTM | 406.83 | 465.10 | 11.02% |
| Prophet | 839.99 | 1053.25 | 25.42% |

LSTM has the lowest recorded error on this fixed historical benchmark. These results do not establish future trading performance and should not be interpreted as investment advice.

## Technology Stack

| Technology | Purpose |
| --- | --- |
| Python 3.12 | Application and notebook runtime |
| Pandas and NumPy | Data preparation and numerical processing |
| scikit-learn | Scaling and evaluation metrics |
| statsmodels and pmdarima | ARIMA forecasting |
| TensorFlow and Keras | LSTM forecasting |
| Prophet | Additive time-series forecasting |
| Matplotlib and Seaborn | Visualization |
| Streamlit | Interactive results app |
| Pytest | Automated validation |

## Project Structure

```text
stock-price-forecasting/
├── app/                    # Streamlit application
├── data/
│   ├── raw/                # Downloaded source data
│   └── processed/          # Features, predictions, and metrics
├── models/                 # Saved Keras model and forecast chart
├── notebooks/              # Analysis and modeling workflow
├── src/                    # Data preparation and model evaluation
├── tests/                  # Automated validation
├── LICENSE
├── requirements.txt        # Streamlit runtime dependencies
└── requirements-dev.txt    # Modeling and notebook dependencies
```

## Setup

Clone the repository and create a virtual environment:

```powershell
git clone https://github.com/abhishek-datascience-ai/stock-price-forecasting.git
cd stock-price-forecasting
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Install the Streamlit runtime dependencies:

```powershell
python -m pip install -r requirements.txt
```

For notebooks, forecasting, and tests, install:

```powershell
python -m pip install -r requirements-dev.txt
```

## Usage

Run the Streamlit app from the repository root:

```powershell
python -m streamlit run app/streamlit_app.py
```

Open the notebook workflow with:

```powershell
python -m jupyter lab
```

After regenerating model predictions, validate and rebuild the shared comparison:

```powershell
python -m src.model_evaluation
python -m pytest tests -q
```

## Important Notes

- The app reads committed prediction artifacts and does not retrain models during requests.
- The saved Keras model represents the recorded LSTM run.
- Market behavior changes over time; retraining and out-of-sample monitoring are required for real-world experimentation.

## License

This project is available under the MIT License.
