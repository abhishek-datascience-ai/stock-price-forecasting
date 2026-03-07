# Stock Price Forecasting (Time Series)

## Project Overview
This project forecasts stock closing prices using Time Series and Deep Learning models.  
The objective is to compare multiple forecasting approaches and identify the best performing model for stock price prediction.

## Business Problem
Stock price forecasting helps investors and traders estimate future price movements.  
This project focuses on predicting future closing prices using historical stock market data.

## Project Objectives
- Perform stock market time series analysis
- Build forecasting models using Baseline, ARIMA, Auto ARIMA, LSTM, and Prophet
- Compare model performance using MAE, RMSE, and MAPE
- Deploy the final solution using Streamlit

## Dataset
Source: Yahoo Finance  
Ticker Used: `TCS.NS`

Features used:
- Date
- Open
- High
- Low
- Close
- Volume

## Project Workflow
1. Problem Definition
2. Dataset Collection
3. Data Cleaning and Preprocessing
4. Exploratory Data Analysis
5. Feature Engineering
6. Train/Test Split
7. Baseline Forecasting
8. ARIMA Modeling
9. Auto ARIMA Tuning
10. LSTM Deep Learning Model
11. Prophet Forecasting
12. Model Evaluation
13. Forecast Visualization
14. Streamlit Deployment

## Models Used
- Baseline Naive Forecast
- ARIMA
- Auto ARIMA
- LSTM
- Prophet

## Evaluation Metrics
- MAE
- RMSE
- MAPE

## Project Structure
```text
stock-price-forecasting
│
├── app
│   └── streamlit_app.py
│
├── data
│   ├── raw
│   └── processed
│
├── models
│
├── notebooks
│
├── src
│
├── requirements.txt
├── README.md
└── .gitignore

## Results
| Model      |  MAE  |  RMSE  |  MAPE  |
| ---------- |-------|--------|--------|
| Baseline   | 97.81 | 126.59 |  2.64  |
| ARIMA      | 455.70| 508.35 | 12.89  |
| Auto ARIMA | 459.86| 510.43 | 13.07  |
| LSTM       | 72.89 | 96.05  | 1.95   |
| Prophet    | 822.19| 1025.96| 24.85  |

## Best Model

LSTM achieved the lowest RMSE and performed best among all tested models.

---

## Streamlit App

This project includes a **Streamlit web app** for interactive forecasting result visualization.

Run the app locally:

```bash
streamlit run app/streamlit_app.py

## Installation

Clone the repository:

```bash
git clone <your-github-repo-link>
cd stock-price-forecasting

Install dependencies:

```bash
pip install -r requirements.txt

## Key Learnings

- Time series preprocessing
- Lag feature engineering
- ARIMA modeling
- LSTM sequence modeling
- Forecast evaluation
- Streamlit deployment

---

## Future Improvements

- Multi-stock forecasting
- Sentiment analysis integration
- Hyperparameter tuning
- Advanced deep learning architectures
- Live market data integration

---

## Author

Abhishek
