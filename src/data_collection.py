import os
import yfinance as yf
import pandas as pd


def download_stock_data(ticker_symbol: str, start_date: str, end_date: str, save_path: str) -> pd.DataFrame:
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    df = yf.download(
        ticker_symbol,
        start=start_date,
        end=end_date,
        auto_adjust=False,
        progress=False
    )

    if df.empty:
        raise ValueError(f"No data found for ticker: {ticker_symbol}")

    df.reset_index(inplace=True)

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0] if isinstance(col, tuple) else col for col in df.columns]

    required_columns = ["Date", "Open", "High", "Low", "Close", "Volume"]
    df = df[required_columns]

    df.to_csv(save_path, index=False)
    return df


if __name__ == "__main__":
    ticker_symbol = "TCS.NS"
    start_date = "2018-01-01"
    end_date = "2025-12-31"
    save_path = "data/raw/tcs_stock_data.csv"

    df = download_stock_data(ticker_symbol, start_date, end_date, save_path)

    print("Dataset Shape:", df.shape)
    print(df.head())
    print(f"\nSaved at: {save_path}")