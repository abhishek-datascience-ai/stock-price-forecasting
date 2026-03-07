import os
import pandas as pd


def clean_stock_data(input_path: str, output_path: str) -> pd.DataFrame:
    # ==============================
    # 1. LOAD DATA
    # ==============================
    df = pd.read_csv(input_path)

    print("Raw Dataset Shape:", df.shape)
    print("\nFirst 5 Rows:")
    print(df.head())

    print("\nColumn Names:")
    print(df.columns.tolist())

    print("\nData Types Before Cleaning:")
    print(df.dtypes)

    # ==============================
    # 2. CONVERT DATE COLUMN
    # ==============================
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

    # ==============================
    # 3. REMOVE ROWS WITH INVALID DATES
    # ==============================
    df = df.dropna(subset=["Date"])

    # ==============================
    # 4. SORT BY DATE
    # ==============================
    df = df.sort_values(by="Date").reset_index(drop=True)

    # ==============================
    # 5. REMOVE DUPLICATES
    # ==============================
    duplicate_count = df.duplicated().sum()
    print("\nDuplicate Rows:", duplicate_count)

    df = df.drop_duplicates().reset_index(drop=True)

    # ==============================
    # 6. CHECK MISSING VALUES
    # ==============================
    print("\nMissing Values Before Handling:")
    print(df.isnull().sum())

    # ==============================
    # 7. HANDLE MISSING VALUES
    # ==============================
    # For stock data, forward fill is commonly used for small gaps
    df = df.ffill()

    # If any values still remain missing, remove them
    df = df.dropna().reset_index(drop=True)

    # ==============================
    # 8. ENSURE NUMERIC COLUMNS ARE NUMERIC
    # ==============================
    numeric_columns = ["Open", "High", "Low", "Close", "Volume"]

    for col in numeric_columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Drop rows if numeric conversion created NaNs
    df = df.dropna().reset_index(drop=True)

    # ==============================
    # 9. FINAL DATA CHECK
    # ==============================
    print("\nData Types After Cleaning:")
    print(df.dtypes)

    print("\nMissing Values After Cleaning:")
    print(df.isnull().sum())

    print("\nCleaned Dataset Shape:", df.shape)

    print("\nFirst 5 Cleaned Rows:")
    print(df.head())

    print("\nLast 5 Cleaned Rows:")
    print(df.tail())

    # ==============================
    # 10. CREATE OUTPUT FOLDER
    # ==============================
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # ==============================
    # 11. SAVE CLEANED DATA
    # ==============================
    df.to_csv(output_path, index=False)
    print(f"\nCleaned dataset saved at: {output_path}")

    return df


if __name__ == "__main__":
    input_path = "data/raw/tcs_stock_data.csv"
    output_path = "data/processed/tcs_stock_data_cleaned.csv"

    clean_stock_data(input_path, output_path)