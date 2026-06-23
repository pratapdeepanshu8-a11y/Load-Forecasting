import pandas as pd
from pathlib import Path

# Load data
BASE_DIR = Path(__file__).resolve().parent.parent
file_path = BASE_DIR / "data" / "hourlyLoadDataIndia.xlsx"

df = pd.read_excel(file_path)

# Datetime conversion
df["datetime"] = pd.to_datetime(df["datetime"])

# Time Features
df["hour"] = df["datetime"].dt.hour
df["day"] = df["datetime"].dt.day
df["month"] = df["datetime"].dt.month
df["weekday"] = df["datetime"].dt.weekday

# Lag Features
df["lag_1"] = df["National Hourly Demand"].shift(1)
df["lag_24"] = df["National Hourly Demand"].shift(24)
df["lag_48"] = df["National Hourly Demand"].shift(48)
df["lag_72"] = df["National Hourly Demand"].shift(72)
df["lag_168"] = df["National Hourly Demand"].shift(168)

# Rolling Features
df["rolling_mean_24"] = (
    df["National Hourly Demand"]
    .rolling(window=24)
    .mean()
)

df["rolling_mean_168"] = (
    df["National Hourly Demand"]
    .rolling(window=168)
    .mean()
)

# Remove rows with NaN created by lag features
df = df.dropna()

print(df.head())

print("\nShape after feature engineering:")
print(df.shape)

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

df.to_csv(
    BASE_DIR / "outputs" / "featured_data.csv",
    index=False
)

print("Feature engineered data saved!")