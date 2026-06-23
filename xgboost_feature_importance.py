import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

from xgboost import XGBRegressor
from xgboost import plot_importance

# Load data
BASE_DIR = Path(__file__).resolve().parent.parent
file_path = BASE_DIR / "data" / "hourlyLoadDataIndia.xlsx"

df = pd.read_excel(file_path)

df["datetime"] = pd.to_datetime(df["datetime"])

# Time features
df["hour"] = df["datetime"].dt.hour
df["day"] = df["datetime"].dt.day
df["month"] = df["datetime"].dt.month
df["weekday"] = df["datetime"].dt.weekday

# Lag features
df["lag_1"] = df["National Hourly Demand"].shift(1)
df["lag_24"] = df["National Hourly Demand"].shift(24)
df["lag_48"] = df["National Hourly Demand"].shift(48)
df["lag_72"] = df["National Hourly Demand"].shift(72)
df["lag_168"] = df["National Hourly Demand"].shift(168)

# Rolling features
df["rolling_mean_24"] = df["National Hourly Demand"].rolling(24).mean()
df["rolling_mean_168"] = df["National Hourly Demand"].rolling(168).mean()

df = df.dropna()

features = [
    "hour",
    "day",
    "month",
    "weekday",
    "lag_1",
    "lag_24",
    "lag_48",
    "lag_72",
    "lag_168",
    "rolling_mean_24",
    "rolling_mean_168"
]

X = df[features]
y = df["National Hourly Demand"]

model = XGBRegressor(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=6,
    random_state=42
)

model.fit(X, y)

plt.figure(figsize=(10, 6))
plot_importance(model)

plt.title("XGBoost Feature Importance")
plt.tight_layout()
plt.show()