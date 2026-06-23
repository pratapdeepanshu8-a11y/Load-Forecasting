import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

from xgboost import XGBRegressor

BASE_DIR = Path(__file__).resolve().parent.parent
file_path = BASE_DIR / "data" / "hourlyLoadDataIndia.xlsx"

df = pd.read_excel(file_path)

df["datetime"] = pd.to_datetime(df["datetime"])

# Features
df["hour"] = df["datetime"].dt.hour
df["day"] = df["datetime"].dt.day
df["month"] = df["datetime"].dt.month
df["weekday"] = df["datetime"].dt.weekday

df["lag_1"] = df["National Hourly Demand"].shift(1)
df["lag_24"] = df["National Hourly Demand"].shift(24)
df["lag_48"] = df["National Hourly Demand"].shift(48)
df["lag_72"] = df["National Hourly Demand"].shift(72)
df["lag_168"] = df["National Hourly Demand"].shift(168)

df["rolling_mean_24"] = df["National Hourly Demand"].rolling(24).mean()
df["rolling_mean_168"] = df["National Hourly Demand"].rolling(168).mean()

df["target"] = df["National Hourly Demand"].shift(-24)

df = df.dropna()

features = [
    "hour","day","month","weekday",
    "lag_1","lag_24","lag_48","lag_72","lag_168",
    "rolling_mean_24","rolling_mean_168"
]

X = df[features]
y = df["target"]

split_index = int(len(df) * 0.8)

X_train = X[:split_index]
X_test = X[split_index:]

y_train = y[:split_index]
y_test = y[split_index:]

model = XGBRegressor(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=6,
    random_state=42
)

model.fit(X_train, y_train)

predictions = model.predict(X_test)

plt.figure(figsize=(15,6))

plt.plot(
    y_test.iloc[:500].values,
    label="Actual"
)

plt.plot(
    predictions[:500],
    label="Predicted"
)

plt.title("Actual vs Predicted Demand (24-Hour Ahead)")
plt.xlabel("Time")
plt.ylabel("Demand (MW)")
plt.legend()
plt.grid(True)

plt.show()