import pandas as pd
from pathlib import Path

from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

# Load data
BASE_DIR = Path(__file__).resolve().parent.parent
file_path = BASE_DIR / "data" / "hourlyLoadDataIndia.xlsx"

df = pd.read_excel(file_path)

# Datetime
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
df["rolling_mean_24"] = df["National Hourly Demand"].rolling(24).mean()
df["rolling_mean_168"] = df["National Hourly Demand"].rolling(168).mean()

# NEW TARGET
df["target"] = df["National Hourly Demand"].shift(-24)

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
y = df["target"]

# Time-based split
split_index = int(len(df) * 0.8)

X_train = X[:split_index]
X_test = X[split_index:]

y_train = y[:split_index]
y_test = y[split_index:]

# Model
model = XGBRegressor(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=6,
    random_state=42
)

model.fit(X_train, y_train)

predictions = model.predict(X_test)

mae = mean_absolute_error(y_test, predictions)
rmse = mean_squared_error(y_test, predictions) ** 0.5
r2 = r2_score(y_test, predictions)

print("\n===== DAY AHEAD FORECASTING =====")
print("MAE :", round(mae, 2))
print("RMSE:", round(rmse, 2))
print("R2  :", round(r2, 4))