import pandas as pd
import joblib
import matplotlib.pyplot as plt

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

# Load model
model = joblib.load(
    BASE_DIR / "models" / "final_load_model.pkl"
)

# Load feature-engineered data
df = pd.read_csv(
    BASE_DIR / "outputs" / "featured_data.csv"
)

# User Input
forecast_date = input(
    "Enter forecast date (DD-MM-YYYY): "
)

forecast_date = pd.to_datetime(
    forecast_date,
    format="%d-%m-%Y"
)

# Take last 24 records
future_df = df.tail(24).copy()

# Update date-related features
future_df["day"] = forecast_date.day
future_df["month"] = forecast_date.month
future_df["weekday"] = forecast_date.weekday()

# Features used for prediction
features = [
    'hour',
    'day',
    'month',
    'weekday',
    'lag_1',
    'lag_24',
    'lag_48',
    'lag_72',
    'lag_168',
    'rolling_mean_24',
    'rolling_mean_168'
]

X_future = future_df[features]

# Predict
predictions = model.predict(X_future)

# Results
forecast_df = pd.DataFrame({
    "Hour": range(24),
    "Predicted Demand (MW)": predictions
})

print("\n===== 24 HOUR FORECAST =====")
print(forecast_df)

print("\nPeak Demand:")
print(round(forecast_df["Predicted Demand (MW)"].max(), 2))

print("\nMinimum Demand:")
print(round(forecast_df["Predicted Demand (MW)"].min(), 2))

print("\nAverage Demand:")
print(round(forecast_df["Predicted Demand (MW)"].mean(), 2))

# Forecast Graph
plt.figure(figsize=(12,6))

plt.plot(
    forecast_df["Hour"],
    forecast_df["Predicted Demand (MW)"],
    marker="o"
)

plt.title(
    f"24-Hour Load Forecast ({forecast_date.date()})"
)

plt.xlabel("Hour")
plt.ylabel("Demand (MW)")
plt.grid(True)

plt.tight_layout()
plt.show()