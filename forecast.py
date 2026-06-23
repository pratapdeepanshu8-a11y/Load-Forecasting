import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# Load Data
file_path = r"C:\Users\DELL\Downloads\archive (7)\hourlyLoadDataIndia.xlsx"

df = pd.read_excel(file_path)

# Convert datetime
df['datetime'] = pd.to_datetime(df['datetime'])

# Time Features
df['hour'] = df['datetime'].dt.hour
df['day'] = df['datetime'].dt.day
df['month'] = df['datetime'].dt.month
df['dayofweek'] = df['datetime'].dt.dayofweek

# Lag Features
df['lag_1'] = df['National Hourly Demand'].shift(1)
df['lag_24'] = df['National Hourly Demand'].shift(24)
df['lag_168'] = df['National Hourly Demand'].shift(168)

# Remove missing rows
df.dropna(inplace=True)

# Features (Input)
X = df[
    [
        'hour',
        'day',
        'month',
        'dayofweek',
        'lag_1',
        'lag_24',
        'lag_168'
    ]
]

# Target (Output)
y = df['National Hourly Demand']

# Train-Test Split
split = int(len(df) * 0.8)

X_train = X[:split]
X_test = X[split:]

y_train = y[:split]
y_test = y[split:]

# Train Model
model = RandomForestRegressor(
    n_estimators=20,
    random_state=42,
    n_jobs=-1
)
model.fit(X_train, y_train)

# Predictions
predictions = model.predict(X_test)

# Accuracy
rmse = np.sqrt(mean_squared_error(y_test, predictions))

print("RMSE =", rmse)

# Sample Predictions
print("\nActual vs Predicted")

for i in range(10):
    print(
        "Actual:",
        round(y_test.iloc[i], 2),
        " Predicted:",
        round(predictions[i], 2)
    )
latest = X.iloc[[-1]]

next_hour_prediction = model.predict(latest)

print("\nPredicted Next Hour Demand:")
print(round(next_hour_prediction[0], 2))

# Last available row
future_data = X.iloc[[-1]].copy()

tomorrow_predictions = []

for i in range(24):

    pred = model.predict(future_data)[0]

    tomorrow_predictions.append(pred)

    # Update lag_1 with latest prediction
    future_data.loc[:, 'lag_1'] = pred

    # Advance hour
    future_data.loc[:, 'hour'] = (future_data['hour'].iloc[0] + 1) % 24

tomorrow_total = sum(tomorrow_predictions)

print("\n===================================")
print("Predicted Tomorrow Total Consumption")
print(round(tomorrow_total, 2), "MWh")
print("===================================")
plt.figure(figsize=(12,6))

plt.plot(range(1,25),
         tomorrow_predictions,
         marker='o')

plt.title("Tomorrow Hourly Demand Forecast")
plt.xlabel("Hour of Tomorrow")
plt.ylabel("Predicted Demand (MW)")

plt.grid(True)

plt.show()
import matplotlib.pyplot as plt

plt.figure(figsize=(12,6))

plt.plot(y_test.values[:100], label="Actual")
plt.plot(predictions[:100], label="Predicted")

plt.title("Actual vs Predicted Electricity Demand")
plt.xlabel("Hours")
plt.ylabel("Demand (MW)")

plt.legend()

plt.show()
