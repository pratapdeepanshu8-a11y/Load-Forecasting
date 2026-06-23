import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Project root
BASE_DIR = Path(__file__).resolve().parent

print("Project Folder:")
print(BASE_DIR)

# Load data
df = pd.read_csv(
    BASE_DIR / "outputs" / "featured_data.csv"
)

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

target = "National Hourly Demand"

X = df[features]
y = df[target]

# Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    shuffle=False
)

# Train Model
model = LinearRegression()
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

assets_dir = BASE_DIR / "assets"
assets_dir.mkdir(exist_ok=True)

# ====================================
# Actual vs Predicted
# ====================================

plt.figure(figsize=(10, 5))

plt.plot(
    y_test.values[:200],
    label="Actual"
)

plt.plot(
    y_pred[:200],
    label="Predicted"
)

plt.title("Actual vs Predicted")
plt.legend()

plt.savefig(
    assets_dir / "actual_vs_predicted.png"
)

plt.close()

# ====================================
# Residual Analysis
# ====================================

residuals = y_test.values - y_pred

plt.figure(figsize=(8, 5))

plt.hist(
    residuals,
    bins=30
)

plt.title("Residual Analysis")

plt.savefig(
    assets_dir / "residual_analysis.png"
)

plt.close()

print("\nModel insight graphs created!")
print("Location:")
print(assets_dir)