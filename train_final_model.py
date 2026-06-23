import pandas as pd
import joblib
from pathlib import Path

from sklearn.linear_model import LinearRegression

# Base directory
BASE_DIR = Path(__file__).resolve().parent.parent

# Load feature engineered dataset
df = pd.read_csv(BASE_DIR / "outputs" / "featured_data.csv")

# Check available columns
print("Columns:")
print(df.columns.tolist())

# Features
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

# Target
target = 'National Hourly Demand'

# Prepare data
X = df[features]
y = df[target]

# Train model
model = LinearRegression()
model.fit(X, y)

# Create models folder if it doesn't exist
models_folder = BASE_DIR / "models"
models_folder.mkdir(exist_ok=True)

# Save model
joblib.dump(
    model,
    models_folder / "final_load_model.pkl"
)

print("\nModel Saved Successfully!")
print("Location:")
print(models_folder / "final_load_model.pkl")