import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

# Load Dataset
BASE_DIR = Path(__file__).resolve().parent.parent

file_path = BASE_DIR / "data" / "hourlyLoadDataIndia.xlsx"

df = pd.read_excel(file_path)

# Convert datetime
df["datetime"] = pd.to_datetime(df["datetime"])

# Check missing values
print("Missing Values:")
print(df.isnull().sum())

# Plot National Demand
plt.figure(figsize=(15, 6))

plt.plot(
    df["datetime"],
    df["National Hourly Demand"]
)

plt.title("National Hourly Demand Over Time")
plt.xlabel("Date")
plt.ylabel("Demand (MW)")
plt.grid(True)

plt.tight_layout()

plt.show()