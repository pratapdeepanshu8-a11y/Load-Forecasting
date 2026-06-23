import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Project root
BASE_DIR = Path(__file__).resolve().parent

print("Project Folder:")
print(BASE_DIR)

# Create assets folder if it doesn't exist
assets_dir = BASE_DIR / "assets"
assets_dir.mkdir(exist_ok=True)

# Load featured data
df = pd.read_csv(BASE_DIR / "outputs" / "featured_data.csv")

# ==========================
# HOURLY DEMAND TREND
# ==========================

hourly = df.groupby("hour")["National Hourly Demand"].mean()

plt.figure(figsize=(10, 5))
plt.plot(hourly.index, hourly.values)
plt.title("Average Hourly Demand")
plt.xlabel("Hour")
plt.ylabel("Demand (MW)")
plt.grid(True)

plt.savefig(assets_dir / "hourly_trend.png")
plt.close()

# ==========================
# MONTHLY DEMAND TREND
# ==========================

monthly = df.groupby("month")["National Hourly Demand"].mean()

plt.figure(figsize=(10, 5))
plt.plot(monthly.index, monthly.values)
plt.title("Average Monthly Demand")
plt.xlabel("Month")
plt.ylabel("Demand (MW)")
plt.grid(True)

plt.savefig(assets_dir / "monthly_trend.png")
plt.close()

print("\nGraphs saved successfully!")
print("Location:")
print(assets_dir)