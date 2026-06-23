import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

BASE_DIR = Path(__file__).resolve().parent.parent
file_path = BASE_DIR / "data" / "hourlyLoadDataIndia.xlsx"

df = pd.read_excel(file_path)

df["datetime"] = pd.to_datetime(df["datetime"])

df["hour"] = df["datetime"].dt.hour

hourly_avg = df.groupby("hour")["National Hourly Demand"].mean()

plt.figure(figsize=(10, 5))
hourly_avg.plot()

plt.title("Average Demand by Hour")
plt.xlabel("Hour")
plt.ylabel("Average Demand (MW)")
plt.grid(True)

plt.show()