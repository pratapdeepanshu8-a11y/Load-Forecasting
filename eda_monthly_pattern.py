import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

BASE_DIR = Path(__file__).resolve().parent.parent
file_path = BASE_DIR / "data" / "hourlyLoadDataIndia.xlsx"

df = pd.read_excel(file_path)

df["datetime"] = pd.to_datetime(df["datetime"])

df["month"] = df["datetime"].dt.month_name()

month_order = [
    "January","February","March","April",
    "May","June","July","August",
    "September","October","November","December"
]

monthly_avg = (
    df.groupby("month")["National Hourly Demand"]
      .mean()
      .reindex(month_order)
)

plt.figure(figsize=(12,5))
monthly_avg.plot(kind="bar")

plt.title("Average Demand by Month")
plt.xlabel("Month")
plt.ylabel("Average Demand (MW)")
plt.grid(axis="y")

plt.show()