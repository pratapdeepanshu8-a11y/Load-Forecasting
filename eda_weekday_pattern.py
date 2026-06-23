import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

BASE_DIR = Path(__file__).resolve().parent.parent
file_path = BASE_DIR / "data" / "hourlyLoadDataIndia.xlsx"

df = pd.read_excel(file_path)

df["datetime"] = pd.to_datetime(df["datetime"])

df["weekday"] = df["datetime"].dt.day_name()

weekday_order = [
    "Monday",
    "Tuesday",
    "Wednesday",
    "Thursday",
    "Friday",
    "Saturday",
    "Sunday"
]

weekly_avg = (
    df.groupby("weekday")["National Hourly Demand"]
      .mean()
      .reindex(weekday_order)
)

plt.figure(figsize=(10,5))
weekly_avg.plot(kind="bar")

plt.title("Average Demand by Weekday")
plt.xlabel("Day")
plt.ylabel("Average Demand (MW)")
plt.grid(axis="y")

plt.show()