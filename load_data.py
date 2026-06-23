import pandas as pd

df = pd.read_excel(
    r"C:\Users\KIIT0001\Desktop\LoadForecastingProject\data\hourlyLoadDataIndia.xlsx"
)

print(df.head())

print("\nShape:")
print(df.shape)

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

data_folder = BASE_DIR / "data"

print(list(data_folder.iterdir()))