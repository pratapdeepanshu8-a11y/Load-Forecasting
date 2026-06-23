import pandas as pd
import matplotlib.pyplot as plt

results = {
    "Model": [
        "Linear Regression",
        "Random Forest",
        "XGBoost",
        "LightGBM",
        "CatBoost"
    ],
    "MAE": [
        2568.47,
        2794.36,
        6733.68,
        6404.45,
        6322.79
    ],
    "RMSE": [
        3348.33,
        4064.92,
        8856.18,
        8357.77,
        8379.76
    ],
    "R2": [
        0.9707,
        0.9568,
        0.7948,
        0.8173,
        0.8163
    ]
}

df = pd.DataFrame(results)

plt.figure(figsize=(15, 5))

# MAE
plt.subplot(1, 3, 1)
plt.bar(df["Model"], df["MAE"])
plt.title("MAE Comparison")
plt.xticks(rotation=45)

# RMSE
plt.subplot(1, 3, 2)
plt.bar(df["Model"], df["RMSE"])
plt.title("RMSE Comparison")
plt.xticks(rotation=45)

# R2
plt.subplot(1, 3, 3)
plt.bar(df["Model"], df["R2"])
plt.title("R² Comparison")
plt.xticks(rotation=45)

plt.tight_layout()
plt.show()