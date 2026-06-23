import pandas as pd
import matplotlib.pyplot as plt

# Model Results
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

print("\n===== MODEL COMPARISON =====")
print(df)

# Save table
df.to_csv("model_comparison.csv", index=False)

# -------------------
# MAE Graph
# -------------------
plt.figure(figsize=(8,5))
plt.bar(df["Model"], df["MAE"])
plt.title("Model Comparison - MAE")
plt.ylabel("MAE")
plt.xticks(rotation=20)
plt.tight_layout()
plt.show()

# -------------------
# RMSE Graph
# -------------------
plt.figure(figsize=(8,5))
plt.bar(df["Model"], df["RMSE"])
plt.title("Model Comparison - RMSE")
plt.ylabel("RMSE")
plt.xticks(rotation=20)
plt.tight_layout()
plt.show()

# -------------------
# R2 Graph
# -------------------
plt.figure(figsize=(8,5))
plt.bar(df["Model"], df["R2"])
plt.title("Model Comparison - R² Score")
plt.ylabel("R²")
plt.xticks(rotation=20)
plt.tight_layout()
plt.show()