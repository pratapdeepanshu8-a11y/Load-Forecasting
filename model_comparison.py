import matplotlib.pyplot as plt

models = [
    "Linear Regression",
    "Random Forest",
    "XGBoost"
]

mae = [
    2568,
    2794,
    2705
]

plt.figure(figsize=(8,5))

plt.bar(models, mae)

plt.title("Model Comparison (MAE)")
plt.ylabel("MAE")

plt.show()