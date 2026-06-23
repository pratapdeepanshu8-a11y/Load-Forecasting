import pandas as pd

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

df.to_csv(
    "outputs/model_results.csv",
    index=False
)

print("Model results saved!")