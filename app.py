import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from pathlib import Path

# ==================================================
# PAGE CONFIG
# ==================================================

st.set_page_config(
    page_title="Day Ahead Load Forecasting",
    page_icon="⚡",
    layout="wide"
)

# ==================================================
# PATHS
# ==================================================

BASE_DIR = Path(__file__).resolve().parent

# ==================================================
# LOAD MODEL
# ==================================================

model = joblib.load(
    BASE_DIR / "models" / "final_load_model.pkl"
)

# ==================================================
# LOAD DATA
# ==================================================

df = pd.read_csv(
    BASE_DIR / "outputs" / "featured_data.csv"
)

# ==================================================
# SIDEBAR
# ==================================================

page = st.sidebar.selectbox(
    "Navigation",
    [
        "Home",
        "Data Analysis",
        "Model Comparison",
        "Model Insights",
        "Feature Importance",
        "Forecasting",
        "About Project"
    ]
)

# ==================================================
# HOME PAGE
# ==================================================

if page == "Home":

    st.title("⚡ Day Ahead Load Forecasting System")

    st.markdown("---")

    st.subheader("Project Overview")

    st.write("""
    This project predicts future electricity demand using
    Machine Learning and historical load data.

    The system compares multiple machine learning models
    and generates a 24-hour demand forecast.
    """)

    st.subheader("Models Used")

    st.write("""
    • Linear Regression

    • Random Forest

    • XGBoost

    • LightGBM

    • CatBoost
    """)

    st.subheader("Features Used")

    st.write("""
    • Hour

    • Day

    • Month

    • Weekday

    • Lag Features

    • Rolling Mean Features
    """)
    
# ==================================================
# DATA ANALYSIS PAGE
# ==================================================

elif page == "Data Analysis":

    st.title("📊 Data Analysis")

    st.subheader("Hourly Demand Trend")

    st.image(
        BASE_DIR / "assets" / "hourly_trend.png",
        use_container_width=True
    )

    st.subheader("Monthly Demand Trend")

    st.image(
        BASE_DIR / "assets" / "monthly_trend.png",
        use_container_width=True
    )    

# ==================================================
# MODEL COMPARISON PAGE
# ==================================================

elif page == "Model Comparison":

    st.title("📊 Model Comparison")

    results = pd.read_csv(
        BASE_DIR / "outputs" / "model_results.csv"
    )

    st.subheader("Performance Metrics")

    st.dataframe(results)

    best_model = results.loc[
        results["R2"].idxmax()
    ]

    col1, col2 = st.columns(2)

    col1.metric(
        "🏆 Best Model",
        best_model["Model"]
    )

    col2.metric(
        "Best R² Score",
        f"{best_model['R2']:.4f}"
    )

    st.subheader("R² Comparison")

    st.bar_chart(
        results.set_index("Model")["R2"]
    )

    st.subheader("MAE Comparison")

    st.bar_chart(
        results.set_index("Model")["MAE"]
    )

    st.subheader("RMSE Comparison")

    st.bar_chart(
        results.set_index("Model")["RMSE"]
    )
    
# ==================================================
# MODEL INSIGHTS PAGE
# ==================================================

elif page == "Model Insights":

    st.title("🧠 Model Insights")

    st.subheader("Actual vs Predicted")

    st.image(
        BASE_DIR / "assets" / "actual_vs_predicted.png",
        use_container_width=True
    )

    st.subheader("Residual Analysis")

    st.image(
        BASE_DIR / "assets" / "residual_analysis.png",
        use_container_width=True
    )

    st.markdown("""
    ### Interpretation

    - Actual vs Predicted graph shows how closely the model predictions
      match the real electricity demand.

    - Residual Analysis shows the prediction errors.

    - Residuals centered around zero indicate good model performance.

    - Linear Regression achieved the highest R² score among all tested models.
    """) 
    
 
# ==================================================
# FEATURE IMPORTANCE PAGE
# ==================================================

elif page == "Feature Importance":

    st.title("📌 Feature Importance")

    st.image(
        BASE_DIR / "assets" / "feature_importance.png",
        use_container_width=True
    )

    st.markdown("""
    ### Key Findings

    - Hour is the most influential feature.

    - Previous hour demand (lag_1) strongly affects current demand.

    - Month captures seasonal patterns.

    - Previous day demand (lag_24) helps improve forecasting accuracy.
    """)       

# ==================================================
# FORECASTING PAGE
# ==================================================

elif page == "Forecasting":

    st.title("📈 24-Hour Load Forecast")

    selected_date = st.date_input(
        "Select Forecast Date"
    )

    if st.button("Generate Forecast"):

        future_df = df.tail(24).copy()

        future_df["day"] = selected_date.day
        future_df["month"] = selected_date.month
        future_df["weekday"] = selected_date.weekday()

        features = [
            "hour",
            "day",
            "month",
            "weekday",
            "lag_1",
            "lag_24",
            "lag_48",
            "lag_72",
            "lag_168",
            "rolling_mean_24",
            "rolling_mean_168"
        ]

        predictions = model.predict(
            future_df[features]
        )

        forecast_df = pd.DataFrame({
            "Hour": range(24),
            "Predicted Demand (MW)": predictions
        })

        st.subheader("Forecast Table")

        st.dataframe(forecast_df)

        csv = forecast_df.to_csv(index=False)

        st.download_button(
            label="📥 Download Forecast CSV",
            data=csv,
            file_name="forecast.csv",
            mime="text/csv"
        )

        col1, col2, col3 = st.columns(3)

        col1.metric(
            "Peak Demand",
            f"{forecast_df['Predicted Demand (MW)'].max():,.0f}"
        )

        col2.metric(
            "Minimum Demand",
            f"{forecast_df['Predicted Demand (MW)'].min():,.0f}"
        )

        col3.metric(
            "Average Demand",
            f"{forecast_df['Predicted Demand (MW)'].mean():,.0f}"
        )

        fig, ax = plt.subplots(figsize=(10, 5))

        ax.plot(
            forecast_df["Hour"],
            forecast_df["Predicted Demand (MW)"],
            marker="o"
        )

        ax.set_title(
            f"24-Hour Forecast ({selected_date})"
        )

        ax.set_xlabel("Hour")
        ax.set_ylabel("Demand (MW)")
        ax.grid(True)

        st.pyplot(fig)

# ==================================================
# ABOUT PROJECT PAGE
# ==================================================

elif page == "About Project":

    st.title("📘 About Project")

    st.write("""
    ### Day Ahead Load Forecasting System

    This project predicts future electricity demand
    using historical hourly load data.

    ### Technologies Used

    - Python
    - Pandas
    - Scikit-Learn
    - Streamlit
    - XGBoost
    - LightGBM
    - CatBoost

    ### Dataset

    - 46,728 hourly records
    - National Load Demand
    - Regional Load Demand

    ### Models Compared

    - Linear Regression
    - Random Forest
    - XGBoost
    - LightGBM
    - CatBoost

    ### Objective

    Forecast next-day electricity demand
    and compare machine learning models.
    """)