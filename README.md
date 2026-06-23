# ⚡ Electricity Load Forecasting Using Machine Learning

## 📌 Project Overview

This project focuses on forecasting electricity load demand using Machine Learning techniques. Accurate load forecasting helps power utilities optimize energy generation, distribution, and resource planning.

The project evaluates multiple machine learning models and provides an interactive Streamlit web application for predicting future electricity demand.

---

## 🎯 Objectives

- Analyze historical electricity consumption data.
- Perform data preprocessing and feature engineering.
- Train and compare multiple machine learning models.
- Evaluate model performance using regression metrics.
- Deploy predictions through an interactive Streamlit application.

---

## 🛠 Technologies Used

### Programming Language
- Python

### Libraries
- Pandas
- NumPy
- Scikit-Learn
- XGBoost
- LightGBM
- Matplotlib
- Seaborn
- Joblib

### Framework
- Streamlit

---

## 📂 Project Structure

```text
Load-Forecasting/
│
├── app.py
├── requirements.txt
├── dataset.csv
├── models/
│   ├── linear_regression.pkl
│   ├── random_forest.pkl
│   ├── gradient_boosting.pkl
│   ├── xgboost.pkl
│   └── lightgbm.pkl
│
├── screenshots/
├── notebooks/
├── report/
└── README.md
```

---

## 🔄 Methodology

### 1. Data Collection
Historical electricity load data is collected and stored in CSV format.

### 2. Data Preprocessing
- Missing value handling
- Feature extraction
- Date and time transformation
- Data normalization (if required)

### 3. Model Training
The following regression algorithms were trained and evaluated:

- Linear Regression
- Random Forest Regressor
- Gradient Boosting Regressor
- XGBoost Regressor
- LightGBM Regressor

### 4. Model Evaluation

Performance was measured using:

- Mean Absolute Error (MAE)
- Mean Squared Error (MSE)
- Root Mean Squared Error (RMSE)
- R² Score

### 5. Deployment

The best-performing model was integrated into a Streamlit web application for real-time prediction.

---

## 🤖 Machine Learning Models Used

### Linear Regression
A basic regression model used as a benchmark for comparison.

### Random Forest Regressor
An ensemble learning algorithm that combines multiple decision trees for improved accuracy.

### Gradient Boosting Regressor
Builds models sequentially to minimize prediction errors.

### XGBoost Regressor
An optimized gradient boosting algorithm known for high performance and efficiency.

### LightGBM Regressor
A fast and scalable gradient boosting framework designed for large datasets.

---

## 📊 Features

✅ Historical Load Analysis

✅ Data Visualization

✅ Model Comparison

✅ Load Prediction

✅ Interactive Streamlit Dashboard

✅ Performance Evaluation Metrics

---

## 🚀 Installation

### Clone Repository

```bash
git clone https://github.com/pratapdeepanshu8-a11y/Load-Forecasting.git
```

### Move into Project Directory

```bash
cd Load-Forecasting
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Run Streamlit Application

```bash
streamlit run app.py
```

---

## 📈 Evaluation Metrics

| Metric | Description |
|----------|-------------|
| MAE | Mean Absolute Error |
| MSE | Mean Squared Error |
| RMSE | Root Mean Squared Error |
| R² Score | Coefficient of Determination |

---

## 📷 Application Screenshots

Add screenshots of:

- Home Page
- Data Visualization Dashboard
- Model Training Results
- Prediction Page
- Performance Comparison Graphs

inside the `screenshots` folder.

---

## 🔮 Future Scope

- Integration with real-time electricity consumption data.
- Deep Learning models such as LSTM and GRU.
- Weather-based load forecasting.
- Cloud deployment.
- Automated model retraining.

---

## 👨‍💻 Author

**Deepanshu Pratap Singh**

B.Tech (Information Technology)

Project: Electricity Load Forecasting Using Machine Learning

---

## 📜 License

This project is developed for educational and research purposes.
requirements.txt

streamlit
pandas
numpy
scikit-learn
xgboost
lightgbm
matplotlib
seaborn
joblib

.gitignore

venv/
__pycache__/
.ipynb_checkpoints/
*.pyc
*.pkl
.DS_Store
