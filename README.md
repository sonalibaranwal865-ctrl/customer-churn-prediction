# Customer Churn Prediction System

A beginner-friendly end-to-end machine learning project suitable for learning Python, EDA, feature engineering, classification, SQL, and simple application development.

## Problem statement
Customer churn occurs when a customer stops using a service. This project explores customer data and trains a classification model to estimate whether a customer is likely to churn.

> Important: This is a learning/demo project. Predictions are not guaranteed to be accurate or suitable for real business decisions.

## Current project scope
- Data loading and basic validation
- Exploratory data analysis (EDA)
- Feature preprocessing using a scikit-learn pipeline
- Logistic Regression and Decision Tree comparison
- Evaluation using precision, recall, F1-score, confusion matrix, and ROC-AUC when available
- Saving the trained model
- Streamlit demo interface
- SQL practice queries

## Dataset
Recommended dataset: IBM Telco Customer Churn (commonly available from IBM's sample datasets and Kaggle mirrors).
Download a CSV and save it as `data/WA_Fn-UseC_-Telco-Customer-Churn.csv`.

This repository intentionally does not include a third-party dataset. Check its license and source terms before redistributing it.

Expected target column: `Churn` with values such as `Yes` and `No`.

## Setup (Windows / VS Code)
1. Install Python 3.10+.
2. Open this folder in VS Code.
3. In the terminal, create a virtual environment:
   ```bash
   python -m venv .venv
   ```
4. Activate it:
   ```powershell
   .venv\Scripts\Activate.ps1
   ```
   If activation is blocked, use the VS Code Python interpreter selector or the equivalent activation command for your shell.
5. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
6. Place the dataset at the path above.
7. Run EDA:
   ```bash
   python src/01_eda.py
   ```
8. Train models:
   ```bash
   python src/02_train.py
   ```
9. Launch the demo:
   ```bash
   streamlit run app.py
   ```

## Files
- `src/01_eda.py` — data checks and visualizations
- `src/02_train.py` — preprocessing, model training, evaluation, and model saving
- `app.py` — simple Streamlit demo
- `sql/analysis_queries.sql` — SQL practice queries
- `reports/` — generated EDA plots and metrics
- `models/` — trained model artifact (generated locally)

## What to discuss in an interview
1. Why is churn prediction a classification problem?
2. Why should preprocessing be fitted only on the training data?
3. Why might accuracy be misleading for imbalanced classes?
4. Why are precision and recall useful for churn?
5. How did you avoid data leakage?
6. What would you improve before deploying to a real company?

## Suggested improvements
- Use cross-validation and tune model hyperparameters.
- Add a threshold slider and explain the precision/recall trade-off.
- Connect a MySQL database and load records using SQLAlchemy.
- Add tests, logging, and a clear model card.
- Compare a simple baseline with more advanced models.
