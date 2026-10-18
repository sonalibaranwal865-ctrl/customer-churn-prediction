"""Streamlit demo for the trained churn prediction pipeline."""
from pathlib import Path
import joblib
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "churn_pipeline.joblib"

st.set_page_config(page_title="Customer Churn Prediction", page_icon="📊", layout="centered")
st.title("Customer Churn Prediction")
st.write(
    "A learning demo that estimates churn from customer details. "
    "It is not a guarantee and should not be used as the sole basis for real customer decisions."
)

if not MODEL_PATH.exists():
    st.warning("Model file not found. First add the dataset and run: python src/02_train.py")
    st.stop()

model = joblib.load(MODEL_PATH)

with st.form("customer_form"):
    st.subheader("Customer details")
    gender = st.selectbox("Gender", ["Female", "Male"])
    senior = st.selectbox("Senior citizen?", [0, 1], format_func=lambda x: "Yes" if x else "No")
    partner = st.selectbox("Has partner?", ["Yes", "No"])
    dependents = st.selectbox("Has dependents?", ["Yes", "No"])
    tenure = st.slider("Tenure (months)", 0, 72, 12)
    phone = st.selectbox("Phone service", ["Yes", "No"])
    multiple = st.selectbox("Multiple lines", ["No phone service", "No", "Yes"])
    internet = st.selectbox("Internet service", ["DSL", "Fiber optic", "No"])
    online_security = st.selectbox("Online security", ["No internet service", "No", "Yes"])
    online_backup = st.selectbox("Online backup", ["No internet service", "No", "Yes"])
    device_protection = st.selectbox("Device protection", ["No internet service", "No", "Yes"])
    tech_support = st.selectbox("Tech support", ["No internet service", "No", "Yes"])
    streaming_tv = st.selectbox("Streaming TV", ["No internet service", "No", "Yes"])
    streaming_movies = st.selectbox("Streaming movies", ["No internet service", "No", "Yes"])
    contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
    paperless = st.selectbox("Paperless billing", ["Yes", "No"])
    payment = st.selectbox("Payment method", [
        "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"
    ])
    monthly = st.number_input("Monthly charges", min_value=0.0, max_value=500.0, value=70.0)
    total = st.number_input("Total charges", min_value=0.0, max_value=100000.0, value=840.0)
    submit = st.form_submit_button("Predict churn")

if submit:
    row = pd.DataFrame([{
        "gender": gender, "SeniorCitizen": senior, "Partner": partner,
        "Dependents": dependents, "tenure": tenure, "PhoneService": phone,
        "MultipleLines": multiple, "InternetService": internet,
        "OnlineSecurity": online_security, "OnlineBackup": online_backup,
        "DeviceProtection": device_protection, "TechSupport": tech_support,
        "StreamingTV": streaming_tv, "StreamingMovies": streaming_movies,
        "Contract": contract, "PaperlessBilling": paperless,
        "PaymentMethod": payment, "MonthlyCharges": monthly, "TotalCharges": total
    }])
    probability = float(model.predict_proba(row)[0, 1])
    prediction = int(probability >= 0.5)
    st.subheader("Prediction")
    st.write("Likely to churn" if prediction else "Not predicted to churn")
    st.metric("Estimated churn probability", f"{probability:.1%}")
    st.caption("The 50% threshold is a simple demo default, not a business-validated threshold.")
