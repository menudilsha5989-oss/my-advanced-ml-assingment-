"""
Streamlit app for the COM763 churn-prediction system.
Run locally with: streamlit run app.py
Deploy on Streamlit Community Cloud by connecting this file's GitHub repo.
"""
import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Telecom Churn Predictor", layout="wide")

model = joblib.load("rf_model.joblib")
scaler = joblib.load("scaler.joblib")
feature_columns = joblib.load("feature_columns.joblib")

st.title("Telecom Customer Churn Predictor")
st.caption("Random Forest model trained on customer account and service data.")

with st.sidebar:
    st.header("Customer Inputs")
    contract = st.selectbox("Contract type", ["Month-to-month", "One year", "Two year"])
    tenure = st.slider("Tenure (months)", 0, 72, 4)
    internet = st.selectbox("Internet service", ["DSL", "Fiber optic", "No"])
    tech_support = st.selectbox("Tech support", ["Yes", "No", "No internet service"])
    monthly_charges = st.number_input("Monthly charges ($)", 18.0, 120.0, 89.5)
    payment = st.selectbox("Payment method",
                            ["Electronic check", "Mailed check", "Bank transfer", "Credit card"])
    predict_btn = st.button("Predict Churn")

if predict_btn:
    raw = pd.DataFrame([{
        "SeniorCitizen": 0, "Partner": "No", "Dependents": "No",
        "tenure": tenure, "Contract": contract, "InternetService": internet,
        "TechSupport": tech_support, "OnlineSecurity": "No",
        "PaymentMethod": payment, "PaperlessBilling": "Yes",
        "NumServices": 3, "MonthlyCharges": monthly_charges,
        "TotalCharges": monthly_charges * max(tenure, 1),
    }])
    X = pd.get_dummies(raw).reindex(columns=feature_columns, fill_value=0)
    num_cols = ["SeniorCitizen", "tenure", "NumServices", "MonthlyCharges", "TotalCharges"]
    X[num_cols] = scaler.transform(X[num_cols])

    prob = model.predict_proba(X)[0, 1]
    st.subheader("Prediction")
    if prob >= 0.5:
        st.error(f"High risk of churn — estimated probability {prob:.0%}")
    else:
        st.success(f"Low risk of churn — estimated probability {prob:.0%}")
