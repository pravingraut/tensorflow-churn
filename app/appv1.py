import streamlit as st
import pandas as pd
import requests

st.set_page_config(page_title="Customer Churn Predictor", page_icon="🤖", layout="centered")

st.title("🧠 Customer Churn Prediction App")
st.write("Enter customer details below to predict churn probability.")

# --- Customer Info ---
st.subheader("Customer Info")
col1, col2, col3, col4 = st.columns(4)
gender = col1.selectbox("Gender", ["Female", "Male"])
SeniorCitizen = col2.selectbox("Senior Citizen", [0, 1])
Partner = col3.selectbox("Partner", ["Yes", "No"])
Dependents = col4.selectbox("Dependents", ["Yes", "No"])

# --- Services ---
st.subheader("Services")
col1, col2, col3 = st.columns(3)
PhoneService = col1.selectbox("Phone Service", ["Yes", "No"])
MultipleLines = col2.selectbox("Multiple Lines", ["No phone service", "No", "Yes"])
InternetService = col3.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])

# --- Internet Features ---
st.subheader("Internet Features")
col1, col2, col3 = st.columns(3)
OnlineSecurity = col1.selectbox("Online Security", ["Yes", "No", "No internet service"])
OnlineBackup = col2.selectbox("Online Backup", ["Yes", "No", "No internet service"])
DeviceProtection = col3.selectbox("Device Protection", ["Yes", "No", "No internet service"])

col1, col2, col3 = st.columns(3)
TechSupport = col1.selectbox("Tech Support", ["Yes", "No", "No internet service"])
StreamingTV = col2.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
StreamingMovies = col3.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])

# --- Contract & Billing ---
st.subheader("Contract & Billing")
col1, col2, col3 = st.columns(3)
Contract = col1.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
PaperlessBilling = col2.selectbox("Paperless Billing", ["Yes", "No"])
PaymentMethod = col3.selectbox("Payment Method", ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])

# --- Numerical Inputs ---
st.subheader("Numerical Inputs")
col1, col2, col3 = st.columns(3)
tenure = col1.number_input("Tenure (months)", min_value=0, max_value=72, value=18, step=1)
col1.caption("Min: 0, Max: 72, Mean: 32.37")

MonthlyCharges = col2.number_input("Monthly Charges", min_value=18.0, max_value=118.0, value=64.76, step=0.1)
col2.caption("Min: 18, Max: 118, Mean: 64.76")

TotalCharges = col3.number_input("Total Charges", min_value=0.0, max_value=8684.0, value=2279.73, step=1.0)
col3.caption("Min: 0, Max: 8684, Mean: 2279.73")

# --- Predict Button ---
if st.button("Predict"):
    input_data = {
        "gender": gender,
        "SeniorCitizen": SeniorCitizen,
        "Partner": Partner,
        "Dependents": Dependents,
        "tenure": tenure,
        "PhoneService": PhoneService,
        "MultipleLines": MultipleLines,
        "InternetService": InternetService,
        "OnlineSecurity": OnlineSecurity,
        "OnlineBackup": OnlineBackup,
        "DeviceProtection": DeviceProtection,
        "TechSupport": TechSupport,
        "StreamingTV": StreamingTV,
        "StreamingMovies": StreamingMovies,
        "Contract": Contract,
        "PaperlessBilling": PaperlessBilling,
        "PaymentMethod": PaymentMethod,
        "MonthlyCharges": MonthlyCharges,
        "TotalCharges": TotalCharges
    }

    try:
        response = requests.post("http://127.0.0.1:8000/predict", json=input_data)
        result = response.json()
        prob = result['churn_probability']
        will_churn = result['will_churn']

        # Display probability
        st.info(f"Churn Probability: {prob:.2f}")

        # Display result with colored background
        if will_churn:
            st.markdown(
                f"<div style='background-color:red; padding:10px; color:white; font-weight:bold;'>"
                f"Customer will CHURN</div>", unsafe_allow_html=True
            )
        else:
            st.markdown(
                f"<div style='background-color:green; padding:10px; color:white; font-weight:bold;'>"
                f"Customer will NOT CHURN</div>", unsafe_allow_html=True
            )

    except Exception as e:
        st.error(f"API error: {e}. Make sure the FastAPI server is running.")

st.write("---")
st.caption("Made with ❤️ using TensorFlow, FastAPI, and Streamlit.")