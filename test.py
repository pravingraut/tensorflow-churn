import requests
from flask import jsonify


url = "http://127.0.0.1:8000/predict"
"""
sample_data = {
    "credit_score": 600,
    "age": 40,
    "balance": 60000.0,
    "num_of_products": 2,
    "is_active_member": 1,
    "estimated_salary": 80000.0
}
"""
sample_data = {
    "gender": "Male",
    "SeniorCitizen": 0,
    "Partner": "Yes",
    "Dependents": "No",
    "tenure": 12,
    "PhoneService": "Yes",
    "MultipleLines": "No",
    "InternetService": "DSL",
    "OnlineSecurity": "No",
    "OnlineBackup": "Yes",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "No",
    "StreamingMovies": "No",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 70.5,
    "TotalCharges": 845.5
}
response = requests.post(url, json=sample_data)
print("Response:", response.json())