import streamlit as st
import pickle
import numpy as np
import pandas as pd

# Load model
with open("customer_churn_model.pkl", "rb") as f:
    model_data = pickle.load(f)

weights = model_data["weights"]
bias = model_data["bias"]
scaler = model_data["scaler"]
columns = model_data["columns"]

# Functions
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def predict(X):
    linear = np.dot(X, weights) + bias
    probs = sigmoid(linear)
    return probs, [1 if p > 0.5 else 0 for p in probs]

# UI
st.title("Customer Churn Prediction 🔍")

# Basic Inputs
tenure = st.number_input("Tenure (months)", 0, 100, 1)
monthly_charges = st.number_input("Monthly Charges", 0.0, 200.0, 50.0)
total_charges = st.number_input("Total Charges", 0.0, 10000.0, 100.0)

# Categorical Inputs
contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
internet = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
payment = st.selectbox("Payment Method", [
    "Electronic check",
    "Mailed check",
    "Bank transfer (automatic)",
    "Credit card (automatic)"
])
senior = st.selectbox("Senior Citizen", ["No", "Yes"])
partner = st.selectbox("Has Partner", ["No", "Yes"])
tech = st.selectbox("Tech Support", ["No", "Yes"])

# Create input dictionary
input_data = pd.DataFrame([{
    "tenure": tenure,
    "MonthlyCharges": monthly_charges,
    "TotalCharges": total_charges,
    "SeniorCitizen": 1 if senior == "Yes" else 0
}])

# Encoding Contract
input_data["Contract_One year"] = 1 if contract == "One year" else 0
input_data["Contract_Two year"] = 1 if contract == "Two year" else 0

# Encoding Internet
input_data["InternetService_Fiber optic"] = 1 if internet == "Fiber optic" else 0
input_data["InternetService_No"] = 1 if internet == "No" else 0

# Encoding Payment
input_data["PaymentMethod_Credit card (automatic)"] = 1 if payment == "Credit card (automatic)" else 0
input_data["PaymentMethod_Electronic check"] = 1 if payment == "Electronic check" else 0
input_data["PaymentMethod_Mailed check"] = 1 if payment == "Mailed check" else 0

# Encoding Partner & Tech Support
input_data["Partner_Yes"] = 1 if partner == "Yes" else 0
input_data["TechSupport_Yes"] = 1 if tech == "Yes" else 0

# Align columns
input_data = input_data.reindex(columns=columns, fill_value=0)

# Scale input
input_scaled = scaler.transform(input_data)

# Predict
if st.button("Predict"):
    probs, result = predict(input_scaled)

    st.write(f"Churn Probability: {probs[0]:.2f}")

    if result[0] == 1:
        st.error("⚠️ Customer will LEAVE")
    else:
        st.success("✅ Customer will STAY")