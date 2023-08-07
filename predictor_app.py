import streamlit as st
import pandas as pd
import numpy as np
from xgboost import XGBClassifier

# Load the trained XGBoost model
loaded_model = XGBClassifier()
loaded_model.load_model('xgb_model.bin')

# Function to make predictions
def predict(type, amount, oldbalanceOrg, newbalanceOrig):
    # Map 'type' to numeric values
    type_mapping = {"CASH_OUT": 1, "PAYMENT": 2, "CASH_IN": 3, "TRANSFER": 4, "DEBIT": 5}
    type = type_mapping.get(type, 0)

    # Prepare input data for prediction
    input_data = pd.DataFrame({
        'type': [type],
        'amount': [amount],
        'oldbalanceOrg': [oldbalanceOrg],
        'newbalanceOrig': [newbalanceOrig]
    })

    # Make a prediction using the loaded model
    prediction = loaded_model.predict(input_data)

    # Return prediction result
    return "Fraud Detected" if prediction[0] == 1 else "No Fraud Detected"

# Streamlit app header
st.title("Online Payments Fraud Detection App")

# Sidebar
st.sidebar.header("User Input")

# Collect user input features
type = st.sidebar.selectbox("Transaction Type", ["CASH_OUT", "PAYMENT", "CASH_IN", "TRANSFER", "DEBIT"])
amount = st.sidebar.number_input("Transaction Amount", min_value=0.0)
oldbalanceOrg = st.sidebar.number_input("Old Balance Orig", min_value=0.0)
newbalanceOrig = st.sidebar.number_input("New Balance Orig", min_value=0.0)

# Button to trigger prediction
if st.sidebar.button("Predict Fraud"):
    # Call the predict function
    prediction_result = predict(type, amount, oldbalanceOrg, newbalanceOrig)

    # Display prediction result
    st.subheader("Prediction:")
    st.write(prediction_result)

# Display model feature importances
st.subheader("Feature Importances")
importance_scores = loaded_model.feature_importances_
feature_names = ['type', 'amount', 'oldbalanceOrg', 'newbalanceOrig']
importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importance_scores})
st.bar_chart(importance_df.set_index('Feature'))
