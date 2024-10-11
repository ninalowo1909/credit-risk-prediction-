import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import numpy as np


# Title of the app
st.title('Credit Risk Prediction App')

# Prediction section
st.write("### Predict Loan Default Risk")
st.write("Enter the following details to predict if the loan will default:")

# Create input fields for user input
person_age = st.number_input('Person Age', min_value=18, max_value=100, value=25)
person_income = st.number_input('Person Income', min_value=0, step=1000, value=50000)
person_home_ownership = st.selectbox('Person Home Ownership', options=[1, 2, 3, 4], format_func=lambda x: {1: 'Rent', 2: 'Mortgage', 3: 'Own', 4: 'Other'}[x])
person_emp_length = st.number_input('Employment Length (months)', min_value=0, step=1, value=12)
loan_intent = st.selectbox('Loan Intent', options=[1, 2, 3, 4, 5, 6], format_func=lambda x: {1: 'Education', 2: 'Medical', 3: 'Venture', 4: 'Personal', 5: 'Debt Consolidation', 6: 'Home Improvement'}[x])
loan_grade = st.selectbox('Loan Grade', options=[1, 2, 3, 4, 5, 6, 7], format_func=lambda x: {1: 'A', 2: 'B', 3: 'C', 4: 'D', 5: 'E', 6: 'F', 7: 'G'}[x])
loan_int_rate = st.number_input('Loan Interest Rate (%)', min_value=0.0, step=0.01, value=10.0)
loan_percent_income = st.number_input('Loan Percent Income', min_value=0.0, step=0.01, value=0.1)
cb_person_default_on_file = st.selectbox('Default on File', options=[0, 1], format_func=lambda x: {0: 'No', 1: 'Yes'}[x])
cb_person_cred_hist_length = st.number_input('Credit History Length (years)', min_value=0, step=1, value=5)



# Button to trigger prediction
if st.button('Predict Credit Risk'):
    # Create a DataFrame for the input data
    input_data = pd.DataFrame({
        'person_age': [person_age],
        'person_income': [person_income],
        'person_home_ownership': [person_home_ownership],
        'person_emp_length': [person_emp_length],
        'loan_intent': [loan_intent],
        'loan_grade': [loan_grade],
        'loan_int_rate': [loan_int_rate],
        'loan_percent_income': [loan_percent_income],
        'cb_person_default_on_file': [cb_person_default_on_file],
        'cb_person_cred_hist_length': [cb_person_cred_hist_length]
    })


    # Load the saved model
    model = joblib.load('credit_risk_model.pkl')

    # Predict the result
    prediction = model.predict(input_data)[0]

    # Display the result
    if prediction == 1:
        st.write("### Prediction: The loan is likely to default.")
    else:
        st.write("### Prediction: The loan is unlikely to default.")
