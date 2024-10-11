import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import numpy as np
import streamlit_option_menu

# Page Configuration
st.set_page_config(page_title="Credit Risk Prediction App", page_icon="💳", layout="wide")

# Custom CSS styling
st.markdown(
    """
    <style>
    .main {
        background-color: #f5f5f5;
    }
    .sidebar .sidebar-content {
        background-color: #f0f2f6;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
    }
    </style>
    """, 
    unsafe_allow_html=True
)

# Sidebar - App Information
st.sidebar.title("ℹ️ About the App")
st.sidebar.write("""
This app uses a **Logistic Regression Model** to predict if a loan is likely to default based on several input features such as income, employment length, and more.
- Upload your data, or input custom values to get predictions.
- See model evaluation metrics such as accuracy, confusion matrix, and classification report.
""")

# Sidebar - Sample Input
st.sidebar.header("📊 Sample Input")
st.sidebar.write("""
Enter values on the left to predict whether the loan is likely to default or not based on the trained Logistic Regression model.
""")

# Title of the app
st.title('💳 Credit Risk Prediction App')
st.markdown("**Predict whether a loan is likely to default based on customer and loan details.**")


# Sidebar image (optional for aesthetics)
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/681/681443.png", use_column_width=True)

# Model Training Section
st.header("🔢 Loan Default Risk Prediction")
st.write("### Please enter the following details to predict if the loan will default:")

# Create input fields for user input
col1, col2 = st.columns(2)

with col1:
    person_age = st.number_input('Person Age', min_value=18, max_value=100, value=25)
    person_income = st.number_input('Person Income', min_value=0, step=1000, value=50000)
    person_home_ownership = st.selectbox('Person Home Ownership', options=[1, 2, 3, 4], format_func=lambda x: {1: 'Rent', 2: 'Mortgage', 3: 'Own', 4: 'Other'}[x])
    person_emp_length = st.number_input('Employment Length (months)', min_value=0, step=1, value=12)
    cb_person_cred_hist_length = st.number_input('Credit History Length (years)', min_value=0, step=1, value=5)

with col2:
    loan_intent = st.selectbox('Loan Intent', options=[1, 2, 3, 4, 5, 6], format_func=lambda x: {1: 'Education', 2: 'Medical', 3: 'Venture', 4: 'Personal', 5: 'Debt Consolidation', 6: 'Home Improvement'}[x])
    loan_grade = st.selectbox('Loan Grade', options=[1, 2, 3, 4, 5, 6, 7], format_func=lambda x: {1: 'A', 2: 'B', 3: 'C', 4: 'D', 5: 'E', 6: 'F', 7: 'G'}[x])
    loan_int_rate = st.number_input('Loan Interest Rate (%)', min_value=0.0, step=0.01, value=10.0)
    loan_percent_income = st.number_input('Loan Percent Income', min_value=0.0, step=0.01, value=0.1)
    cb_person_default_on_file = st.selectbox('Default on File', options=[0, 1], format_func=lambda x: {0: 'No', 1: 'Yes'}[x])

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
    model = joblib.load('Credit_risk.model')

    # Predict the result
    prediction = model.predict(input_data)[0]

    # Display the result with color coding
    if prediction == 1:
        st.markdown("<h3 style='color: red;'>🚨 The loan is likely to default.</h3>", unsafe_allow_html=True)
    else:
        st.markdown("<h3 style='color: green;'>✅ The loan is unlikely to default.</h3>", unsafe_allow_html=True)
    
