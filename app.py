import streamlit as st
import pandas as pd
import joblib


model = joblib.load("bank_brain.pkl")


st.title("Bank Customer Churn Predictor 🏦")
st.write("Enter the customer's details below to predict if they will leave the bank.")


gender = st.selectbox("Gender", ["Male", "Female"])
age = st.number_input("Age", min_value=18, max_value=100, value=30)
balance = st.number_input("Account Balance ($)", min_value=0.0, value=50000.0)
products = st.number_input("Number of Products", min_value=1, max_value=4, value=1)
credit_score = st.number_input("Credit Score", min_value=300, max_value=850, value=650)


if st.button("Predict Churn"):


    gender_binary = 1 if gender == "Female" else 0


    input_data = pd.DataFrame([[gender_binary, age, balance, products, credit_score]],
                              columns=['Gender', 'Age', 'Balance', 'NumOfProducts', 'CreditScore'])


    prediction = model.predict(input_data)


    if prediction[0] == 1:
        st.error(" High Risk: This customer is predicted to CHURN ie Leave haha ")
    else:
        st.success(" Safe: This customer is predicted to STAY duhhh.")