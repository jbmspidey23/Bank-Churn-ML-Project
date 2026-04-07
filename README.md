# Bank Customer Churn Prediction 🏦

## Overview
This project is an end-to-end Machine Learning pipeline and interactive web application built to predict whether a bank customer will churn (leave the bank) based on their demographics and account information. 

## Project Architecture
This project has been upgraded from a procedural script to a production-ready architecture:
1. **Object-Oriented Pipeline (`oopnew.py`):** The data ingestion, preprocessing, and model training steps are encapsulated within a custom Python Class for reusability.
2. **Model Persistence:** The trained Logistic Regression model is exported as a `.pkl` file using `joblib`, preventing the need to retrain the model on every run.
3. **Web Application (`app.py`):** A frontend interface built with Streamlit allows users to input new customer data and receive instant churn predictions.

## The Dataset
The model was trained on a standard Bank Customer Churn dataset, utilizing features such as:
* Credit Score
* Age
* Account Balance
* Number of Products
* Gender 

## Results
* **Accuracy:** ~80.00% 
The model successfully learned the underlying patterns of customer retention and predicts churn on unseen data with 80% accuracy.

## Tech Stack
* **Language:** Python
* **Data Manipulation:** Pandas
* **Machine Learning:** Scikit-Learn
* **Web Framework:** Streamlit
* **Serialization:** Joblib

## How to Run Locally
1. Clone the repository to your local machine.
2. Install the required dependencies:
   `pip install -r requirements.txt`
3. Start the Streamlit server:
   `streamlit run app.py`
## The actual link to the streamline app 
* **https://bank-churn-ml-project-nh76mpvb6qcisxtsb3esdj.streamlit.app/
