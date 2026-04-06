# Bank Customer Churn Prediction 🏦

## Overview
This project is an end-to-end Machine Learning pipeline built to predict whether a customer will leave a bank (churn) based on their demographics and account information. 

## The Dataset
The data is sourced from a standard Bank Customer Churn dataset. It includes details such as:
* Credit Score
* Age
* Account Balance
* Number of Products
* Gender 

## Pipeline Steps
1. **Data Ingestion:** Automatically pulls the raw CSV data directly from a public repository using Pandas.
2. **Data Cleaning & Preprocessing:**
   * Dropped irrelevant features (`RowNumber`, `CustomerId`, `Surname`).
   * Handled missing values by imputing the median for `Age` and `Balance`.
   * Encoded categorical text data (`Gender`) into machine-readable binary values (0 and 1).
3. **Data Splitting:** Separated the dataset into an 80% training set and a 20% testing set to prevent data leakage and ensure accurate evaluation.
4. **Model Training:** Initialized and trained a **Logistic Regression** classification model to identify patterns in customer churn.
5. **Evaluation:** Tested the model against unseen data.

## Results
* **Accuracy:** ~80.00% 
The model successfully learned the underlying patterns of customer retention and can predict churn on unseen test data with 80% accuracy.

## Tech Stack
* Python
* Pandas (Data manipulation)
* Scikit-Learn (Machine learning & evaluation)