import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt

# Load the pre-trained model
model_path = "best_random_forest_model.pkl"  # Updated model filename
with open(model_path, "rb") as file:
    model = pickle.load(file)

# App title
st.title("Advertising Sales Prediction")

# Input fields for advertising budgets
st.header("Input Advertising Budgets")
tv_budget = st.number_input("TV Budget ($)", min_value=0.0, step=1.0)
radio_budget = st.number_input("Radio Budget ($)", min_value=0.0, step=1.0)
newspaper_budget = st.number_input("Newspaper Budget ($)", min_value=0.0, step=1.0)

# Predict sales
if st.button("Predict Sales"):
    input_data = np.array([[tv_budget, radio_budget, newspaper_budget]])
    predicted_sales = model.predict(input_data)[0]
    st.subheader("Predicted Sales")
    st.write(f"${predicted_sales:.2f}")
    
    # Visualization: Line plot from 0 to predicted sales
    st.subheader("Sales Prediction Visualization")
    sales_range = np.linspace(0, predicted_sales, 100)
    fig, ax = plt.subplots()
    ax.plot(sales_range, label="Predicted Sales", color="blue")
    ax.fill_between(sales_range, 0, sales_range, color="blue", alpha=0.2)
    ax.set_title("Sales Prediction Line Plot")
    ax.set_xlabel("Steps")
    ax.set_ylabel("Sales ($)")
    ax.legend()
    st.pyplot(fig)
else:
    st.write("Enter advertising budgets above and click 'Predict Sales' to see the results.")
