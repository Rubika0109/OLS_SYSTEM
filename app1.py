import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load trained model
@st.cache_resource
def load_model():
    with open("model.pkl", "rb") as f:
        model = pickle.load(f)
    return model

model = load_model()

# App title
st.title("🧬 Cancer Prediction System")
st.write("This app uses a regression model trained on the **Nripper Cancer Regression Challenge** dataset to predict cancer outcomes.")

# Input form
st.header("Enter Patient Data")

# Replace with actual features used in your model
feature_names = ['mean_radius', 'mean_texture', 'mean_perimeter', 'mean_area', 'mean_smoothness']
input_data = {}

for feature in feature_names:
    input_data[feature] = st.number_input(f"{feature.replace('_', ' ').title()}:", min_value=0.0)

if st.button("Predict"):
    # Convert input to DataFrame
    input_df = pd.DataFrame([input_data])
    
    # Make prediction
    prediction = model.predict(input_df)[0]
    
    # Show prediction
    st.subheader("Prediction Result")
    st.success(f"The predicted cancer score is: {prediction:.2f}")

