import streamlit as st
import numpy as np
import joblib

#Load the trained model

model = joblib.load('WQP_regression_model.pkl')

# Main app
def main():
    st.title("Wine Quality Prediction App")
    st.write('''this app predicts health risks based on user imput.Please provide the following details:''')

    # User input for prediction
fixed_acidity = st.number_input('Fixed Acidity', min_value=0.0, max_value=20.0, value=8.0, step=0.1)
volatile_acidity = st.number_input('Volatile Acidity', min_value=0.0, max_value=50.0, value=0.5, step=0.01)
citric_acid = st.number_input('Citric Acid', min_value=0.0, max_value=50.0, value=0.0, step=0.01)
residual_sugar = st.number_input('Residual Sugar', min_value=0.0, max_value=500.0, value=5.0, step=0.1)
chlorides = st.number_input('Chlorides', min_value=0.0, max_value=50.0, value=0.05, step=0.01)
free_sulfur_dioxide = st.number_input('Free Sulfur Dioxide', min_value=0.0, max_value=100.0, value=15.0, step=1.0)
total_sulfur_dioxide = st.number_input('Total Sulfur Dioxide', min_value=0.0, max_value=300.0, value=50.0, step=1.0)
density = st.number_input('Density', min_value=0.990, max_value=10.1, value=0.995, step=0.001)
ph = st.number_input('pH', min_value=0.0, max_value=50.0, value=3.0, step=0.1)
sulphates = st.number_input('Sulphates', min_value=0.0, max_value=50.0, value=0.5, step=0.01)
alcohol = st.number_input('Alcohol', min_value=0.0, max_value=50.0, value=10.0, step=0.1)

inputs = np.array([[fixed_acidity, volatile_acidity, citric_acid, residual_sugar, chlorides, free_sulfur_dioxide,
                    total_sulfur_dioxide, density, ph, sulphates,alcohol]])
    
#Prediction
    # Make prediction
if st.button("Predict Quality"):
    prediction=model.predict(inputs)
    risk= 'High Quality'if prediction[0]==1 else 'Low quality'
    st.write(f'prediction:**{risk}**')


