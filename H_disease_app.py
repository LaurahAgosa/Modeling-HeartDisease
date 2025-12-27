import streamlit as st
import pickle as pkl
import numpy as np
import pandas as pd

# ---------------------------
# Load the trained model & scaler
# ---------------------------
with open('heart_disease.pkl', 'rb') as model_file:
    model = pkl.load(model_file)

with open('scaler.pkl', 'rb') as scaler_file:
    scaler = pkl.load(scaler_file)

# ---------------------------
# App Title
# ---------------------------
st.title("❤️ Heart Disease Prediction (LogisticRegression Model)")

# ---------------------------
# --- load dataset for display ---
@st.cache_data
def load_data():
    path = r"C:\Users\hp\Desktop\DS_25\Heart_Disease_Project\heart.csv"
    data = pd.read_csv(path)  
    return data

# Load and display
data = load_data()     # calling the function 
st.dataframe(data.head())   #converting the data into a pandas dataframe and reading the first rows


st.write("Enter patient clinical details below to estimate heart disease risk.")

# ---------------------------
# Input Fields
# ---------------------------
age = st.number_input("Age", min_value=1, max_value=120, value=50)
sex = st.selectbox("Sex (1 = Male, 0 = Female)", [1, 0])
cp = st.selectbox("Chest Pain Type (0–3)", [0, 1, 2, 3])
trestbps = st.number_input("Resting Blood Pressure (mm Hg)", value=120)
chol = st.number_input("Cholesterol (mg/dl)", value=200)
fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl (1 = Yes, 0 = No)", [1, 0])
restecg = st.selectbox("Resting ECG Results (0–2)", [0, 1, 2])
thalach = st.number_input("Max Heart Rate Achieved", value=150)
exang = st.selectbox("Exercise-Induced Angina (1 = Yes, 0 = No)", [1, 0])
oldpeak = st.number_input("Oldpeak (ST Depression)", value=1.0, format="%.1f")
slope = st.selectbox("Slope of ST Segment (0–2)", [0, 1, 2])
ca = st.selectbox("Number of Major Vessels (0–3)", [0, 1, 2, 3])
thal = st.selectbox("Thalassemia (0 = Normal, 1 = Fixed, 2 = Reversible)", [0, 1, 2])

# ---------------------------
# Prepare Input for Prediction
# ---------------------------
input_data = np.array([[age, sex, cp, trestbps, chol, fbs, restecg,
                        thalach, exang, oldpeak, slope, ca, thal]])

# Standardize the input using the loaded scaler
scaled_input = scaler.transform(input_data)

# ---------------------------
# Predict Button
# ---------------------------
if st.button("Predict"):
    prediction = model.predict(scaled_input)
    probability = model.predict_proba(scaled_input)[0][1]  # probability of heart disease

    st.subheader("🔍 Prediction Result")

    if prediction[0] == 1:
        st.error(f"""**High Risk of Heart Disease**  
        Probability: {probability:.2f}""")
    else:
        st.success(f"""**Low Risk of Heart Disease**  
        Probability: {probability:.2f}""")

# ---------------------------
# Footer
# ---------------------------
st.write("---")
st.caption("Model developed using Machine Learning on the UCI Heart Disease dataset.")
