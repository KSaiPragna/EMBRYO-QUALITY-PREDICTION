import streamlit as st
import pandas as pd
import numpy as np
import joblib
import base64

# Set page config
st.set_page_config(page_title="Embryo Quality Prediction", layout="wide")

# Strong CSS to force big label fonts
st.markdown("""
    <style>

    /* Make full page wider to avoid cutting */
    .main .block-container {
        max-width: 90% !important;
        padding-left: 5% !important;
        padding-right: 5% !important;
    }

    /* Feature names */
    div[data-testid="stNumberInput"] label p {
        text-align: center !important;
        font-size: 25px !important;
        font-weight: bold !important;
        color: white !important;
    }

    /* Input box */
    div[data-testid="stNumberInput"] {
        width: 75% !important;
        margin: auto;
    }

    /* Input values */
    input[type="number"] {
        font-size: 20px !important;
        font-weight: bold !important;
        color: white !important;
    }

    /* Predict Button size */
    button[kind="primary"] {
        font-size: 26px !important;  /* text size same as inputs */
        font-weight: bold !important;
        padding: 40px 150px !important;  /* increased height & width */
        color: black !important;
        background-color: #F5F5F5 !important;
        border: 2px solid black !important;
        border-radius: 15px !important;
    }

    /* Predict button horizontally */
    div.stButton {
        display: flex;
        justify-content: center;
        align-items: center;
    }
    </style>
""", unsafe_allow_html=True)



# Load model and scaler
model = joblib.load(r"C:\Users\PRAGNA\OneDrive\Desktop\XGBModel.pkl")
scaler = joblib.load(r"C:\Users\PRAGNA\OneDrive\Desktop\Scaler_EQP.pkl")

# Features order used for model (Cycle removed)
features = ['FSH(mIU/mL)', 'LH(mIU/mL)', 'Age (yrs)', 'AMH(ng/mL)', 'BMI', 'AFC']

# Logo
image_path = r"C:\Users\PRAGNA\OneDrive\Desktop\images.png"
with open(image_path, "rb") as image_file:
    encoded = base64.b64encode(image_file.read()).decode()

st.markdown(
    f"""
    <div style="display: flex; justify-content: center; align-items: center; margin-bottom: 20px;">
        <img src="data:image/png;base64,{encoded}" width="200" />
    </div>
    """,
    unsafe_allow_html=True,
)

# Title
st.markdown(
    "<h1 style='text-align: center; font-family: Arial, sans-serif;'>🧬 EMBRYO QUALITY PREDICTION USING PGx - AI & ML</h1>", 
    unsafe_allow_html=True
)

# Description
st.markdown("""
<div style='font-size:20px'>
This AI-powered tool predicts embryo quality based on PGx (Pharmacogenomics) features using advanced machine learning algorithms.

<b>Why it matters:</b> Embryo quality is crucial in fertility treatments like IVF. This tool leverages clinical & genetic data to assist medical decisions.

<h3>🔬 What is Embryo Quality Prediction?</h3>
Embryo quality is a critical factor in fertility treatments such as IVF. Factors like hormone levels, age, and genetic predispositions influence embryo viability.

<h3>🧪 Role of PGx (Pharmacogenomics)</h3>
Pharmacogenomics involves studying how genes affect a person's response to medications or treatments. By integrating PGx data, we aim to enhance the prediction of embryo viability and success rates of fertility treatments.

<h3>🧰 Powered By:</h3>
<ul>
<li>Machine Learning (XGBoost)</li>
<li>Streamlit for user interaction</li>
<li>Scikit-learn for data processing</li>
<li>PGx-based biological insights</li>
</ul>

<b>Let’s get started - Input your information below to help us evaluate embryo quality:</b>
</div>
""", unsafe_allow_html=True)

# Input Sections Centered One Below Another
st.header("👩‍⚕️ Patient Information")

age = st.number_input("Age (yrs)", min_value=18, max_value=50, value=30, key="age")
bmi = st.number_input("BMI", min_value=10.0, max_value=60.0, value=26.0, format="%.2f", key="bmi")
fsh = st.number_input("FSH (mIU/mL)", min_value=0.1, max_value=100.0, value=8.0, format="%.2f", key="fsh")
lh = st.number_input("LH (mIU/mL)", min_value=0.1, max_value=100.0, value=10.0, format="%.2f", key="lh")
amh = st.number_input("AMH (ng/mL)", min_value=0.1, max_value=20.0, value=2.5, format="%.2f", key="amh")
afc = st.number_input("AFC", min_value=1, max_value=50, value=10, key="afc")

# Prepare final input data 
input_data = pd.DataFrame([[fsh, lh, age, amh, bmi, afc]], columns=features)
input_scaled = scaler.transform(input_data)

# Prediction button centered
if st.button("🚀 Predict Embryo Quality"):
    prediction = model.predict(input_scaled)

    label_map = {0: 'Best', 1: 'Fair', 2: 'Poor'}
    predicted_class = label_map[prediction[0]]

    st.subheader("🔍 Prediction Result")

    if predicted_class == 'Best':
        st.markdown("<h2 style='font-size: 35px; color: green; text-align: center;'>✅ Embryo Quality: BEST</h2>", unsafe_allow_html=True)
    elif predicted_class == 'Fair':
        st.markdown("<h2 style='font-size: 35px; color: orange; text-align: center;'>⚠️ Embryo Quality: FAIR</h2>", unsafe_allow_html=True)
    else:
        st.markdown("<h2 style='font-size: 35px; color: red; text-align: center;'>🚫 Embryo Quality: POOR</h2>", unsafe_allow_html=True)

    # Prediction probabilities
    st.subheader("📊 Prediction Probabilities")
    probabilities = model.predict_proba(input_scaled)[0]

    prob_map = {'Best': probabilities[0], 'Fair': probabilities[1], 'Poor': probabilities[2]}
    color_map = {'Best': 'green', 'Fair': 'orange', 'Poor': 'red'}

    for label, prob in prob_map.items():
        if label == predicted_class:
            st.markdown(
                f"<h4 style='font-size:22px; text-align:center; color:{color_map[label]};'><b>{label.upper()}: {prob*100:.2f}% ✅</b></h4>",
                unsafe_allow_html=True)
        else:
            st.markdown(
                f"<h4 style='font-size:22px; text-align:center;'>{label.upper()}: {prob*100:.2f}%</h4>",
                unsafe_allow_html=True)

# Footer
st.markdown("""
---
<div style='text-align:center; color:white; font-size:20px'>
    <p><strong>Developed by K Sai Pragna </strong></p>
    <p>Powered by ML | Streamlit | PGx</p>
</div>
""", unsafe_allow_html=True)
