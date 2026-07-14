"""
Streamlit Web App
Car Price Prediction using Machine Learning

Author: Shibaji Biswas
"""

import streamlit as st
import joblib
import pandas as pd

# ---------------------------------
# Page Configuration
# ---------------------------------

st.set_page_config(
    page_title="Car Price Prediction",
    page_icon="🚗",
    layout="wide"
)

# ---------------------------------
# Load Trained Model
# ---------------------------------

model = joblib.load("models/best_model.pkl")

# ---------------------------------
# Header
# ---------------------------------

st.title("🚗 Car Price Prediction using Machine Learning")

st.markdown("""
Predict the estimated market price of a car using a trained **Random Forest Regression Model**.

This application was developed as an end-to-end Machine Learning project including:

- Data Cleaning
- Feature Engineering
- Exploratory Data Analysis
- Model Training
- Model Evaluation
- Model Deployment using Streamlit
""")

st.markdown("---")

# ---------------------------------
# Model Performance
# ---------------------------------

st.markdown("## 📊 Model Performance")

col1, col2, col3 = st.columns(3)

col1.metric("Best Model", "Random Forest")
col2.metric("R² Score", "95.95%")
col3.metric("RMSE", "1786.98")

st.markdown("---")

# ---------------------------------
# Two Column Layout
# ---------------------------------

left_col, right_col = st.columns([1, 1])

# ---------------------------------
# Left Column - Inputs
# ---------------------------------

with left_col:

    st.header("🚘 Enter Car Details")

    st.info("""
Enter the specifications of the car.

The trained Random Forest Regression model will estimate the expected market price.
""")

    symboling = st.slider("Symboling", -2, 3, 0)

    wheelbase = st.number_input("Wheel Base", value=95.0)

    carlength = st.number_input("Car Length", value=170.0)

    carwidth = st.number_input("Car Width", value=65.0)

    carheight = st.number_input("Car Height", value=54.0)

    curbweight = st.number_input("Curb Weight", value=2500)

    enginesize = st.number_input("Engine Size", value=120)

    boreratio = st.number_input("Bore Ratio", value=3.20)

    stroke = st.number_input("Stroke", value=3.20)

    compressionratio = st.number_input(
        "Compression Ratio",
        value=9.0
    )

    horsepower = st.number_input(
        "Horse Power",
        value=100
    )

    peakrpm = st.number_input(
        "Peak RPM",
        value=5500
    )

    citympg = st.number_input(
        "City MPG",
        value=25
    )

    highwaympg = st.number_input(
        "Highway MPG",
        value=30
    )

# ---------------------------------
# Prepare Input Data
# ---------------------------------

input_data = pd.DataFrame(
    [[
        symboling,
        wheelbase,
        carlength,
        carwidth,
        carheight,
        curbweight,
        enginesize,
        boreratio,
        stroke,
        compressionratio,
        horsepower,
        peakrpm,
        citympg,
        highwaympg
    ]],
    columns=[
        "symboling",
        "wheelbase",
        "carlength",
        "carwidth",
        "carheight",
        "curbweight",
        "enginesize",
        "boreratio",
        "stroke",
        "compressionratio",
        "horsepower",
        "peakrpm",
        "citympg",
        "highwaympg",
    ]
)

expected_columns = model.feature_names_in_

for col in expected_columns:
    if col not in input_data.columns:
        input_data[col] = 0

input_data = input_data[expected_columns]

# ---------------------------------
# Right Column - Prediction
# ---------------------------------

with right_col:

    st.header("📈 Prediction")

    st.metric("Best Model", "Random Forest")

    st.metric("R² Score", "95.95%")

    st.metric("RMSE", "1786.98")

    st.markdown("---")

    if st.button("🚗 Predict Car Price"):

        prediction = model.predict(input_data)[0]

        st.success("✅ Prediction Completed Successfully!")

        st.markdown("## 💰 Estimated Car Price")

        st.markdown(
            f"""
            <div style="
                background-color:#d4edda;
                padding:25px;
                border-radius:12px;
                text-align:center;
                font-size:36px;
                font-weight:bold;
                color:#155724;">
                ${prediction:,.2f}
            </div>
            """,
            unsafe_allow_html=True,
        )

# ---------------------------------
# About Project
# ---------------------------------

st.markdown("---")

st.subheader("📘 About This Project")

st.write("""
This application predicts the price of a car using Machine Learning.

### Dataset
- 205 car records
- 26 original features
- 65 engineered features

### Algorithms Compared
- Linear Regression
- Decision Tree
- Random Forest
- Gradient Boosting

### Best Performing Model
🏆 Random Forest Regressor

### Model Performance
- **R² Score:** 95.95%
- **RMSE:** 1786.98
- **MAE:** 1248.12
""")

st.markdown("---")

st.caption(
    "🚀 Developed by Shibaji Biswas | Chandigarh University | Machine Learning Portfolio Project"
)