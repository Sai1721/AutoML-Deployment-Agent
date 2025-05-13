# predictor_ui.py
import streamlit as st
import pickle
import pandas as pd
import os

st.set_page_config(layout="centered")
st.title("📡 Deployed Prediction UI")

# Load model
if not os.path.exists("trained_model.pkl"):
    st.error("❌ Trained model not found!")
    st.stop()

with open("trained_model.pkl", "rb") as f:
    model = pickle.load(f)

# Load feature types
if not os.path.exists("feature_types.pkl"):
    st.error("❌ Feature types not found!")
    st.stop()

with open("feature_types.pkl", "rb") as f:
    feature_types = pickle.load(f)

# Determine which features the model expects
model_features = list(model.feature_names_in_) if hasattr(model, "feature_names_in_") else list(feature_types.keys())

# UI Inputs
inputs = {}
st.markdown("### ✍️ Enter Input Values")

for feature in model_features:
    dtype = feature_types.get(feature, "float64")

    if dtype in ["float64", "float32"]:
        value = st.number_input(f"{feature}", value=0.0, format="%.4f")
    elif dtype in ["int64", "int32"]:
        value = st.number_input(f"{feature}", value=0, step=1, format="%d")
    elif dtype == "bool":
        value = st.checkbox(f"{feature}")
    else:
        value = st.text_input(f"{feature}")
    
    inputs[feature] = value

# Predict button
if st.button("Predict"):
    try:
        input_df = pd.DataFrame([inputs])

        # Ensure correct column order and type conversion
        input_df = input_df[model_features]  # align columns
        for feature in model_features:
            dtype = feature_types.get(feature, "float64")
            if dtype in ["int64", "int32"]:
                input_df[feature] = input_df[feature].astype(int)
            elif dtype in ["float64", "float32"]:
                input_df[feature] = input_df[feature].astype(float)
            elif dtype == "bool":
                input_df[feature] = input_df[feature].astype(bool)
            else:
                input_df[feature] = input_df[feature].astype(str)

        prediction = model.predict(input_df)
        st.success(f"🎯 Prediction: {prediction[0]}")
    except Exception as e:
        st.error(f"❌ Prediction failed: {e}")
