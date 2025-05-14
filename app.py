# app.py
import streamlit as st
import pandas as pd
import os
import pickle
import subprocess
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import numpy as np
from PIL import Image

from agent import AutoMLAgent
from pipeline import run_automl, generate_shap, plot_target_distribution

st.set_page_config(layout="wide")
st.title("AutoML Agent by MSR")

# --- Session State Initialization ---
for key in ["df", "model_trained", "deploy_clicked", "target_col"]:
    if key not in st.session_state:
        st.session_state[key] = None if key == "df" else False

# --- Dataset Upload ---
uploaded_file = st.file_uploader("📂 Upload CSV Dataset", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.session_state.df = df
    st.success("✅ Dataset uploaded successfully!")
    st.write(df.head())

    st.subheader("Gemini Cleaning Suggestions")
    agent = AutoMLAgent()
    with st.spinner("Generating suggestions..."):
        suggestion = agent.get_cleaning_suggestion(df)
    st.markdown(suggestion)

# --- EDA Function ---
def run_eda(df):
    st.header("🔎 Exploratory Data Analysis")

    st.subheader("📊 Dataset Preview")
    st.dataframe(df.head(50))

    st.subheader("📈 Basic Statistics")
    st.write(df.describe(include="all"))

    st.subheader("📉 Missing Values Heatmap")
    fig, ax = plt.subplots()
    sns.heatmap(df.isnull(), cbar=False, cmap="YlOrRd", ax=ax)
    st.pyplot(fig)

    st.subheader("🧮 Feature Types")
    feature_types = df.dtypes.reset_index()
    feature_types.columns = ["Feature", "Type"]
    st.dataframe(feature_types)

    st.subheader("📌 Feature Distribution")
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    if numeric_cols:
        selected_col = st.selectbox("Select a feature", numeric_cols)
        fig = px.histogram(df, x=selected_col, marginal="box", nbins=30)
        st.plotly_chart(fig)
    else:
        st.warning("No numeric columns available.")

    st.subheader("📉 Correlation Heatmap")
    if len(numeric_cols) >= 2:
        corr = df[numeric_cols].corr()
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
        st.pyplot(fig)
    else:
        st.warning("Not enough numeric columns.")

# --- Main Control Logic ---
if st.session_state.df is not None:
    df = st.session_state.df

    st.markdown("### Choose an action:")
    action = st.radio("Actions", ["Explore Dataset", "Run AutoML", "Training Status", "Retrain Model"])

    # EDA
    if action == "Explore Dataset":
        run_eda(df)

    # Train
    elif action == "Run AutoML":
        st.subheader("🎯 Select Target Column")
        target = st.selectbox("Select Target Column", df.columns)
        st.success(f"✅ Selected: {target}")

        if st.button("Run AutoML Agent"):
            agent = AutoMLAgent()
            with st.spinner("🔍 Gemini Agent analyzing..."):
                #cleaning_suggestions = agent.get_cleaning_suggestion(df)
                task_type = agent.get_task_type(df)

            #st.markdown("### 🧼 Cleaning Suggestions")
            #st.write(cleaning_suggestions)

            st.markdown(f"### 🔍 Detected Task Type: **{task_type.upper()}**")
            st.markdown("### 📊 Target Distribution")
            plot_target_distribution(df, target)
            st.image("outputs/target_dist.png")

            st.markdown("###  Training FLAML Model...")
            model, X = run_automl(df, target)

            with open("trained_model.pkl", "wb") as f:
                pickle.dump(model, f)

            feature_types = X.dtypes.apply(lambda dt: dt.name).to_dict()
            with open("feature_types.pkl", "wb") as f:
                pickle.dump(feature_types, f)

            st.session_state.model_trained = True
            st.session_state.target_col = target
            
            
            st.success("✅ Model training completed!")
            
            
            st.markdown("### 📈 SHAP Feature Importance")
            try:
                generate_shap(model, X)
                st.image("outputs/shap_plot.png", caption="SHAP Feature Importance")
            except Exception as e:
                st.error(f"⚠️ Failed to generate SHAP plot: {e}")


            

                    
        # Deployment Button
        if st.session_state.model_trained:
            st.markdown("### Deployment")
            if st.button("Deploy Model"):
                if not st.session_state.deploy_clicked:
                    st.session_state.deploy_clicked = True
                    try:
                        subprocess.Popen(["streamlit", "run", "predictor_ui.py"])
                        st.success("✅ Predictor UI launched!")
                    except Exception as e:
                        st.error(f"❌ Launch failed: {e}")
                else:
                    st.info("ℹ️ Prediction UI is already running.")


    # Status
    elif action == "Training Status":
        if os.path.exists("trained_model.pkl"):
            st.success("✅ Model is trained and ready.")
            if st.button("Show SHAP Plot"):
                if os.path.exists("outputs/shap_plot.png"):
                    st.image("outputs/shap_plot.png")
                else:
                    st.warning("⚠️ SHAP plot not found.")

            if st.button("Deploy Model"):
                if not st.session_state.deploy_clicked:
                        st.session_state.deploy_clicked = True
                        try:
                            subprocess.Popen(["streamlit", "run", "predictor_ui.py"])
                            st.success("✅ Predictor UI launched!")
                        except Exception as e:
                            st.error(f"❌ Launch failed: {e}")
                else:
                    st.info("ℹ️ Prediction UI is already running.")
                        
        else:
            st.warning("⚠️ Model is not yet trained.")

    # Retrain
    elif action == "Retrain Model":
        if os.path.exists("trained_model.pkl"):
            if st.button("Retrain Now"):
                with open("trained_model.pkl", "rb") as f:
                    model = pickle.load(f)
                X = st.session_state.df.drop(columns=[st.session_state.target_col])
                generate_shap(model, X)
                st.success("✅ Model retrained and SHAP plot updated.")

            if st.button("Deploy Model"):
                if st.button("Launch Deployed UI"):
                    if not st.session_state.deploy_clicked:
                        st.session_state.deploy_clicked = True
                        try:
                            subprocess.Popen(["streamlit", "run", "predictor_ui.py"])
                            st.success("✅ Predictor UI launched!")
                        except Exception as e:
                            st.error(f"❌ Launch failed: {e}")
                    else:
                        st.info("ℹ️ Prediction UI is already running.")
        else:
            st.warning("⚠️ No model found to retrain.")

# Download Model
if os.path.exists("trained_model.pkl"):
    st.download_button(
        label="⬇️ Download Trained Model",
        data=open("trained_model.pkl", "rb"),
        file_name="trained_model.pkl",
        mime="application/octet-stream",
    )
