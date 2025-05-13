# app.py
import streamlit as st
import pandas as pd
import os
import pickle
import subprocess
from PIL import Image
from agent import AutoMLAgent
from pipeline import run_automl, generate_shap, plot_target_distribution

st.set_page_config(layout="wide")
st.title("AutoML Agent by MSR")

# Session states
if "df" not in st.session_state:
    st.session_state.df = None
if "model_trained" not in st.session_state:
    st.session_state.model_trained = False
if "deploy_clicked" not in st.session_state:
    st.session_state.deploy_clicked = False
if "target_col" not in st.session_state:
    st.session_state.target_col = None

# File upload
uploaded_file = st.file_uploader("Upload CSV Dataset", type="csv")
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.session_state.df = df
    st.success("✅ Dataset uploaded successfully!")
    st.write(df.head())

# Main logic
if st.session_state.df is not None:
    df = st.session_state.df

    st.markdown("### Choose what you want to do with the dataset:")
    action = st.radio(
        "Actions", 
        ["Run AutoML", "Visualize Dataset", "Training Status", "Retrain Model"]
    )

    if action == "Run AutoML":
        target = st.selectbox("Select Target Column", df.columns)

        agent = AutoMLAgent()

        if st.button("Run AutoML Agent"):
            with st.spinner("Analyzing with Gemini Agent..."):
                cleaning_suggestions = agent.get_cleaning_suggestion(df)
                task_type = agent.get_task_type(df)

            st.markdown("### 🔧 Cleaning Suggestions")
            st.write(cleaning_suggestions)

            st.markdown(f"### 🔍 Detected Task Type: **{task_type.upper()}**")

            st.markdown("### 📊 Target Distribution")
            plot_target_distribution(df, target)
            st.image("outputs/target_dist.png")

            st.markdown("### 🤖 Training AutoML Model...")
            model, X = run_automl(df, target)

            # Save model
            with open("trained_model.pkl", "wb") as f:
                pickle.dump(model, f)

            # ✅ Save feature types
            # Save only features used in training (X.columns)
            feature_types = X.dtypes.apply(lambda dt: dt.name).to_dict()
            with open("feature_types.pkl", "wb") as f:
                pickle.dump(feature_types, f)

            st.session_state.model_trained = True
            st.session_state.target_col = target

            st.markdown("### 📈 SHAP Feature Importance")
            if st.checkbox("Show SHAP Feature Importance Plot"):
                generate_shap(model, X)
                if os.path.exists("outputs/shap_plot.png"):
                    st.image("outputs/shap_plot.png")
                else:
                    st.warning("⚠️ SHAP plot could not be generated.")

            st.success("✅ Model training complete!")

        # Deployment section (after training)
        if st.session_state.model_trained:
            deploy_check = st.checkbox("Deploy Model Automatically")

            if deploy_check:
                if st.button("Launch Deployed UI"):
                    if not st.session_state.deploy_clicked:
                        st.session_state.deploy_clicked = True
                        try:
                            subprocess.Popen(["streamlit", "run", "predictor_ui.py"])
                            st.success("✅ Prediction UI launched!")
                        except Exception as e:
                            st.error(f"❌ Failed to launch prediction UI: {e}")
                    else:
                        st.info("ℹ️ Prediction UI is already running.")

    elif action == "Visualize Dataset":
        st.write(df.describe())
        st.markdown("### Explore Data Visualization")
        plot_target_distribution(df, df.columns[0])  # Example: first column
        st.image("outputs/target_dist.png")

    elif action == "Training Status":
        if os.path.exists("trained_model.pkl"):
            st.success("✅ Model is ready for use.")
        else:
            st.warning("⚠️ Model is not trained yet. Please train the model.")

    elif action == "Retrain Model":
        st.markdown("### 🔄 Retrain Model:")
        if os.path.exists("trained_model.pkl"):
            if st.button("Retrain Model with Existing Data"):
                with open("trained_model.pkl", "rb") as f:
                    model = pickle.load(f)
                st.success("✅ Model retrained successfully!")
            else:
                st.info("Click the button to retrain.")
        else:
            st.warning("⚠️ Train a model first to retrain.")

# Download trained model
if os.path.exists("trained_model.pkl"):
    st.download_button(
        label="⬇️ Download Trained Model",
        data=open("trained_model.pkl", "rb"),
        file_name="trained_model.pkl",
        mime="application/octet-stream",
    )
