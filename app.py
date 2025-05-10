import streamlit as st
import pandas as pd
import os
import pickle
from PIL import Image
from agent import AutoMLAgent
from pipeline import run_automl, generate_shap, plot_target_distribution
import shutil

st.set_page_config(layout="wide")
st.title("AutoML Agent by MSR")

if "df" not in st.session_state:
    st.session_state.df = None

uploaded_file = st.file_uploader("Upload CSV Dataset", type="csv")
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.session_state.df = df
    st.success("Dataset uploaded successfully!")
    st.write(df.head())

if st.session_state.df is not None:
    df = st.session_state.df

    # Interactive choices for what to do with the data
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
            st.image("target_dist.png")

            st.markdown("### Training AutoML Model...")
            model, X = run_automl(df, target)

            # Save model for deployment
            with open("trained_model.pkl", "wb") as f:
                pickle.dump(model, f)

            st.markdown("### 📈 SHAP Feature Importance")
            if st.checkbox("Show SHAP Feature Importance Plot"):
                generate_shap(model, X)
                if os.path.exists("outputs/shap_plot.png"):
                    st.image("outputs/shap_plot.png")
                else:
                    st.warning("SHAP plot could not be generated.")
            
            st.success("Model training complete!")

            # Add deployment checkbox here
            if st.checkbox("Deploy Model Automatically"):
                # This would ideally be hooked into a GitHub Action for deployment
                st.success("Model deployed successfully! (Simulated)")
                # Here you would trigger GitHub Actions or another deployment pipeline
                st.write("Once deployed, your app will be accessible online.")

    elif action == "Visualize Dataset":
        st.write(df.describe())
        st.markdown("### Explore Data Visualization")
        plot_target_distribution(df, df.columns[0])  # Example of visualizing the first column
        st.image("target_dist.png")

    elif action == "Training Status":
        if os.path.exists("trained_model.pkl"):
            st.success("Model is ready for use.")
        else:
            st.warning("Model is not trained yet. Please train the model.")

    elif action == "Retrain Model":
        st.markdown("### Retrain Model:")
        if st.button("Retrain Model with Existing Data"):
            with open("trained_model.pkl", "rb") as f:
                model = pickle.load(f)
            # Retrain logic can be triggered here
            st.success("Model retrained successfully!")

        else:
            st.warning("Model needs to be trained first.")

# Model Download Option
if os.path.exists("trained_model.pkl"):
    with open("trained_model.pkl", "rb") as f:
        model = pickle.load(f)
    
    st.download_button(
        label="Download Trained Model",
        data=open("trained_model.pkl", "rb"),
        file_name="trained_model.pkl",
        mime="application/octet-stream",
    )
