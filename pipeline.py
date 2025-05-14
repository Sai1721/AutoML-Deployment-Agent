import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import joblib  # Import joblib for saving models
from flaml import AutoML

def run_automl(df: pd.DataFrame, target_col: str):
    """Runs AutoML on the dataset and returns the trained model and feature data."""
    X = df.drop(columns=[target_col])
    y = df[target_col]

    automl = AutoML()
    task_type = "classification" if y.nunique() <= 20 else "regression"
    metric = "accuracy" if task_type == "classification" else "r2"

    automl_settings = {
        "time_budget": 60,  # seconds
        "metric": metric,
        "task": task_type,
        "log_file_name": "automl.log",
    }

    automl.fit(X_train=X, y_train=y, **automl_settings)
    
    #os.makedirs("outputs", exist_ok=True)  # Ensure outputs directory exists
    joblib.dump(automl.model, "trained_model.pkl")  # Save the model
    
    return automl.model, X


def detect_uninformative_columns(df: pd.DataFrame):
    """Detects columns that are likely uninformative."""
    drop_cols = []
    for col in df.columns:
        if df[col].nunique() == len(df):  # Unique values → ID-like
            drop_cols.append(col)
        elif df[col].dtype == 'object' and df[col].nunique() / len(df) > 0.95:
            drop_cols.append(col)
    return drop_cols


def generate_shap(model, X):
    print("🧪 SHAP Debug: Model type =", type(model))
    print("🧪 SHAP Debug: X shape =", X.shape)
    print("🧪 SHAP Debug: Columns =", list(X.columns))

    # Detect and drop uninformative columns
    ignore_cols = detect_uninformative_columns(X)
    if ignore_cols:
        print("⚠️ Dropped likely ID/uninformative columns for SHAP:", set(ignore_cols))
        X = X.drop(columns=ignore_cols)

    # Drop non-numeric columns
    non_numeric_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
    if non_numeric_cols:
        print("⚠️ Dropped non-numeric columns for SHAP:", set(non_numeric_cols))
        X = X.drop(columns=non_numeric_cols)

    try:
        # Get native model from FLAML's XGBoostSklearnEstimator or other wrapper
        if hasattr(model, "model"):
            native_model = model.model
        else:
            native_model = model

        explainer = shap.Explainer(native_model.predict, X)
        shap_values = explainer(X)

        os.makedirs("outputs", exist_ok=True)  # Ensure outputs directory exists
        shap.summary_plot(shap_values, X, show=False)
        plt.tight_layout()
        plt.savefig("outputs/shap_plot.png")
        plt.close()
        print("✅ SHAP plot saved to outputs/shap_plot.png")

    except Exception as e:
        print("[SHAP Fallback] TreeExplainer failed:", e)

        try:
            explainer = shap.KernelExplainer(native_model.predict, shap.sample(X, 100))
            shap_values = explainer.shap_values(X.iloc[:100])
            shap.summary_plot(shap_values, X.iloc[:100], show=False)
            plt.tight_layout()
            plt.savefig("outputs/shap_plot.png")
            plt.close()
            print("✅ SHAP plot saved with KernelExplainer fallback.")
        except Exception as e2:
            print("❌ SHAP generation error: SHAP failed for both Explainers.")
            print("Details:", e2)
            raise RuntimeError("SHAP failed for both Explainers.")


def plot_target_distribution(df: pd.DataFrame, target_col: str):
    """Plots and saves the target column distribution as an image."""
    plt.figure(figsize=(8, 4))
    try:
        sns.countplot(data=df, x=target_col)
    except Exception:
        sns.histplot(data=df, x=target_col, bins=30)

    plt.title("Target Column Distribution")
    plt.xticks(rotation=45)
    os.makedirs("outputs", exist_ok=True)  # Ensure outputs directory exists
    plt.tight_layout()
    plt.savefig("outputs/target_dist.png")
    plt.close()