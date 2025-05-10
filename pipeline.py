import pandas as pd
from flaml import AutoML
import shap
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np

def run_automl(df, target):
    automl = AutoML()
    X = df.drop(columns=[target])
    y = df[target]

    task = "classification" if y.nunique() < 10 else "regression"
    automl.fit(X_train=X, y_train=y, task=task, time_budget=60)
    return automl, X

def generate_shap(automl, X):
    model = automl.model
    os.makedirs("outputs", exist_ok=True)

    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X)
    except Exception as tree_error:
        print("TreeExplainer failed, trying KernelExplainer. Reason:", tree_error)
        try:
            background = shap.sample(X, 100, random_state=42)
            explainer = shap.KernelExplainer(model.predict, background)
            shap_values = explainer.shap_values(X.iloc[:50], nsamples=100)
            X = X.iloc[:50]  # limit rows to match explainer input
        except Exception as kernel_error:
            print("SHAP explanation failed completely:", kernel_error)
            return

    # Generate summary plot
    try:
        plt.figure()
        shap.summary_plot(shap_values, X, show=False)
        plt.tight_layout()
        plt.savefig("outputs/shap_plot.png")
        plt.close()
    except Exception as plot_error:
        print("SHAP plot generation failed:", plot_error)

def plot_target_distribution(df, target):
    os.makedirs("outputs", exist_ok=True)
    plt.figure()
    sns.countplot(data=df, x=target)
    plt.title("Target Distribution")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("outputs/target_dist.png")
    plt.close()
