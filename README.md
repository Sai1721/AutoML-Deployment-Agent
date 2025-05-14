AutoML Deployment Agent 
An end-to-end AutoML pipeline powered by Gemini Pro, FLAML, and Streamlit that enables users to upload CSV data, receive intelligent cleaning suggestions, train machine learning models automatically, visualize results, and deploy with optional retraining capabilities.

📌 Project Highlights
🧠 LLM-Powered Suggestions (via Gemini Pro)

🔧 AutoML with FLAML

📊 Feature Summary & Target Distribution

📤 Model Download + Streamlit-Based Deployment

♻️ Retraining with Upload/Checkbox Trigger

🧩 Modular Microservice Architecture

🗂️ Folder Structure
bash
Copy
Edit
AutoML-Agent/
│
├── app.py                   # Streamlit UI: Upload, Train, Visualize, Deploy
├── agent.py                 # Gemini API logic (task detection, cleaning suggestions)
├── pipeline.py              # AutoML logic using FLAML + SHAP
├── predictor_ui.py          # Lightweight Streamlit prediction UI
├── outputs/
│   ├── trained_model.pkl    # Saved trained model
│   └── shap_summary.png     # SHAP plot (if generated)
├── .env                     # Gemini API Key
└── README.md
🔁 Workflow Overview
Upload a CSV dataset

Gemini Agent:

Suggests data cleaning

Detects task type (classification/regression)

AutoML Engine (FLAML):

Applies suggestions

Trains best model automatically

Generates SHAP plots & metrics

Utilities:

Show target distribution

Show summary stats

Deployment:

Launches prediction UI if deployment checkbox is enabled

Allows retraining via file re-upload or checkbox

🛠️ Technologies Used
Module	Stack/Library
UI Interface	Streamlit
AutoML Engine	FLAML, SHAP
LLM Integration	Gemini Pro via LangChain
Deployment (Simulated)	Streamlit Local Instance
Optional External Deployment	GitHub Actions + Streamlit Cloud

📷 Architecture Diagram
AutoML Agent Architecture:


🚀 Running the App Locally
Clone the repo:

bash
Copy
Edit
git clone https://github.com/yourusername/AutoML-Agent.git
cd AutoML-Agent
Install dependencies:

bash
Copy
Edit
pip install -r requirements.txt
Add your Gemini API key in a .env file:

ini
Copy
Edit
GEMINI_API_KEY=your_api_key_here
Run the Streamlit app:

bash
Copy
Edit
streamlit run app.py
After training, if "Deploy Automatically" is checked, the predictor UI (predictor_ui.py) will launch.

✅ To Do (Optional Enhancements)
 Auto GitHub Deployment (Streamlit Cloud/HuggingFace)

 Enhanced EDA Visuals (boxplots, pairplots)

 LLM-based model explanation (why the model performed the way it did)

 HuggingFace Spaces version

🙌 Contributors
Sairaman Mathivelan