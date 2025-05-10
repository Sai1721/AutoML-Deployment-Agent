# AutoML Agent (Streamlit + Gemini + FLAML)

🚀 Features:
- Upload CSV
- Gemini-powered Agent suggests cleaning & detects task
- AutoML via FLAML
- Visualizations + SHAP

## Setup
```bash
git clone https://github.com/YOUR_USERNAME/automl-agent.git
cd automl-agent
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Setup Gemini API
1. Go to https://makersuite.google.com/app
2. Get API Key
3. Create .env file:
```
GOOGLE_API_KEY=your-key-here
```

## Run App
```bash
streamlit run app.py
```

## Deploy on Streamlit Cloud
1. Push code to GitHub
2. Go to https://share.streamlit.io/
3. Link repo → Set GOOGLE_API_KEY as secret in app settings