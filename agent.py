# agent.py
from dotenv import load_dotenv
load_dotenv()

import os
import pandas as pd
from langchain_google_genai import ChatGoogleGenerativeAI

class AutoMLAgent:
    def __init__(self, model="models/gemini-2.0-flash"):
        self.llm = ChatGoogleGenerativeAI(
            model=model,
            google_api_key=os.getenv("GOOGLE_API_KEY")
        )

    def ask(self, question: str) -> str:
        response = self.llm.invoke(question)
        return response.content.strip()

    def get_task_type(self, df: pd.DataFrame) -> str:
        prompt = f"""
        You are a data scientist. Given the dataset with columns:
        {list(df.columns)},
        determine the type of ML problem (classification or regression).
        Return one word only: "classification" or "regression".
        """
        return self.ask(prompt).lower()

    def get_cleaning_suggestion(self, df: pd.DataFrame) -> str:
        prompt = f"""
        You are a data science assistant. Given this sample of the dataset:
        {df.head(10).to_string(index=False)},
        suggest:
        - Handling missing values
        - Feature engineering or dropping unnecessary columns
        - Data type conversions

        Respond in bullet points.
        """
        response = self.llm.invoke(prompt)
        return response.content.strip()
