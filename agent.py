from dotenv import load_dotenv
load_dotenv()

import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate

llm = ChatGoogleGenerativeAI(model="models/gemini-2.0-flash")
class AutoMLAgent:
    def __init__(self, model="models/gemini-2.0-flash"):
        self.llm = ChatGoogleGenerativeAI(model=model, google_api_key=os.getenv("GOOGLE_API_KEY"))

    def ask(self, question: str) -> str:
        return self.llm.invoke(question)

    def get_task_type(self, df):
        schema = f"Columns: {list(df.columns)}"
        response = self.ask(f"Given this dataset schema: {schema}, is this a regression or classification task? Return one word.")
        return response.content.strip().lower()

    def get_cleaning_suggestion(self, df):
        schema = str(df.dtypes)
        return self.ask(f"Suggest data cleaning steps for dataset with types: {schema}")