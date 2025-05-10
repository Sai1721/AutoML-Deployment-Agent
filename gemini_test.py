from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage

llm = ChatGoogleGenerativeAI(model="models/gemini-2.0-flash", google_api_key="AIzaSyCUPYvgWR8w_KqY5Pw6PYhJr08y6StQG94")

response = llm.invoke("Tell me a joke about AI.")
print(response.content)
