from dotenv import load_dotenv
from langchain_groq import ChatGroq
import os

load_dotenv()

llm = ChatGroq(groq_api_key=os.getenv("GROQ_API_KEY"),model_name="meta-llama/llama-4-scout-17b-16e-instruct")

if __name__ == "__main__":
    response = llm.invoke("what is the two main ingrediants used in samosa")
    print(response.content)