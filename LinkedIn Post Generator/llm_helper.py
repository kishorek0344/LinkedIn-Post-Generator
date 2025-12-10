import os

# Check if we are running in Streamlit
try:
    import streamlit as st
    STREAMLIT_MODE = True
except ImportError:
    STREAMLIT_MODE = False

# Load .env only if running locally
if not STREAMLIT_MODE:
    from dotenv import load_dotenv
    load_dotenv()

from langchain_groq import ChatGroq

# Get API key
if STREAMLIT_MODE:
    groq_api_key = st.secrets["GROQ_API_KEY"]
else:
    groq_api_key = os.getenv("GROQ_API_KEY")

# Initialize LLM
llm = ChatGroq(groq_api_key=groq_api_key, model_name="meta-llama/llama-4-scout-17b-16e-instruct")

# Test
if __name__ == "__main__":
    response = llm.invoke("what are the two main ingredients used in samosa?")
    print(response.content)
