# memory/followup_detector.py

from langchain_openai import ChatOpenAI
import os

llm = ChatOpenAI(model_name="gpt-3.5-turbo", openai_api_key=os.getenv("OPENAI_API_KEY"))

def is_followup(current_query: str, last_query: str, last_response: str) -> bool:
    """
    GPT-based follow-up detector — returns True if current query builds on previous exchange.
    """
    prompt = f"""
    You are a follow-up detector. Given the previous query and response, determine if the current query depends on it.
    
    Only return true or false. Do not explain.
    
    ---
    Last User Query: "{last_query}"
    Last Assistant Response: "{last_response}"
    Current Query: "{current_query}"
    ---
    Is this a follow-up? Return only true or false.
    """

    response = llm.invoke(prompt)
    raw = response.content.strip().lower()

    print(f"🧠 [FOLLOWUP] Detected follow-up? → {raw}")
    return "true" in raw
