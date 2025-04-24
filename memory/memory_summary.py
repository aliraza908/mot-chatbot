# memory/memory_summary.py

from langchain_openai import ChatOpenAI
import os

llm = ChatOpenAI(model_name="gpt-3.5-turbo", openai_api_key=os.getenv("OPENAI_API_KEY"))

def update_memory_summary(existing_summary: str, new_user_msg: str, new_assistant_msg: str) -> str:
    """
    Given a running memory summary + new messages, GPT summarizes the entire memory again.
    """
    prompt = f"""
You are an AI assistant maintaining a running memory of a conversation.

Here is the existing memory summary:
---
{existing_summary}
---

Now add the following exchange:
User: {new_user_msg}
Assistant: {new_assistant_msg}

Return the updated summary:
"""

    response = llm.invoke(prompt)
    updated_summary = response.content.strip()

    print("📘 [MEMORY SUMMARY] Updated memory summary generated.")
    return updated_summary
