from langchain_openai import ChatOpenAI
import os
import json
import re

from agents.entity_resolver_agent import resolve_entities
from tools.entity_extractor import extract_entities

llm = ChatOpenAI(model_name="gpt-3.5-turbo", openai_api_key=os.getenv("OPENAI_API_KEY"))

def classify_query(query: str, last_user: str = "", last_response: str = "", memory_summary: str = "", structured_memory: dict = {}) -> dict:
    """GPT-4 powered planner with intelligent action routing for multi-agent execution."""

    print(f"\n🧠 [PLANNER] Classifying query: '{query}'")

    # --- Step 1: Build enhanced planning prompt ---
    prompt = f"""
You are an intelligent planner for an ecommerce toy store assistant that sells Beyblades from the CX, UX, and BX series.

Your job is to understand the user's intent and produce a structured JSON plan to help route the query to the correct AI agents.

Use memory, past turns, and query content to reason intelligently.

---

You must output the following fields:

1. "intents": List of high-level intents detected in the query. Choose one or more from:
   - "product"
   - "order"
   - "policy"
   - "blog"
   - "general"

2. "needs_memory": true if the query refers to a previous topic or uses vague terms like "it", "that one", etc.

3. "info_complete": true if the query includes all required details (like product codes or order numbers) for agent execution.

4. "order_number": Extract the order number if provided (e.g., "#123456" or "order 54321"). If not present, return null.

5. "actions": List of agent calls required to answer the query. Only add an action of type "order" if an actual order number is detected.
   Each action must contain:
   - "type": one of "product", "policy", "blog", or "order"
   - "input": the product code, policy topic, blog topic, or order number

---

Return ONLY a valid JSON object like this:
{{
  "intents": [...],
  "needs_memory": true | false,
  "info_complete": true | false,
  "order_number": "..." | null,
  "actions": [{{"type": "...", "input": "..."}}]
}}

---

Previous User Message:
"{last_user}"

Previous Assistant Response:
"{last_response}"

Memory Summary:
{memory_summary}

Current User Query:
"{query}"
""".strip()

    # --- Step 2: Ask GPT for high-level plan ---
    try:
        res = llm.invoke(prompt)
        raw = res.content if hasattr(res, "content") else str(res)
        print(f"📥 [PLANNER RAW] GPT returned:\n{raw.strip()}")

        json_match = re.search(r"\{.*\}", raw, re.DOTALL)
        base_plan = json.loads(json_match.group(0)) if json_match else {}
    except Exception as e:
        print(f"❌ [PLANNER ERROR] Failed to classify base plan: {e}")
        base_plan = {
            "intents": ["general"],
            "needs_memory": False,
            "info_complete": False,
            "order_number": None,
            "actions": []
        }

    # --- Step 3: Entity resolver (adds refined query + resolved entities) ---
    resolved = resolve_entities(query, last_user, last_response, memory_summary, structured_memory)

    final_plan = {
        **base_plan,
        "resolved_entities": resolved.get("resolved_entities", []),
        "refined_query": resolved.get("refined_query", query)
    }

    print(f"✅ [PLANNER FINAL PLAN] {final_plan}")
    return final_plan
