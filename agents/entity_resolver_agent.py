# agents/entity_resolver_agent.py

from langchain_openai import ChatOpenAI
from tools.entity_extractor import extract_entities  # You'll create this next

llm = ChatOpenAI(model_name="gpt-3.5-turbo")  # Use GPT-4 for best reasoning

def resolve_entities(query, last_user, last_response, memory_summary, structured_memory):
    """
    Uses GPT to resolve vague references using memory and previous turns.
    """
    known_entities = list(structured_memory.keys())
    known_text = ", ".join(known_entities) if known_entities else "none"

    prompt = f"""
    You are an intelligent entity resolver helping a toy store assistant understand what products the user is referring to.

    Your job is to:
    1. ONLY reference memory if the query contains vague terms like "it", "that", "the above product", "compare it", "tell me more".
    2. If the query is self-contained (e.g., "recommend Beyblades from CX and UX series"), DO NOT include memory entities.
    3. Determine which known entities (from memory) the user may be referring to.
    4. Rewrite the query only if vague references are present.

    KNOWN ENTITIES (from memory): {known_text}

   

    MEMORY SUMMARY:
    {memory_summary}


    CURRENT QUERY:
    {query}

    ---

    Respond ONLY in this JSON format:
    {{
      "resolved_entities": [...],        // e.g., ["UX-09", "CX-01"]
      "refined_query": "..."             // e.g., "Compare UX-09 with CX-01"
    }}

    RULES:
    - If the query contains vague references, use memory to resolve them.
    - If the query is fresh and clear (like "recommend beyblades from CX series"), ignore memory entities even if they exist.
    - Never assume the user wants to compare unless clearly stated.
    """

    res = llm.invoke(prompt)
    content = res.content.strip()

    try:
        import json, re
        json_block = re.search(r"\{.*\}", content, re.DOTALL).group(0)
        result = json.loads(json_block)
        result["resolved_entities"] = extract_entities(result.get("refined_query", ""))
        return result
    except:
        return {
            "resolved_entities": extract_entities(query),
            "refined_query": query
        }
