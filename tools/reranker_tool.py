import json
import re
from langchain.chat_models import ChatOpenAI
import os

api_key = os.getenv("OPENAI_API_KEY")
llm = ChatOpenAI(model_name="gpt-3.5-turbo", openai_api_key=api_key)

def build_rerank_prompt(query, chunks):
    prompt = f"""
You are a strict and intelligent assistant that reranks the following text chunks based on how well they directly answer or match the user's query.

RULES:
- Only rank chunks that contain facts, details, or keywords **directly related** to the query.
- If a chunk **doesn't clearly answer or relate** to the query, give it a lower priority.
- DO NOT rank chunks based on similarity alone. Avoid guessing.
- Focus on accuracy, completeness, and relevance to the user query.

---

🔍 Query:
{query}

📚 Chunks:
"""
    for i, chunk in enumerate(chunks, 1):
        safe_chunk = chunk.replace("\n", " ").strip()
        prompt += f"\n{i}. {safe_chunk[:500]}"

    prompt += """

---

🎯 Now return the most relevant chunk numbers in JSON array format like:
[3, 1, 2]

Return ONLY the JSON array and nothing else.
"""
    return prompt.strip()

def gpt_rerank(query, chunks, top_k=5):
    prompt = build_rerank_prompt(query, chunks)
    res = llm.invoke(prompt)
    raw = res.content if hasattr(res, "content") else str(res)

    match = re.search(r'\[.*?\]', raw)
    if not match:
        return chunks[:top_k]

    try:
        indexes = json.loads(match.group())
        return [chunks[i - 1] for i in indexes if 0 < i <= len(chunks)]
    except Exception as e:
        print(f"❌ [RERANKER ERROR] Failed to parse response: {e}")
        return chunks[:top_k]
