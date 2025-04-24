# memory/structured_memory_manager.py

def store_entity_memory(memory_dict, entity, entity_type, chunks, desc=None, summary=None):
    """
    Stores structured memory per entity like UX-09 or Order #1234.
    """
    memory_dict[entity] = {
        "type": entity_type,  # e.g., "product" or "order"
        "chunks": chunks,
        "summary": summary or "",
        "desc": desc or f"Referenced in context",
        "turns": []
    }
    return memory_dict


def get_entity_chunks(memory_dict, entities):
    """
    Retrieves chunks for the given list of entities.
    """
    chunks = []
    for ent in entities:
        if ent in memory_dict:
            chunks.extend(memory_dict[ent]["chunks"])
    return chunks


def update_entity_turn(memory_dict, entity, turn_text):
    """
    Records conversational turns where entity was mentioned.
    """
    if entity in memory_dict:
        memory_dict[entity]["turns"].append(turn_text)
    return memory_dict


def summarize_chunks(chunks):
    """
    Summarizes chunks using GPT (used when storing).
    """
    from langchain_openai import ChatOpenAI
    llm = ChatOpenAI(model_name="gpt-3.5-turbo")

    joined = "\n".join(chunks[:5])  # Limit to top 5 for performance
    prompt = f"Summarize the following context for future use:\n\n{joined}"
    res = llm.invoke(prompt)
    return res.content.strip()
