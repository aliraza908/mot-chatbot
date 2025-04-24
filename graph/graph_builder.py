# graph/graph_builder.py

from graph.state import GraphState  # ✅ Required schema

from langgraph.graph import StateGraph, END

from agents.planner_agent import classify_query
from agents.product_agent import handle_product_query
from agents.blog_agent import handle_blog_query
from agents.policy_agent import handle_policy_query
from agents.order_agent import handle_order_tracking
from memory.memory_summary import update_memory_summary
from memory.followup_detector import is_followup


def planner_node(state: GraphState) -> GraphState:
    query = state["query"]
    last_user = state.get("last_user", "")
    last_response = state.get("last_response", "")
    memory_summary = state.get("memory_summary", "")

    plan = classify_query(query, last_user, last_response, memory_summary)
    state["plan"] = plan
    return state

from memory.structured_memory_manager import (
    store_entity_memory,
    summarize_chunks,
)
from tools.entity_extractor import extract_entities

def product_node(state: GraphState) -> GraphState:
    from tools.entity_extractor import extract_entities
    from memory.structured_memory_manager import store_entity_memory, summarize_chunks

    plan = state.get("plan", {})
    structured_memory = state.get("structured_memory", {})
    all_chunks = []

    print(f"\n🔁 [PRODUCT NODE] Executing product actions...")

    for action in plan.get("actions", []):
        if action.get("type") != "product":
            continue

        input_text = action.get("input", "")
        chunks = handle_product_query(input_text)

        if chunks:
            summary = summarize_chunks(chunks)
            structured_memory = store_entity_memory(
                memory_dict=structured_memory,
                entity=input_text.upper(),  # e.g., "UX-09"
                entity_type="product",
                chunks=chunks,
                desc=f"From planner action for: {input_text}",
                summary=summary
            )

            print(f"🧠 [MEMORY] Stored {len(chunks)} chunks for {input_text}")
            all_chunks.extend(chunks)
        else:
            print(f"⚠️ [PRODUCT NODE] No chunks found for {input_text}")

    # ✅ Do not overwrite state["query"]
    if all_chunks:
        state["chunks"] = all_chunks  # single write
        state["last_chunks"] = all_chunks
    else:
        state["chunks"] = []  # ensure downstream final_node has a value

    state["structured_memory"] = structured_memory
    return state




def blog_node(state: GraphState) -> GraphState:
    query = state["plan"].get("refined_query", state["query"])
    chunks = handle_blog_query(query)

    state.setdefault("chunks", []).extend(chunks)
    return state

def policy_node(state: GraphState) -> GraphState:
    query = state["plan"].get("refined_query", state["query"])
    chunks = handle_blog_query(query)

    state.setdefault("chunks", []).extend(chunks)
    return state

def order_node(state: GraphState) -> GraphState:
    order_number = state["plan"].get("order_number")
    order_details = handle_order_tracking(order_number)
    state["order_details"] = order_details
    return state

def memory_node(state: GraphState) -> GraphState:
    if state.get("last_user") and state.get("last_response"):
        followup = is_followup(state["query"], state["last_user"], state["last_response"])
        state["is_followup"] = followup

        summary = update_memory_summary(
            state.get("memory_summary", ""),
            state["last_user"],
            state["last_response"]
        )
        state["memory_summary"] = summary
    else:
        state["is_followup"] = False
    return state

def final_node(state: GraphState) -> GraphState:
    from tools.reranker_tool import gpt_rerank
    from langchain_openai import ChatOpenAI
    import os

    from memory.structured_memory_manager import get_entity_chunks

    llm = ChatOpenAI(model_name="gpt-3.5-turbo", openai_api_key=os.getenv("OPENAI_API_KEY"))



    memory_summary = state.get("memory_summary", "")
    last_user = state.get("last_user", "")
    last_response = state.get("last_response", "")
    order_details = state.get("order_details", None)
    plan = state.get("plan", {})
    query = plan.get("refined_query", state.get("query", ""))
    resolved_entities = plan.get("resolved_entities", [])
    structured_memory = state.get("structured_memory", {})

    # 🔍 Intelligent: Only get chunks for resolved entities
    chunks = state.get("chunks", []) or []
    if plan.get("needs_memory") and not chunks:
        chunks = get_entity_chunks(structured_memory, resolved_entities)
        if chunks:
            print(f"🔁 [FINAL NODE] Retrieved {len(chunks)} chunks from structured memory.")
            state["last_chunks"] = chunks
        else:
            chunks = state.get("last_chunks", [])
            print(f"🔁 [FINAL NODE] Fallback to last_chunks: {len(chunks)}")
    elif chunks:
        state["last_chunks"] = chunks
        print(f"📎 Saved chunks to last_chunks: {len(chunks)}")
    else:
        print("⚠️ No new chunks retrieved, last_chunks not updated.")

    # Optional: Rerank if too many chunks
    if len(chunks) > 8:
        chunks = gpt_rerank(query, chunks, top_k=5)

    # 🔍 Filter structured memory: Only include resolved_entities
    focused_entity_memory = ""
    for entity in resolved_entities:
        if entity in structured_memory:
            summary = structured_memory[entity].get("summary", "")
            if summary and len(summary.strip()) > 10:
                focused_entity_memory += f"{entity} ({structured_memory[entity].get('type', 'entity')}): {summary.strip()}\n"

    # 📦 Order context builder
    order_context = ""
    if order_details:
        product_lines = "\n".join([
            f"- {item['product_name']} (Qty: {item['quantity']}, Price: {item['price']})"
            for item in order_details.get('products', [])
        ])
        order_context = f"""
--- Order Details ---
Order Number: {order_details.get('order_number', 'N/A')}
Customer Name: {order_details.get('customer_name', 'N/A')}
Customer Email: {order_details.get('customer_email', 'N/A')}
Status: {order_details.get('status', 'N/A')}
Payment Status: {order_details.get('payment_status', 'N/A')}
Total Price: {order_details.get('total_price', 'N/A')}
Created At: {order_details.get('created_at', 'N/A')}
Updated At: {order_details.get('updated_at', 'N/A')}
Shipping Address: {order_details.get('shipping_address', 'N/A')}
Billing Address: {order_details.get('billing_address', 'N/A')}
Products:
{product_lines}
"""

    # 🧠 Build context
    context = "\n".join(f"• {c.strip()}" for c in chunks if c.strip())
    entity_list = ", ".join(resolved_entities)

    # 🧠 DEBUG
    print("\n" + "=" * 60)
    print("🧠 [FINAL NODE DEBUG]")
    print("🔹 Query:\n", query)
    print("🔹 Memory Summary:\n", memory_summary)
    print("🔹 Retrieved Chunks:\n", len(chunks))
    print("🔹 Order Context:\n", order_context)
    print("🔹 Last User:\n", last_user)
    print("🔹 Last Response:\n", last_response)
    print("🔹 Resolved Entities:\n", entity_list)
    print("🔹 Focused Structured Memory:\n", focused_entity_memory)
    print("=" * 60 + "\n")

    prompt = f"""
    You are a smart, natural-sounding assistant for the Mall of Toys website.

    Your job is to help users with Beyblade-related questions. Use the information below to respond **intelligently and naturally**, based on the user's current message and past conversation.

    ---

    🧠 Your behavior rules:

    - If the user asks vague questions (e.g. “do you have Beyblades”, “I want to buy one”), ask a polite follow-up question to help them narrow it down (e.g. by type or series).
    
    💬 Last User Message:
    {last_user}

    🤖 Last Assistant Response:
    {last_response}

    🎯 Resolved Entities:
    {entity_list}

    📦 Structured Product Memory:
    {focused_entity_memory or "None"}

    📦 Order Info:
    {order_context}

    📚 Retrieved Info:
    {context}

    🗣️ Query:
    {query}
    """.strip()

    # Run GPT
    res = llm.invoke(prompt)
    final_answer = res.content if hasattr(res, "content") else str(res)

    print("🧠 Final Answer:\n", final_answer)

    state["response"] = final_answer
    return state


