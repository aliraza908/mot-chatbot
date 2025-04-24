import streamlit as st
from graph.langgraph_runner import run_graph
from memory.memory_summary import update_memory_summary

st.set_page_config(page_title="Mall of Toys Chatbot", page_icon="🧠", layout="centered")

# --- Initialize Session State ---
if "memory_summary" not in st.session_state:
    st.session_state.memory_summary = ""
if "last_user" not in st.session_state:
    st.session_state.last_user = ""
if "last_response" not in st.session_state:
    st.session_state.last_response = ""
if "messages" not in st.session_state:
    st.session_state.messages = []
# 🧠 Initialize session state once
if "last_chunks" not in st.session_state:
    st.session_state.last_chunks = []

if "structured_memory" not in st.session_state:
    st.session_state.structured_memory = {}


st.title("🧠 Mall of Toys - Agentic RAG Chatbot")

# --- Display Chat History (Styled Like ChatGPT) ---
for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.markdown(f"""
        <div style='
            background-color: #3B82F6;
            color:white; 
            padding:12px; 
            border-radius:12px; 
            margin-bottom:10px; 
            max-width:70%; 
            margin-left:auto; 
            text-align:right;'>
            <strong>🧑 You:</strong><br>{msg['content']}
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div style='
            background-color: #000000;
            color:#e0ffe5; 
            padding:12px; 
            border-radius:12px; 
            margin-bottom:10px; 
            max-width:70%; 
            margin-right:auto; 
            text-align:left;'>
            <strong>🤖 Assistant:</strong><br>{msg['content']}
        </div>
        """, unsafe_allow_html=True)


# --- Chat Input at Bottom ---
user_query = st.chat_input("Type your message here...")

if user_query:
    # Append user message
    st.session_state.messages.append({"role": "user", "content": user_query})

    # Run LangGraph Agentic RAG
    # Run LangGraph Agentic RAG
    with st.spinner("🤖 Thinking..."):
        graph_input = {
            "query": user_query,
            "last_user": st.session_state.last_user,
            "last_response": st.session_state.last_response,
            "memory_summary": st.session_state.memory_summary,
            "last_chunks": st.session_state.last_chunks
        }
        state = run_graph(
            query=user_query,
            last_user=st.session_state.last_user,
            last_response=st.session_state.last_response,
            memory_summary=st.session_state.memory_summary,
            last_chunks=st.session_state.last_chunks,
            structured_memory=st.session_state.structured_memory
        )

    assistant_message = str(state.get("response", "⚠️ No response generated")).strip()

    # Append assistant message
    st.session_state.messages.append({"role": "assistant", "content": assistant_message})

    # Update memory + chunks
    st.session_state.memory_summary = update_memory_summary(
        st.session_state.memory_summary, user_query, assistant_message
    )
    st.session_state.structured_memory = state.get("structured_memory", {})

    st.session_state.last_user = user_query
    st.session_state.last_response = assistant_message
    st.session_state.last_chunks = state.get("last_chunks", [])  # ✅ Save reused chunks

    st.rerun()
