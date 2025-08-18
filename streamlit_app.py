import streamlit as st
import os
from dotenv import load_dotenv
import google.generativeai as genai

# --- ADD THE FIX HERE ---
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import chromadb
from sentence_transformers import SentenceTransformer
import zipfile

# Check if the database is unzipped. If not, unzip it.
if not os.path.exists("chroma_store"):
    print("Database not found. Unzipping...")
    with zipfile.ZipFile("chroma_store.zip", 'r') as zip_ref:
        zip_ref.extractall()
    print("Database unzipped successfully!")


# -----------------------------
# Core RAG-Gemini Logic
# -----------------------------

# Load environment variables
load_dotenv()

google_api_key = os.getenv("GOOGLE_API_KEY")
if not google_api_key:
    st.error("❌ Missing GOOGLE_API_KEY in .env file.")
    st.stop()

# Connect to existing Chroma store (persistent)
@st.cache_resource
def get_chroma_collection():
    try:
        chroma_client = chromadb.PersistentClient(path="chroma_store")
        collection = chroma_client.get_or_create_collection(name="gemini_rag")
        if collection.count() == 0:
            st.warning("⚠️ Chroma collection is empty. Please run your ingestion script first.")
        return collection
    except Exception as e:
        st.error(f"❌ Error connecting to ChromaDB: {e}")
        st.stop()

collection = get_chroma_collection()

# Load embedding model
@st.cache_resource
def get_embedding_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

embedding_model = get_embedding_model()

# Configure Google Generative AI
@st.cache_resource
def get_gemini_model():
    try:
        genai.configure(api_key=google_api_key)
        return genai.GenerativeModel("gemini-1.5-flash")
    except Exception as e:
        st.error(f"❌ Error configuring Gemini API: {e}")
        st.stop()
        
gemini_model = get_gemini_model()

# -----------------------------
# Streamlit Dashboard Front-end
# -----------------------------

# Custom CSS for styling the dashboard
st.markdown("""
<style>
/* Main app styling */
body {
    background-color: #f0f2f6;
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
}
.stApp {
    background-color: #f0f2f6;
}
.main .block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
    padding-left: 2rem;
    padding-right: 2rem;
    max-width: 1000px;
}
.sidebar .sidebar-content {
    background-color: #ffffff;
    border-right: 1px solid #e0e0e0;
    padding: 20px;
    box-shadow: 2px 0 5px rgba(0,0,0,0.05);
}

/* Chat bubble styling */
.chat-message {
    padding: 10px 15px;
    border-radius: 20px;
    margin-bottom: 15px;
    max-width: 80%;
    position: relative;
    font-size: 16px;
    line-height: 1.4;
    word-wrap: break-word;
}
.user-message {
    background-color: #e6f7ff;
    color: #004080;
    margin-left: auto;
    border-top-right-radius: 5px;
    text-align: right;
}
.bot-message {
    background-color: #ffffff;
    color: #333333;
    border: 1px solid #e0e0e0;
    border-top-left-radius: 5px;
    text-align: left;
}

/* Sidebar styling */
.sidebar-metrics {
    padding: 10px;
    border-radius: 10px;
    background-color: #f8f9fa;
    margin-bottom: 10px;
    border: 1px solid #e9ecef;
}
.logo-container {
    display: flex;
    align-items: center;
    gap: 10px;
    margin-bottom: 20px;
    font-weight: bold;
    font-size: 20px;
    color: #007bff;
}
.logo-container img {
    height: 40px;
}
.logo-text {
    font-size: 16px;
    color: #555555;
    font-weight: normal;
}
.info-text p {
    font-size: 14px;
    color: #777777;
    line-height: 1.5;
}
.sidebar-metrics .metric-label {
    font-size: 14px;
    color: #6c757d;
}
.sidebar-metrics .metric-value {
    font-size: 24px;
    font-weight: bold;
    color: #343a40;
}
.stTextInput > div > div > input {
    border-radius: 20px;
    padding: 10px;
    border: 1px solid #ced4da;
}
.stButton > button {
    border-radius: 20px;
    font-weight: bold;
    color: white;
    background-color: #007bff;
    border: none;
}
.stButton > button:hover {
    background-color: #0056b3;
}
/* Hiding Streamlit's default headers and footers */
footer {visibility: hidden;}
header {visibility: hidden;}
.css-1d391kg {display: none;}
.css-1v3fvz2 {display: none;}
</style>
""", unsafe_allow_html=True)


# Mock metrics based on the session state
total_questions = len([turn for turn in st.session_state.get("history", []) if turn["role"] == "user"])
kb_hits = sum(1 for turn in st.session_state.get("history", []) if turn.get("context") and turn["role"]=="assistant")
general_responses = total_questions - kb_hits

# Initialize conversation history in session state
if "history" not in st.session_state:
    st.session_state["history"] = []

# ---------------- Sidebar ----------------
with st.sidebar:
    st.markdown('<div class="logo-container">@FlowWithBisola<span class="logo-text">Gemini RAG Chat</span></div>', unsafe_allow_html=True)
    st.markdown("""
        <div class="info-text">
            <p>💻 Tech Stack: Streamlit, ChromaDB, Google Gemini API</p>
            <p>📂 Features: RAG bot, KB transparency, copy answers, modern UI</p>
            <p>🎯 Portfolio Demo: Showcases AI + Web Dev skills</p>
        </div>
    """, unsafe_allow_html=True)

    st.markdown(f'<div class="sidebar-metrics"><span class="metric-label">Total Questions</span><br><span class="metric-value">{total_questions}</span></div>', unsafe_allow_html=True)
    st.markdown(f'<div class="sidebar-metrics"><span class="metric-label">KB Hits</span><br><span class="metric-value">{kb_hits}</span></div>', unsafe_allow_html=True)
    st.markdown(f'<div class="sidebar-metrics"><span class="metric-label">General Responses</span><br><span class="metric-value">{general_responses}</span></div>', unsafe_allow_html=True)

    st.markdown(f'<div class="sidebar-metrics"><span class="metric-label">Total Questions</span><br><span class="metric-value">{total_questions}</span></div>', unsafe_allow_html=True)
    st.markdown(f'<div class="sidebar-metrics"><span class="metric-label">KB Hits</span><br><span class="metric-value">{kb_hits}</span></div>', unsafe_allow_html=True)
    st.markdown(f'<div class="sidebar-metrics"><span class="metric-label">General Responses</span><br><span class="metric-value">{general_responses}</span></div>', unsafe_allow_html=True)
# ---------------- Main Content ----------------
st.title("Conversation")

# Display the conversation history
for turn in st.session_state.get("history", []):
    role_class = "user-message" if turn["role"] == "user" else "bot-message"
    with st.container():
        st.markdown(f'<div class="chat-message {role_class}">{turn["content"]}</div>', unsafe_allow_html=True)
        if turn["role"] == "assistant" and turn.get("context"):
            with st.expander("View KB Context"):
                st.markdown(f'<div class="context-box">{turn["context"]}</div>', unsafe_allow_html=True)
        if turn["role"] == "assistant":
            # Using a simple button for demonstration
            st.button("Copy Answer", key=f"copy_btn_{len(st.session_state['history'])}")

# Create the input area at the bottom
with st.form(key="chat_form", clear_on_submit=True):
    col1, col2 = st.columns([0.85, 0.15])
    with col1:
        user_input = st.text_input("", placeholder="Type your question here...", label_visibility="collapsed", key="user_input")
    with col2:
        submit_button = st.form_submit_button("Submit")

    if submit_button and user_input:
        # Add user query to history
        st.session_state["history"].append({"role": "user", "content": user_input})

        # --- Perform the RAG-Gemini Logic ---
        with st.spinner("Thinking..."):
            try:
                # Step 1: Embed query
                query_embedding = embedding_model.encode([user_input]).tolist()

                # Step 2: Query Chroma for relevant docs
                results = collection.query(
                    query_embeddings=query_embedding,
                    n_results=3
                )
                
                context_texts = results["documents"][0] if results["documents"] else []
                
                # Step 3: Build a single, smarter prompt
                context = "\n".join(context_texts)
                prompt = f"""
                You are a helpful AI assistant. Your goal is to answer the user's question.
                
                Provided Context:
                {context}
                
                Conversation so far:
                {''.join([f"{t['role']}: {t['content']}\n" for t in st.session_state['history']])}
                
                Instructions:
                1. Answer the user's question to the best of your ability.
                2. ONLY use the 'Provided Context' if it is highly relevant.
                3. If the context is not relevant, answer based on your general knowledge.
                4. Be clear about the source of your information. For example, if you use the context, you might say "Based on the knowledge base..."
                5. If you use your general knowledge, you might say "Based on my general knowledge..." or similar phrasing.
                
                Your reply:
                """
                
                # Step 4: Generate response
                response = gemini_model.generate_content(prompt)
                bot_reply = response.text.strip()
                
                # Check if the bot used the knowledge base for metrics
                kb_used = "knowledge base" in bot_reply.lower()

            except Exception as e:
                bot_reply = f"❌ An error occurred: {e}"
                kb_used = False
            
        # Add the assistant's response and context to history
        st.session_state["history"].append({
            "role": "assistant",
            "content": bot_reply,
            "context": context if kb_used else None
        })
        
        # Rerun the app to update the chat display
        st.rerun()

