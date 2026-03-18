import streamlit as st
import os
import time
import json
from dotenv import load_dotenv
from groq import Groq
from src.processor import process_pdf
from src.vector_store import create_vector_store, get_retriever
from src.summarizer import generate_summary

load_dotenv() 

# --- UTILITY FUNCTIONS ---
def save_chat_locally():
    """Saves the current session state to a JSON file for persistence."""
    if st.session_state.get("messages"):
        # Ensure data directory exists
        if not os.path.exists("data"):
            os.makedirs("data")
        with open("data/chat_backup.json", "w") as f:
            json.dump(st.session_state.messages, f)

# --- UI CONFIGURATION ---
st.set_page_config(page_title="VelocityReader", page_icon="⚡", layout="wide")

# Custom CSS for a modern look
st.markdown("""
    <style>
    .main {
        background-color: #f8f9fa;
    }
    .stChatMessage {
        border-radius: 15px;
        padding: 1.5rem;
        margin-bottom: 1rem;
        border: 1px solid #e0e0e0;
        background-color: white !important;
    }
    [data-testid="stSidebar"] {
        background-color: #ffffff;
        border-right: 1px solid #eee;
    }
    .stHeader {
        background: linear-gradient(90deg, #FF4B4B, #FF8F8F);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
    }
    </style>
""", unsafe_allow_html=True)

st.title("⚡ VelocityReader")
st.markdown("##### *Instant PDF Intelligence powered by Groq*")

# --- SIDEBAR ---
with st.sidebar:
    st.header("📂 Upload Documents")
    uploaded_files = st.file_uploader("Upload one or more PDFs to get started", type=["pdf"], accept_multiple_files=True)
    
    if uploaded_files:
        all_chunks = []
        for uploaded_file in uploaded_files:
            temp_path = f"temp_{uploaded_file.name}"
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            with st.spinner(f"Indexing {uploaded_file.name}..."):
                chunks = process_pdf(temp_path)
                all_chunks.extend(chunks)
                os.remove(temp_path)
        
        if all_chunks:
            create_vector_store(all_chunks)
            st.success(f"✅ {len(uploaded_files)} documents indexed!")
            
            with st.expander("📊 Document Summary", expanded=True):
                with st.spinner("Generating summary..."):
                    summary_text = generate_summary(all_chunks[:5])
                    st.markdown(summary_text)
                    st.info("💡 **Try asking:** 'What are the main risks?'")

    st.divider()
    st.subheader("🚀 Performance Stats")
    st.info("Inference Engine: **Groq LPU™**")
    st.info("Model: **Llama-3.3-70B**")
    
    # --- SESSION MANAGEMENT ---
    st.divider()
    st.subheader("💾 Session Management")
    
    if st.session_state.get("messages"):
        # Create a formatted Research Report string
        report_content = f"RESEARCH REPORT - {time.strftime('%Y-%m-%d %H:%M:%S')}\n"
        report_content += "="*40 + "\n\n"
        for msg in st.session_state.messages:
            role = "USER" if msg["role"] == "user" else "VELOCITY_READER"
            report_content += f"[{role}]: {msg['content']}\n\n"

        st.download_button(
            label="📥 Download Research Report",
            data=report_content,
            file_name=f"research_report_{int(time.time())}.txt",
            mime="text/plain"
        )

        if st.button("🗑️ Clear Chat History"):
            st.session_state.messages = []
            if os.path.exists("data/chat_backup.json"):
                os.remove("data/chat_backup.json")
            st.rerun()

# --- CHAT INTERFACE ---
if "messages" not in st.session_state:
    # Try to load from backup if it exists
    if os.path.exists("data/chat_backup.json"):
        with open("data/chat_backup.json", "r") as f:
            st.session_state.messages = json.load(f)
    else:
        st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask a question about your documents..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 1. Search PDFs for context
    retriever = get_retriever()
    docs = retriever.invoke(prompt)
    context = "\n".join([doc.page_content for doc in docs])
    
    # 2. Extract Citations
    sources = list({f"📄 {doc.metadata.get('source_file', 'Unknown')} (Pg. {doc.metadata.get('page', 0) + 1})" for doc in docs})

    # 3. Call Groq
    client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    
    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        full_response = ""
        start_time = time.time()
        
        completion = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": f"Answer based on this context: {context}"},
                {"role": "user", "content": prompt}
            ],
            stream=True
        )
        
        for chunk in completion:
            if chunk.choices[0].delta.content:
                full_response += chunk.choices[0].delta.content
                response_placeholder.markdown(full_response + "▌")
        
        response_placeholder.markdown(full_response)
        
        # Display Sources and Speed
        duration = round(time.time() - start_time, 2)
        st.caption(f"🚀 {duration}s | Sources: {', '.join(sources)}")
        
        st.session_state.messages.append({"role": "assistant", "content": full_response})
        
        # PERSISTENCE: Save after every assistant message
        save_chat_locally()