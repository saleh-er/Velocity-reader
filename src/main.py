import streamlit as st
import os
import time
from dotenv import load_dotenv
from groq import Groq
from processor import process_pdf
from vector_store import create_vector_store, get_retriever
from summarizer import generate_summary

load_dotenv() 

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
    st.header("📂 Upload Document")
    uploaded_files = st.file_uploader("Uploade one or more PDFs to get started", type=["pdf"], accept_multiple_files=True)
    
    if uploaded_files:
        all_chunks = []
        for uploaded_file in uploaded_files:
            temp_path = f"temp_{uploaded_file.name}"
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            with st.spinner(f"indexing {uploaded_file.name}..."):
                chunks = process_pdf(temp_path)
                all_chunks.extend(chunks)
                # Clean up the temp file after processclsing
                os.remove(temp_path)
            if all_chunks:
             create_vector_store(all_chunks)
             st.success(f"✅ {len(uploaded_files)} documents indexed!")
            
            # This line must be indented with 8 spaces (aligned with st.success)
            with st.expander("📊 Document Summary", expanded=True):
                with st.spinner("Generating summary..."):
                    summary_text = generate_summary(all_chunks[:5])
                    st.markdown(summary_text)
                    st.info("💡 **Try asking:** 'What are the main risks?'")


    st.divider()
    st.subheader("🚀 Performance Stats")
    # Placeholder metrics to show off the 'Velocity' branding
    st.info("Inference Engine: **Groq LPU™**")
    st.info("Model: **Llama-3.3-70B**")

# --- CHAT INTERFACE ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask a question about your document..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Search the PDF for context
    retriever = get_retriever()
    docs = retriever.invoke(prompt)
    context = "\n".join([doc.page_content for doc in docs])

    # Call Groq
    client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    
    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        full_response = ""
        
        # Start timer for speed metrics
        start_time = time.time()
        
        completion = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": f"You are a helpful assistant. Use this context to answer: {context}"},
                {"role": "user", "content": prompt}
            ],
            stream=True
        )
        
        for chunk in completion:
            if chunk.choices[0].delta.content:
                full_response += chunk.choices[0].delta.content
                response_placeholder.markdown(full_response + "▌")
        
        response_placeholder.markdown(full_response)
        
        # Calculate time taken for that "Wow" factor
        end_time = time.time()
        duration = round(end_time - start_time, 2)
        st.caption(f"Generated in {duration}s using Groq Cloud")
        
        st.session_state.messages.append({"role": "assistant", "content": full_response})