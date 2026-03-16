import streamlit as st
import os
from dotenv import load_dotenv
from groq import Groq
from processor import process_pdf
from vector_store import create_vector_store, get_retriever

load_dotenv() # Load your API Key from .env

st.set_page_config(page_title="VelocityReader", page_icon="⚡")
st.title("⚡ VelocityReader: Instant PDF Chat")

# Sidebar for PDF Upload
with st.sidebar:
    st.header("Upload Document")
    uploaded_file = st.file_uploader("Choose a PDF", type="pdf")
    
    if uploaded_file:
        with open("temp.pdf", "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        with st.spinner("Processing PDF..."):
            chunks = process_pdf("temp.pdf")
            create_vector_store(chunks)
            st.success("Ready to Chat!")

# Chat Interface
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask about the PDF..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Search the PDF for context
    retriever = get_retriever()
    docs = retriever.get_relevant_documents(prompt)
    context = "\n".join([doc.page_content for doc in docs])

    # Call Groq
    client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    
    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        full_response = ""
        
        completion = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": f"Use this context to answer: {context}"},
                {"role": "user", "content": prompt}
            ],
            stream=True
        )
        
        for chunk in completion:
            if chunk.choices[0].delta.content:
                full_response += chunk.choices[0].delta.content
                response_placeholder.markdown(full_response + "▌")
        
        response_placeholder.markdown(full_response)
        st.session_state.messages.append({"role": "assistant", "content": full_response})