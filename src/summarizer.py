import os
from groq import Groq
from langchain_community.chains.summarize import load_summarize_chain
from langchain_groq import ChatGroq

def generate_summary(chunks):
    """Generates a structured summary using Groq."""
    llm = ChatGroq(
        temperature=0, 
        model_name="llama-3.3-70b-versatile",
        groq_api_key=os.getenv("GROQ_API_KEY")
    )
    
    # 'map_reduce' is best for long PDFs as it handles token limits
    chain = load_summarize_chain(llm, chain_type="map_reduce")
    summary = chain.invoke(chunks)
    
    return summary["output_text"]