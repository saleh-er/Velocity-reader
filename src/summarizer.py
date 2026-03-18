import os
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

def generate_summary(chunks):
    """Generates a structured summary using a custom LCEL chain."""
    llm = ChatGroq(
        temperature=0, 
        model_name="llama-3.3-70b-versatile",
        groq_api_key=os.getenv("GROQ_API_KEY")
    )

    # Combine the text from the chunks
    combined_text = "\n".join([doc.page_content for doc in chunks])

    # Create a custom prompt
    prompt = ChatPromptTemplate.from_template(
        "Write a concise summary of the following text. "
        "Focus on the main arguments and key takeaways:\n\n{text}"
    )

    # Create the chain: Prompt -> LLM -> Output as String
    chain = prompt | llm | StrOutputParser()
    
    # Run the chain
    summary = chain.invoke({"text": combined_text})
    
    return summary