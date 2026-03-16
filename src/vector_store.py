from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

def create_vector_store(chunks, persist_directory="./chroma_db"):
    """
    Converts text chunks into embeddings and saves them to a local directory.
    Using HuggingFace for free, local embeddings.
    """
    # This model runs locally on your machine to turn text into numbers
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    # Create the search database
    vector_db = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=persist_directory
    )
    
    return vector_db

def get_retriever(persist_directory="./chroma_db"):
    """Loads the existing database to search through later."""
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vector_db = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
    return vector_db.as_retriever(search_kwargs={"k": 3})