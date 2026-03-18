from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os

def process_pdf(file_path):
    """Loads a PDF and splits it into chunks with metadata."""
    loader = PyPDFLoader(file_path)
    pages = loader.load()
    
    # Get the filename to add to metadata
    file_name = os.path.basename(file_path)
    for page in pages:
        page.metadata["source_file"] = file_name
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100
    )
    chunks = text_splitter.split_documents(pages)
    return chunks