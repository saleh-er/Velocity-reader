from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

def process_pdf(file_path):
    """Loads a PDF and splits it into manageable chunks."""
    loader = PyPDFLoader(file_path)
    pages = loader.load()
    
    # We split text so the AI doesn't get overwhelmed
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100
    )
    chunks = text_splitter.split_documents(pages)
    return chunks