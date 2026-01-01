from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
import os

PERSIST_DIR = "data/chroma"

def get_vectorstore(chunks):
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # Load existing DB if present
    if os.path.exists(PERSIST_DIR) and os.listdir(PERSIST_DIR):
        print("📦 Loading existing ChromaDB...")
        return Chroma(
            persist_directory=PERSIST_DIR,
            embedding_function=embeddings
        )

    # Create new DB
    print("🧠 Creating new ChromaDB...")
    vectordb = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=PERSIST_DIR
    )

    vectordb.persist()
    return vectordb
