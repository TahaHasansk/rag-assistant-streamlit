from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings


def get_vectorstore(documents):
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectorstore = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        persist_directory="data/chroma"
    )

    return vectorstore
