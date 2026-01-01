from langchain_community.document_loaders import TextLoader
from pathlib import Path


def load_documents(documents_path: str):
    """
    Loads all .txt files from the given directory
    and returns a list of LangChain Document objects.
    """

    documents = []

    for file_path in Path(documents_path).glob("*.txt"):
        loader = TextLoader(str(file_path))
        docs = loader.load()
        documents.extend(docs)

    return documents
