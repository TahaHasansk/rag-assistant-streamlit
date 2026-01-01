import os
import streamlit as st

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_groq import ChatGroq

# ---------------- CONFIG ----------------
st.set_page_config(page_title="RAG Assistant", page_icon="📚")

PERSIST_DIR = "chroma_db"
COLLECTION_NAME = "rag_docs"

# ---------------- UI ----------------
st.title("📚 RAG Assistant")
st.write("Upload a TXT file and ask questions about it.")

uploaded_file = st.file_uploader("Upload a .txt document", type=["txt"])

question = st.text_input("Ask a question")

# ---------------- LLM ----------------
llm = ChatGroq(
    model="llama3-8b-8192",
    temperature=0,
)

# ---------------- EMBEDDINGS ----------------
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# ---------------- VECTOR STORE ----------------
vectorstore = Chroma(
    collection_name=COLLECTION_NAME,
    embedding_function=embeddings,
    persist_directory=PERSIST_DIR,
)

# ---------------- INGEST DOCUMENT ----------------
if uploaded_file:
    text = uploaded_file.read().decode("utf-8")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100,
    )

    docs = splitter.split_text(text)
    documents = [Document(page_content=d) for d in docs]

    vectorstore.add_documents(documents)

    st.success("✅ Document indexed successfully!")

# ---------------- QUERY ----------------
if question:
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    docs = retriever.get_relevant_documents(question)

    context = "\n\n".join(d.page_content for d in docs)

    prompt = f"""
Use ONLY the context below to answer the question.
If the answer is not present, say "I don't know".

Context:
{context}

Question:
{question}
"""

    response = llm.invoke(prompt)
    st.markdown("### 🤖 Answer")
    st.write(response.content)
