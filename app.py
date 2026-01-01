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

# ---------------- UI ----------------
st.title("📚 RAG Assistant")
st.write("Upload a TXT file and ask questions about it.")

uploaded_file = st.file_uploader("Upload a .txt document", type=["txt"])

# ---------------- EMBEDDINGS ----------------
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# ---------------- LOAD VECTOR STORE ----------------
vectorstore = Chroma(
    persist_directory=PERSIST_DIR,
    embedding_function=embeddings
)

# ---------------- FILE INGESTION ----------------
if uploaded_file:
    text = uploaded_file.read().decode("utf-8")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )

    chunks = splitter.split_text(text)
    documents = [Document(page_content=c) for c in chunks]

    vectorstore.add_documents(documents)

    st.success("✅ Document indexed successfully!")

# ---------------- QUESTION ----------------
question = st.text_input("Ask a question")

if question:
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

    docs = retriever.invoke(question)

    if not docs:
        st.warning("No relevant content found.")
    else:
        context = "\n\n".join(d.page_content for d in docs)

        llm = ChatGroq(
            model="llama3-8b-8192",
            temperature=0,
            groq_api_key=st.secrets["GROQ_API_KEY"]
        )

        prompt = f"""
You are a helpful assistant.
Answer ONLY using the context below.

Context:
{context}

Question:
{question}
"""

        response = llm.invoke(prompt)

        st.subheader("Answer")
        st.write(response.content)
