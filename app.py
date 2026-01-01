import os
import streamlit as st
from typing import List

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

from pypdf import PdfReader


# -----------------------------
# STREAMLIT CONFIG
# -----------------------------
st.set_page_config(
    page_title="RAG Assistant",
    page_icon="📚",
    layout="centered",
)

st.title("📚 RAG Assistant")
st.caption("Upload TXT or PDF files and ask questions about them.")


# -----------------------------
# CHECK GROQ API KEY
# -----------------------------
if "GROQ_API_KEY" not in os.environ:
    st.error("❌ GROQ_API_KEY not found. Add it in Streamlit → App settings → Secrets.")
    st.stop()


# -----------------------------
# FILE READERS
# -----------------------------
def read_txt(file) -> str:
    return file.read().decode("utf-8", errors="ignore")


def read_pdf(file) -> str:
    reader = PdfReader(file)
    text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"
    return text


# -----------------------------
# FILE UPLOAD
# -----------------------------
uploaded_files = st.file_uploader(
    "Upload TXT or PDF files",
    type=["txt", "pdf"],
    accept_multiple_files=True,
)

if not uploaded_files:
    st.stop()


# -----------------------------
# LOAD DOCUMENTS
# -----------------------------
documents: List[Document] = []

for file in uploaded_files:
    if file.type == "text/plain":
        content = read_txt(file)
    elif file.type == "application/pdf":
        content = read_pdf(file)
    else:
        continue

    if content.strip():
        documents.append(
            Document(
                page_content=content,
                metadata={"source": file.name},
            )
        )

if not documents:
    st.error("No readable content found.")
    st.stop()


# -----------------------------
# SPLIT DOCUMENTS
# -----------------------------
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=150,
)

chunks = splitter.split_documents(documents)

if not chunks:
    st.error("Document splitting failed.")
    st.stop()


# -----------------------------
# EMBEDDINGS (SAFE)
# -----------------------------
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


# -----------------------------
# VECTOR STORE
# -----------------------------
vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
)

retriever = vectorstore.as_retriever(search_kwargs={"k": 4})


# -----------------------------
# LLM (GROQ)
# -----------------------------
llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0,
)


# -----------------------------
# PROMPT
# -----------------------------
prompt = ChatPromptTemplate.from_template(
    """You are a helpful assistant.
Use ONLY the context below to answer the question.
If the answer is not in the context, say you don't know.

Context:
{context}

Question:
{question}
"""
)


# -----------------------------
# RAG CHAIN (LC 1.x SAFE)
# -----------------------------
chain = (
    {
        "context": retriever | (lambda docs: "\n\n".join(d.page_content for d in docs)),
        "question": RunnablePassthrough(),
    }
    | prompt
    | llm
    | StrOutputParser()
)


# -----------------------------
# QUESTION UI
# -----------------------------
question = st.text_input("Ask a question")

if question:
    with st.spinner("Thinking..."):
        answer = chain.invoke(question)

    st.subheader("Answer")
    st.write(answer)
