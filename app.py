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

from pypdf import PdfReader

# -----------------------------
# STREAMLIT CONFIG
# -----------------------------
st.set_page_config(page_title="RAG Assistant", page_icon="📚")
st.title("📚 RAG Assistant")
st.caption("Upload TXT or PDF files and ask questions about them.")

# -----------------------------
# CHECK GROQ API KEY
# -----------------------------
if "GROQ_API_KEY" not in os.environ:
    st.error("❌ GROQ_API_KEY not found. Add it in Streamlit → App settings → Secrets.")
    st.stop()

# -----------------------------
# EMBEDDINGS & VECTOR STORE
# -----------------------------
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

VECTOR_DIR = "chroma_db"

vectorstore = Chroma(
    persist_directory=VECTOR_DIR,
    embedding_function=embeddings
)

# -----------------------------
# FILE UPLOAD (MULTI + TXT + PDF)
# -----------------------------
uploaded_files = st.file_uploader(
    "Upload documents",
    type=["txt", "pdf"],
    accept_multiple_files=True
)

def load_documents(files) -> List[Document]:
    documents = []

    for file in files:
        if file.type == "text/plain":
            text = file.read().decode("utf-8")
            documents.append(
                Document(
                    page_content=text,
                    metadata={"source": file.name}
                )
            )

        elif file.type == "application/pdf":
            reader = PdfReader(file)
            full_text = ""
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    full_text += page_text + "\n"

            documents.append(
                Document(
                    page_content=full_text,
                    metadata={"source": file.name}
                )
            )

    return documents

if uploaded_files:
    with st.spinner("📄 Processing documents..."):
        docs = load_documents(uploaded_files)

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )

        chunks = splitter.split_documents(docs)
        vectorstore.add_documents(chunks)

    st.success("✅ Documents indexed successfully!")

# -----------------------------
# QUESTION ANSWERING
# -----------------------------
question = st.text_input("Ask a question")

if question:
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

    context_docs = retriever.invoke(question)
    context_text = "\n\n".join(doc.page_content for doc in context_docs)

    prompt = ChatPromptTemplate.from_template(
        """
You are a helpful assistant.
Answer the question ONLY using the context below.
If the answer is not in the context, say "I don't know".

Context:
{context}

Question:
{question}
"""
    )

    llm = ChatGroq(
        model="llama-3.1-8b-instant",
        temperature=0
    )

    chain = (
        prompt
        | llm
        | StrOutputParser()
    )

    with st.spinner("🤖 Thinking..."):
        answer = chain.invoke(
            {"context": context_text, "question": question}
        )

    st.markdown("### Answer")
    st.write(answer)
