import os
import streamlit as st
from typing import List

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import SentenceTransformerEmbeddings
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
# EMBEDDINGS (STREAMLIT SAFE)
# -----------------------------
embeddings = SentenceTransformerEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    device="cpu"
)

# -----------------------------
# VECTOR STORE (PERSISTENT)
# -----------------------------
VECTOR_DIR = "chroma_db"

vectorstore = Chroma(
    persist_directory=VECTOR_DIR,
    embedding_function=embeddings
)

retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

# -----------------------------
# LLM (GROQ)
# -----------------------------
llm = ChatGroq(
    model="llama3-70b-8192",
    temperature=0
)

# -----------------------------
# PROMPT
# -----------------------------
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful assistant. Answer ONLY using the provided context.\n"
            "If the answer is not in the context, say you don't know."
        ),
        ("human", "Context:\n{context}\n\nQuestion:\n{question}")
    ]
)

chain = prompt | llm | StrOutputParser()

# -----------------------------
# FILE PROCESSING
# -----------------------------
def read_txt(file) -> str:
    return file.read().decode("utf-8", errors="ignore")

def read_pdf(file) -> str:
    reader = PdfReader(file)
    text = ""
    for page in reader.pages:
        text += page.extract_text() or ""
    return text

def process_files(files: List) -> List[Document]:
    documents: List[Document] = []

    for file in files:
        if file.name.lower().endswith(".txt"):
            text = read_txt(file)
        elif file.name.lower().endswith(".pdf"):
            text = read_pdf(file)
        else:
            continue

        if not text.strip():
            continue

        documents.append(
            Document(
                page_content=text,
                metadata={"source": file.name}
            )
        )

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150
    )

    return splitter.split_documents(documents)

# -----------------------------
# UI — FILE UPLOAD
# -----------------------------
uploaded_files = st.file_uploader(
    "Upload TXT or PDF files",
    type=["txt", "pdf"],
    accept_multiple_files=True
)

if uploaded_files:
    docs = process_files(uploaded_files)

    if docs:
        vectorstore.add_documents(docs)
        vectorstore.persist()
        st.success("✅ Document(s) indexed successfully!")
    else:
        st.warning("⚠️ No readable text found in uploaded files.")

# -----------------------------
# UI — QUESTION ANSWERING
# -----------------------------
st.divider()
st.subheader("Ask a question")

question = st.text_input("Enter your question")

if question:
    retrieved_docs = retriever.invoke(question)

    if not retrieved_docs:
        st.info("No relevant context found.")
    else:
        context = "\n\n".join(doc.page_content for doc in retrieved_docs)

        response = chain.invoke(
            {
                "context": context,
                "question": question
            }
        )

        st.markdown("### Answer")
        st.write(response)

        st.markdown("### Sources")
        for doc in retrieved_docs:
            st.write(f"- {doc.metadata.get('source', 'unknown')}")
