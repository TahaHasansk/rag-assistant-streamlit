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

# =====================================================
# STREAMLIT CONFIG
# =====================================================
st.set_page_config(
    page_title="RAG Assistant",
    page_icon="📚",
    layout="centered"
)

# =====================================================
# APPLE-STYLE GLOBAL CSS
# =====================================================
st.markdown("""
<style>
html, body, [class*="css"] {
    font-family: -apple-system, BlinkMacSystemFont, "San Francisco",
                 "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
}

.stApp {
    background: linear-gradient(180deg, #0f0f11 0%, #121212 100%);
    color: #ffffff;
}

h1, h2, h3 {
    font-weight: 600;
    letter-spacing: -0.02em;
}

.card {
    background: rgba(255, 255, 255, 0.06);
    border-radius: 18px;
    padding: 1.6rem;
    margin-bottom: 1.6rem;
    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.35);
    backdrop-filter: blur(14px);
    border: 1px solid rgba(255, 255, 255, 0.08);
}

[data-testid="stFileUploader"] {
    background: rgba(255, 255, 255, 0.04);
    border-radius: 14px;
    padding: 1rem;
    border: 1px dashed rgba(255, 255, 255, 0.2);
}

input {
    background: rgba(255, 255, 255, 0.06) !important;
    border-radius: 12px !important;
    border: 1px solid rgba(255, 255, 255, 0.15) !important;
    color: white !important;
}

button {
    border-radius: 14px !important;
    padding: 0.55rem 1.3rem !important;
    background: linear-gradient(135deg, #3a3a3c, #2c2c2e) !important;
    color: white !important;
    border: none !important;
    font-weight: 500 !important;
    transition: all 0.25s ease;
}

button:hover {
    transform: scale(1.03);
    background: linear-gradient(135deg, #4a4a4c, #3a3a3c) !important;
}

footer {
    visibility: hidden;
}
</style>
""", unsafe_allow_html=True)

# =====================================================
# HERO HEADER
# =====================================================
st.markdown("""
<div style="text-align:center; padding: 2.5rem 0;">
    <h1>📚 RAG Assistant</h1>
    <p style="opacity:0.7; font-size:1.05rem;">
        Ask questions. Get answers. Powered only by your documents.
    </p>
</div>
""", unsafe_allow_html=True)

# =====================================================
# CHECK GROQ API KEY
# =====================================================
if "GROQ_API_KEY" not in os.environ:
    st.error("❌ GROQ_API_KEY not found. Add it in Streamlit → App settings → Secrets.")
    st.stop()

# =====================================================
# EMBEDDINGS & VECTOR STORE
# =====================================================
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

VECTOR_DIR = "chroma_db"

vectorstore = Chroma(
    persist_directory=VECTOR_DIR,
    embedding_function=embeddings
)

# =====================================================
# FILE LOADING HELPERS
# =====================================================
def load_documents(files) -> List[Document]:
    documents: List[Document] = []

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

# =====================================================
# UPLOAD SECTION
# =====================================================
st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader("📂 Upload Documents")
st.caption("TXT and PDF supported • Multiple files allowed")

uploaded_files = st.file_uploader(
    "Upload files",
    type=["txt", "pdf"],
    accept_multiple_files=True
)

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

st.markdown('</div>', unsafe_allow_html=True)

# =====================================================
# QUESTION SECTION
# =====================================================
st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader("❓ Ask a Question")

question = st.text_input(
    "Your question",
    placeholder="e.g. What is my name?"
)

st.markdown('</div>', unsafe_allow_html=True)

# =====================================================
# ANSWER SECTION
# =====================================================
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

    chain = prompt | llm | StrOutputParser()

    with st.spinner("🤖 Thinking..."):
        answer = chain.invoke(
            {"context": context_text, "question": question}
        )

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("🧠 Answer")
    st.markdown(answer)
    st.markdown('</div>', unsafe_allow_html=True)

# =====================================================
# FOOTER
# =====================================================
st.markdown("---")
st.caption(
    "Built with Streamlit • LangChain • ChromaDB • Groq | "
    "RAG Assistant Project"
)
