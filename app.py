import os
import streamlit as st

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_groq import ChatGroq

from pypdf import PdfReader

# =====================================================
# PAGE CONFIG
# =====================================================
st.set_page_config(
    page_title="RAG Assistant",
    page_icon="🧠",
    layout="wide"
)

st.title("🧠 RAG Assistant")
st.caption("Upload documents and ask questions based on their content")

# =====================================================
# LLM
# =====================================================
llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0
)

# =====================================================
# EMBEDDINGS + VECTORSTORE (cached)
# =====================================================
@st.cache_resource
def get_vectorstore():
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    return Chroma(
        embedding_function=embeddings,
        persist_directory="./chroma_db"
    )

vectorstore = get_vectorstore()

# =====================================================
# DOCUMENT LOADING
# =====================================================
def load_documents(uploaded_files):
    documents = []

    for file in uploaded_files:
        if file.name.endswith(".txt"):
            text = file.read().decode("utf-8", errors="ignore")

        elif file.name.endswith(".pdf"):
            reader = PdfReader(file)
            text = ""
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"

        else:
            continue

        if text.strip():
            documents.append(
                Document(
                    page_content=text,
                    metadata={"source": file.name}
                )
            )

    return documents

# =====================================================
# UPLOAD SECTION
# =====================================================
st.markdown("---")
st.subheader("📂 Upload Documents")
st.caption("TXT & PDF • Multiple files supported")

uploaded_files = st.file_uploader(
    "Upload files",
    type=["txt", "pdf"],
    accept_multiple_files=True
)

if uploaded_files:
    with st.spinner("📄 Processing documents..."):
        docs = load_documents(uploaded_files)

        if not docs:
            st.error("❌ No readable text found in uploaded files.")
            st.stop()

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )

        # 🔥 SPLIT
        chunks = splitter.split_documents(docs)

        # 🔥 CRITICAL FIX — FILTER EMPTY CHUNKS
        filtered_chunks = [
            c for c in chunks
            if c.page_content and c.page_content.strip()
        ]

        if not filtered_chunks:
            st.error("❌ All document chunks were empty after processing.")
            st.stop()

        vectorstore.add_documents(filtered_chunks)

        st.success(f"✅ Indexed {len(filtered_chunks)} chunks successfully!")

# =====================================================
# QUESTION SECTION
# =====================================================
st.markdown("---")
st.subheader("💬 Ask a Question")

query = st.text_input("Enter your question")

if query:
    with st.spinner("🤖 Thinking..."):
        retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
        docs = retriever.get_relevant_documents(query)

        context = "\n\n".join(d.page_content for d in docs)

        prompt = f"""
You are a helpful assistant.
Answer the question using ONLY the context below.

Context:
{context}

Question:
{query}
"""

        response = llm.invoke(prompt)

        st.markdown("### ✅ Answer")
        st.write(response.content)

        with st.expander("📚 Sources"):
            for d in docs:
                st.markdown(f"- **{d.metadata.get('source', 'unknown')}**")
