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
st.caption("Upload TXT or PDF documents and ask questions based on their content")

# =====================================================
# LLM (Groq)
# =====================================================
llm = ChatGroq(
    model="llama-3.1-8b-instant",  # ✅ VALID MODEL
    temperature=0
)

# =====================================================
# VECTORSTORE (cached)
# =====================================================
@st.cache_resource
def get_vectorstore():
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    return Chroma(
        embedding_function=embeddings,
        persist_directory="chroma_db"
    )

vectorstore = get_vectorstore()

# =====================================================
# FILE UPLOAD
# =====================================================
st.subheader("📄 Upload documents")
uploaded_files = st.file_uploader(
    "Upload TXT or PDF files",
    type=["txt", "pdf"],
    accept_multiple_files=True
)

if uploaded_files:
    documents = []

    for uploaded_file in uploaded_files:
        if uploaded_file.name.endswith(".txt"):
            text = uploaded_file.read().decode("utf-8", errors="ignore")
            documents.append(
                Document(
                    page_content=text,
                    metadata={"source": uploaded_file.name}
                )
            )

        elif uploaded_file.name.endswith(".pdf"):
            reader = PdfReader(uploaded_file)
            for page_num, page in enumerate(reader.pages):
                page_text = page.extract_text()
                if page_text and page_text.strip():
                    documents.append(
                        Document(
                            page_content=page_text,
                            metadata={
                                "source": uploaded_file.name,
                                "page": page_num + 1
                            }
                        )
                    )

    if documents:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=150
        )

        chunks = splitter.split_documents(documents)

        # ✅ CRITICAL FIX: remove empty chunks
        chunks = [c for c in chunks if c.page_content.strip()]

        if chunks:
            vectorstore.add_documents(chunks)
            st.success(f"✅ Added {len(chunks)} chunks to the knowledge base")
        else:
            st.warning("⚠️ No valid text found after splitting")

# =====================================================
# QUESTION ANSWERING
# =====================================================
st.divider()
st.subheader("💬 Ask a Question")

question = st.text_input("Enter your question")

if question:
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

    # ✅ LangChain 1.x correct call
    docs = retriever.invoke(question)

    if not docs:
        st.warning("No relevant context found.")
    else:
        context = "\n\n".join(d.page_content for d in docs)

        prompt = f"""
You are a helpful assistant.
Answer the question using ONLY the context below.

Context:
{context}

Question:
{question}

Answer:
"""

        answer = llm.invoke(prompt)

        st.markdown("### 🧠 Answer")
        st.write(answer.content)

        # =====================================================
        # SOURCES
        # =====================================================
        st.markdown("### 📚 Sources")
        for i, doc in enumerate(docs, start=1):
            source = doc.metadata.get("source", "Unknown")
            page = doc.metadata.get("page")
            if page:
                st.write(f"{i}. {source} (page {page})")
            else:
                st.write(f"{i}. {source}")
