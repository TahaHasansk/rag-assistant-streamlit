import streamlit as st
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from pypdf import PdfReader

# =====================================================
# PAGE CONFIG (Apple-style)
# =====================================================
st.set_page_config(
    page_title="RAG Assistant",
    page_icon="🧠",
    layout="wide"
)

st.markdown(
    """
    <style>
    body {
        background-color: #0e0e11;
        color: #ffffff;
    }
    .block-container {
        padding-top: 3rem;
        max-width: 1100px;
    }
    input, textarea {
        border-radius: 14px !important;
        border: 1px solid #333 !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("🧠 RAG Assistant")
st.caption("Upload a document. Ask questions. Get grounded answers.")

# =====================================================
# LLM (Groq)
# =====================================================
llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0
)

# =====================================================
# SESSION STATE
# =====================================================
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

# =====================================================
# FILE UPLOAD
# =====================================================
st.subheader("📄 Upload documents")

uploaded_files = st.file_uploader(
    "TXT or PDF only",
    type=["txt", "pdf"],
    accept_multiple_files=True
)

if uploaded_files:
    documents: list[Document] = []

    for file in uploaded_files:
        if file.name.endswith(".txt"):
            text = file.read().decode("utf-8", errors="ignore")
            if text.strip():
                documents.append(
                    Document(
                        page_content=text,
                        metadata={"source": file.name}
                    )
                )

        elif file.name.endswith(".pdf"):
            reader = PdfReader(file)
            for page_num, page in enumerate(reader.pages, start=1):
                page_text = page.extract_text()
                if page_text and page_text.strip():
                    documents.append(
                        Document(
                            page_content=page_text,
                            metadata={
                                "source": file.name,
                                "page": page_num
                            }
                        )
                    )

    if not documents:
        st.warning("No readable text found in uploaded files.")
    else:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=150
        )

        chunks = splitter.split_documents(documents)

        # 🔥 CRITICAL FIX: remove empty chunks
        chunks = [c for c in chunks if c.page_content.strip()]

        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

        # 🔥 CRITICAL FIX: NEW VECTORSTORE PER UPLOAD (NO PERSISTENCE)
        st.session_state.vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings
        )

        st.success(f"Indexed {len(chunks)} chunks successfully.")

# =====================================================
# QUESTION ANSWERING
# =====================================================
st.divider()
st.subheader("💬 Ask a question")

question = st.text_input(
    "What would you like to know?",
    placeholder="What is this document about?"
)

if question:
    if st.session_state.vectorstore is None:
        st.warning("Please upload a document first.")
    else:
        retriever = st.session_state.vectorstore.as_retriever(
            search_kwargs={"k": 4}
        )

        # ✅ CORRECT LangChain 1.x API
        docs = retriever.invoke(question)

        if not docs:
            st.warning("No relevant information found.")
        else:
            context = "\n\n".join(doc.page_content for doc in docs)

            prompt = f"""
You are a precise assistant.
Answer ONLY using the context below.
If the answer is not present, say "Not found in the document."

Context:
{context}

Question:
{question}

Answer:
"""

            response = llm.invoke(prompt)

            st.markdown("### 🧠 Answer")
            st.write(response.content)

            st.markdown("### 📚 Sources")
            for i, doc in enumerate(docs, start=1):
                source = doc.metadata.get("source", "Unknown")
                page = doc.metadata.get("page")
                if page:
                    st.write(f"{i}. {source} — page {page}")
                else:
                    st.write(f"{i}. {source}")
