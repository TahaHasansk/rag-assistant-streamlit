import os
import streamlit as st

from langchain.schema import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain.schema import SystemMessage, HumanMessage

# -------------------------------------------------
# Streamlit config
# -------------------------------------------------
st.set_page_config(
    page_title="RAG Assistant",
    page_icon="📚",
    layout="centered",
)

st.title("📚 RAG Assistant")

# -------------------------------------------------
# API Key
# -------------------------------------------------
GROQ_API_KEY = st.secrets.get("GROQ_API_KEY") or os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    st.error("❌ GROQ_API_KEY missing in Streamlit secrets")
    st.stop()

os.environ["GROQ_API_KEY"] = GROQ_API_KEY

# -------------------------------------------------
# Session state
# -------------------------------------------------
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

# -------------------------------------------------
# Upload document
# -------------------------------------------------
st.header("📄 Upload a document")

uploaded_file = st.file_uploader(
    "Upload a TXT file",
    type=["txt"]
)

if uploaded_file:
    text = uploaded_file.read().decode("utf-8")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )

    chunks = splitter.split_text(text)

    documents = [Document(page_content=chunk) for chunk in chunks]

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    st.session_state.vectorstore = Chroma.from_documents(
        documents=documents,
        embedding=embeddings
    )

    st.success("✅ Document indexed successfully!")

# -------------------------------------------------
# Ask question
# -------------------------------------------------
st.header("💬 Ask a question")

question = st.text_input("Enter your question")

if st.button("Ask"):
    if st.session_state.vectorstore is None:
        st.warning("⚠️ Upload a document first")
        st.stop()

    retriever = st.session_state.vectorstore.as_retriever(search_kwargs={"k": 3})
    docs = retriever.get_relevant_documents(question)

    context = "\n\n".join(doc.page_content for doc in docs)

    llm = ChatGroq(
        model="llama3-70b-8192",
        temperature=0
    )

    messages = [
        SystemMessage(
            content=(
                "You are a helpful assistant. "
                "Answer ONLY using the provided context. "
                "If the answer is not in the context, say 'I don't know'."
            )
        ),
        HumanMessage(
            content=f"Context:\n{context}\n\nQuestion:\n{question}"
        )
    ]

    response = llm.invoke(messages)

    st.markdown("### 🧠 Answer")
    st.write(response.content)
