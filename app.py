import streamlit as st
import tempfile
import os

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

from langchain_groq import ChatGroq

# -----------------------------
# Streamlit config
# -----------------------------
st.set_page_config(
    page_title="RAG Assistant",
    page_icon="📚",
    layout="wide",
)

st.title("📚 RAG Assistant")

# -----------------------------
# Initialize session state
# -----------------------------
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# -----------------------------
# LLM (Groq)
# -----------------------------
llm = ChatGroq(
    api_key=os.environ.get("GROQ_API_KEY"),
    model="llama3-70b-8192",
    temperature=0,
)

# -----------------------------
# File upload
# -----------------------------
st.header("📄 Upload a document")

uploaded_file = st.file_uploader(
    "Upload a TXT document",
    type=["txt"],
)

if uploaded_file is not None:
    with st.spinner("Indexing document..."):
        # Save file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".txt") as tmp:
            tmp.write(uploaded_file.read())
            file_path = tmp.name

        # Load text
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read()

        # Split text
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=150,
        )
        docs = splitter.create_documents([text])

        # Embeddings
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

        # Vector store
        vectorstore = Chroma.from_documents(
            docs,
            embedding=embeddings,
        )

        st.session_state.vectorstore = vectorstore
        os.unlink(file_path)

    st.success("✅ Document indexed successfully!")

# -----------------------------
# Ask questions
# -----------------------------
st.header("💬 Ask a question")

question = st.text_input("Enter your question")

if st.button("Ask") and question:
    if st.session_state.vectorstore is None:
        st.error("❌ Please upload a document first.")
    else:
        with st.spinner("Thinking..."):
            # Retrieve documents
            docs = st.session_state.vectorstore.similarity_search(
                question,
                k=3,
            )

            context = "\n\n".join(d.page_content for d in docs)

            prompt = f"""
You are a Retrieval-Augmented Generation (RAG) assistant.

Answer ONLY using the context below.
If the answer is not present, say: "I don't know."

Context:
{context}

Question:
{question}

Answer:
"""

            # ✅ CORRECT Groq invocation (CHAT FORMAT)
            response = llm.invoke(
                [
                    {
                        "role": "system",
                        "content": "You answer strictly from provided context.",
                    },
                    {
                        "role": "user",
                        "content": prompt,
                    },
                ]
            )

            answer = response.content

        # Save history
        st.session_state.chat_history.append(("You", question))
        st.session_state.chat_history.append(("Assistant", answer))

# -----------------------------
# Chat history
# -----------------------------
if st.session_state.chat_history:
    st.divider()
    st.subheader("🧠 Chat History")

    for role, message in st.session_state.chat_history:
        if role == "You":
            st.markdown(f"**🧑 You:** {message}")
        else:
            st.markdown(f"**🤖 Assistant:** {message}")
