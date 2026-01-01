import streamlit as st
from langchain.text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
import tempfile
import os

# ===============================
# STREAMLIT CONFIG
# ===============================
st.set_page_config(
    page_title="RAG Assistant",
    page_icon="📚",
    layout="wide"
)

st.title("📚 RAG Assistant")
st.caption("Upload a document and ask questions based only on its content.")

# ===============================
# GROQ LLM (FREE)
# ===============================
llm = ChatGroq(
    api_key=st.secrets["GROQ_API_KEY"],
    model="llama3-70b-8192",
    temperature=0
)

# ===============================
# SESSION STATE
# ===============================
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# ===============================
# FILE UPLOAD
# ===============================
st.header("📄 Upload a TXT file")

uploaded_file = st.file_uploader(
    "Choose a .txt file",
    type=["txt"]
)

if uploaded_file:
    with st.spinner("Processing document..."):
        text = uploaded_file.read().decode("utf-8")

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=100
        )
        chunks = splitter.split_text(text)

        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

        st.session_state.vectorstore = Chroma.from_texts(
            chunks,
            embedding=embeddings
        )

    st.success("✅ Document indexed successfully!")

# ===============================
# QUESTION ASKING
# ===============================
st.header("💬 Ask a question")

question = st.text_input("Enter your question")

if st.button("Ask") and question:
    if st.session_state.vectorstore is None:
        st.error("❌ Please upload a document first.")
    else:
        with st.spinner("Thinking..."):
            docs = st.session_state.vectorstore.similarity_search(
                question,
                k=3
            )

            context = "\n\n".join(d.page_content for d in docs)

            prompt = f"""
You are a RAG assistant.
Answer ONLY using the context below.
If the answer is not present, say "I don't know."

Context:
{context}

Question:
{question}

Answer:
"""

            response = llm.invoke(prompt)

        st.session_state.chat_history.append(
            ("You", question)
        )
        st.session_state.chat_history.append(
            ("Assistant", response.content)
        )

# ===============================
# CHAT HISTORY
# ===============================
st.divider()
st.subheader("🧠 Chat History")

for role, message in st.session_state.chat_history:
    if role == "You":
        st.markdown(f"**🧑 You:** {message}")
    else:
        st.markdown(f"**🤖 Assistant:** {message}")
