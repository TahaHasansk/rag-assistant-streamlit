import streamlit as st
import os

from typing import List
from groq import Groq

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

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
# Secrets / API key
# -----------------------------
GROQ_API_KEY = st.secrets.get("GROQ_API_KEY") or os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    st.error("❌ GROQ_API_KEY is missing in Streamlit secrets.")
    st.stop()

groq_client = Groq(api_key=GROQ_API_KEY)

# -----------------------------
# Embeddings + Vector DB
# -----------------------------
EMBEDDINGS = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

@st.cache_resource
def load_vectorstore():
    return Chroma(
        persist_directory="chroma_db",
        embedding_function=EMBEDDINGS,
    )

vectorstore = load_vectorstore()

# -----------------------------
# Upload document
# -----------------------------
st.subheader("📄 Upload a document")

uploaded_file = st.file_uploader(
    "Upload a TXT document",
    type=["txt"],
)

if uploaded_file:
    text = uploaded_file.read().decode("utf-8")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
    )

    docs: List[Document] = [
        Document(page_content=chunk)
        for chunk in splitter.split_text(text)
    ]

    vectorstore.add_documents(docs)
    vectorstore.persist()

    st.success("✅ Document indexed successfully!")

# -----------------------------
# Ask a question
# -----------------------------
st.subheader("💬 Ask a question")

question = st.text_input("Enter your question")

if st.button("Ask") and question.strip():

    results = vectorstore.similarity_search(question, k=4)

    if not results:
        st.warning("⚠️ No relevant context found.")
        st.stop()

    context = "\n\n".join([doc.page_content for doc in results])

    prompt = f"""
You are a helpful assistant.
Answer ONLY using the context below.
If the answer is not in the context, say "I don't know."

Context:
{context}

Question:
{question}

Answer:
"""

    try:
        response = groq_client.chat.completions.create(
            model="llama3-70b-8192",
            messages=[
                {"role": "system", "content": "You are a helpful RAG assistant."},
                {"role": "user", "content": prompt},
            ],
            temperature=0,
        )

        answer = response.choices[0].message.content

        st.markdown("### 🤖 Answer")
        st.write(answer)

    except Exception as e:
        st.error("❌ Error while generating answer")
        st.exception(e)
