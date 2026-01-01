import os
import streamlit as st

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

# -----------------------------
# STREAMLIT CONFIG
# -----------------------------
st.set_page_config(page_title="RAG Assistant", page_icon="📚")
st.title("📚 RAG Assistant")
st.caption("Upload a TXT file and ask questions about it.")

# -----------------------------
# CHECK GROQ API KEY
# -----------------------------
if "GROQ_API_KEY" not in os.environ:
    st.error("❌ GROQ_API_KEY not found. Add it in Streamlit → App settings → Secrets.")
    st.stop()

# -----------------------------
# EMBEDDINGS & VECTORSTORE
# -----------------------------
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

VECTOR_DIR = "chroma_db"

vectorstore = Chroma(
    persist_directory=VECTOR_DIR,
    embedding_function=embeddings
)

retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

# -----------------------------
# FILE UPLOAD
# -----------------------------
uploaded_files = st.file_uploader("Upload  .txt documents", type=["txt"],accept_multiple_files=True)

if uploaded_files:
    for uploaded_file in uploaded_files:
	text = uploaded_file.read().decode("utf-8")
	documents.append(text)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )

    docs = splitter.split_documents([Document(page_content=text)])
    vectorstore.add_documents(docs)

    st.success("✅ Document indexed successfully!")

# -----------------------------
# QUESTION INPUT
# -----------------------------
question = st.text_input("Ask a question")

if question:
    # Retrieve context
    docs = retriever.invoke(question)

    if not docs:
        st.warning("No relevant context found.")
        st.stop()

    context = "\n\n".join(d.page_content for d in docs)

    # -----------------------------
    # PROMPT
    # -----------------------------
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant. Answer ONLY using the context below."),
        ("human", "Context:\n{context}\n\nQuestion:\n{question}")
    ])

    # -----------------------------
    # GROQ LLM
    # -----------------------------
    llm = ChatGroq(
        model="llama-3.1-8b-instant",
        temperature=0
    )

    chain = prompt | llm

    response = chain.invoke({
        "context": context,
        "question": question
    })

    st.subheader("Answer")
    st.write(response.content)
