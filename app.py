import os
import streamlit as st

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

# -------------------------------
# Streamlit UI
# -------------------------------
st.set_page_config(page_title="RAG Assistant", page_icon="📚")
st.title("📚 RAG Assistant")
st.write("Upload a TXT file and ask questions about it.")

# -------------------------------
# Environment
# -------------------------------
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    st.error("❌ GROQ_API_KEY not set in Streamlit secrets")
    st.stop()

# -------------------------------
# Upload file
# -------------------------------
uploaded_file = st.file_uploader("Upload a .txt document", type=["txt"])

# -------------------------------
# Initialize embeddings once
# -------------------------------
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# -------------------------------
# Handle document upload
# -------------------------------
if uploaded_file:
    text = uploaded_file.read().decode("utf-8")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )

    chunks = splitter.split_text(text)

    documents = [Document(page_content=chunk) for chunk in chunks]

    vectorstore = Chroma.from_documents(
        documents=documents,
        embedding=embeddings
    )

    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

    st.success("✅ Document indexed successfully!")

    # -------------------------------
    # Ask a question
    # -------------------------------
    question = st.text_input("Ask a question")

    ask = st.button("Ask")

    if ask and question:
        # Retrieve documents (CORRECT API)
        docs = retriever.invoke(question)

        context = "\n\n".join(d.page_content for d in docs)

        prompt = ChatPromptTemplate.from_template(
            """You are a helpful assistant.
Use ONLY the context below to answer.
If the answer is not in the context, say "I don't know".

Context:
{context}

Question:
{question}
"""
        )

        llm = ChatGroq(
            api_key=GROQ_API_KEY,
            model="llama-3.1-70b-versatile",
            temperature=0
        )

        chain = prompt | llm

        response = chain.invoke(
            {"context": context, "question": question}
        )

        st.markdown("### Answer")
        st.write(response.content)
