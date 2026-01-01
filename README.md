# 📚 RAG Assistant (Streamlit)

A Retrieval-Augmented Generation (RAG) application built with **Streamlit**, **LangChain 1.x**, **ChromaDB**, and **Groq**.

Upload documents and ask questions — answers are generated **only from your data**, not from the model’s imagination.

---

## 🚀 Features

- ✅ Multi-file upload
- ✅ Supports **TXT** and **PDF**
- ✅ Vector search using **ChromaDB**
- ✅ Fast inference with **Groq**
- ✅ No hallucinations (context-grounded answers)
- ✅ Clean Streamlit UI

---

## 🧠 How It Works

1. Documents are uploaded (TXT / PDF)
2. Text is split into chunks
3. Chunks are embedded using `sentence-transformers`
4. Stored in ChromaDB
5. Relevant chunks are retrieved per question
6. Groq LLM generates an answer using **only retrieved context**

---

## 🛠 Tech Stack

- **Frontend:** Streamlit
- **LLM:** Groq (llama-3.1-8b-instant)
- **Embeddings:** HuggingFace (`all-MiniLM-L6-v2`)
- **Vector Store:** ChromaDB
- **Framework:** LangChain 1.x

---

## 📦 Installation (Local)

```bash
git clone https://github.com/TahaHasansk/rag-assistant-streamlit.git
cd rag-assistant-streamlit
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
