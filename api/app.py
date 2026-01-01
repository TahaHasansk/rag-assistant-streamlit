from fastapi import FastAPI
from pydantic import BaseModel
from app.ingestion.loader import load_documents
from app.ingestion.splitter import split_documents
from app.vectorstore.chroma_store import get_vectorstore
from app.llm.rag_chain import generate_answer

app = FastAPI(title="RAG Assistant API")

# --- Load RAG components once (on startup) ---
documents = load_documents("data/documents")
chunks = split_documents(documents)
vectorstore = get_vectorstore(chunks)


class AskRequest(BaseModel):
    question: str


class AskResponse(BaseModel):
    answer: str


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/documents")
def list_documents():
    sources = set()
    for doc in documents:
        if "source" in doc.metadata:
            sources.add(doc.metadata["source"])
    return {"documents": list(sources)}


@app.post("/ask", response_model=AskResponse)
def ask_question(payload: AskRequest):
    question = payload.question.strip()
    if not question:
        return {"answer": "Question cannot be empty."}

    # MMR retrieval (same as CLI)
    retrieved_docs = vectorstore.max_marginal_relevance_search(
        query=question,
        k=5,
        fetch_k=10,
        lambda_mult=0.5,
    )

    answer = generate_answer(question, retrieved_docs)
    return {"answer": answer}
