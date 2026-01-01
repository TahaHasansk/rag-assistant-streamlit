from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
from pathlib import Path
import shutil

# ==============================
# INGESTION
# ==============================
from app.ingestion.loader import load_documents
from app.ingestion.splitter import split_documents

# ==============================
# VECTOR STORE
# ==============================
from app.vectorstore.chroma import get_vectorstore

# ==============================
# RAG CHAIN
# ==============================
from app.llm.rag_chain import get_rag_chain


app = FastAPI(title="RAG Assistant API")

# ==============================
# GLOBAL PATHS
# ==============================
DOCUMENTS_DIR = Path("data/documents")
DOCUMENTS_DIR.mkdir(parents=True, exist_ok=True)

# ==============================
# GLOBAL STATE
# ==============================
vectorstore = None
retriever = None
rag_chain = None


# ==============================
# BUILD / REBUILD RAG
# ==============================
def rebuild_rag():
    global vectorstore, retriever, rag_chain

    documents = load_documents(str(DOCUMENTS_DIR))
    chunks = split_documents(documents)

    vectorstore = get_vectorstore(chunks)

    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 5, "fetch_k": 20},
    )

    rag_chain = get_rag_chain(retriever)


# Build once on startup
rebuild_rag()


# ==============================
# MODELS
# ==============================
class QuestionRequest(BaseModel):
    question: str


class AnswerResponse(BaseModel):
    answer: str


# ==============================
# ROUTES
# ==============================
@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/ask", response_model=AnswerResponse)
def ask(payload: QuestionRequest):
    answer = rag_chain(payload.question)
    return {"answer": answer}


@app.post("/upload")
def upload_document(file: UploadFile = File(...)):
    file_path = DOCUMENTS_DIR / file.filename

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # 🔥 CRITICAL STEP: RE-INDEX EVERYTHING
    rebuild_rag()

    return {
        "message": f"{file.filename} uploaded and indexed successfully"
    }
