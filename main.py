from fastapi import FastAPI
from pydantic import BaseModel

from app.llm.rag_chain import get_rag_chain

app = FastAPI(title="RAG Assistant API")

# Load RAG once on startup
rag_chain = get_rag_chain()


class QuestionRequest(BaseModel):
    question: str


class AnswerResponse(BaseModel):
    answer: str


@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.post("/ask", response_model=AnswerResponse)
def ask_question(payload: QuestionRequest):
    result = rag_chain(payload.question)
    return {"answer": result}

