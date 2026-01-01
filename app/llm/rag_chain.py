from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate


SYSTEM_PROMPT = """
You are a professional Retrieval-Augmented Generation (RAG) assistant.

Rules:
- Use ONLY the provided context to answer.
- If the answer is not present in the context, say exactly:
  "I don't know based on the provided documents."
- Do NOT use outside knowledge.
- Be concise, factual, and clear.
- Do NOT mention that you are an AI model.

Context:
{context}

Question:
{question}

Answer:
"""


PROMPT = PromptTemplate.from_template(SYSTEM_PROMPT)


def get_rag_chain(retriever):
    """
    Returns a callable RAG function.
    """

    llm = OllamaLLM(
        model="llama3",
        base_url="http://ollama:11434",
        temperature=0,
        streaming=True,
    )

    def rag_chain(question: str) -> str:
        # Retrieve documents
        docs = retriever.invoke(question)

        # Build context
        context = "\n\n".join(doc.page_content for doc in docs)

        # Build prompt
        prompt = PROMPT.format(
            context=context,
            question=question
        )

        # Call LLM
        return llm.invoke(prompt)

    return rag_chain
