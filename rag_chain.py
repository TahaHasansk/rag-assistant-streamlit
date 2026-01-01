from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate


llm = OllamaLLM(
    model="llama3",
    base_url="http://ollama:11434",
    temperature=0
)


RAG_PROMPT = PromptTemplate.from_template("""
You are a Retrieval-Augmented Generation assistant.

Rules:
- Use ONLY the provided context.
- If the answer is not present, say:
  "I don't know based on the provided documents."
- Be concise and factual.

Context:
{context}

Question:
{question}

Answer:
""")


def generate_answer(question, documents):
    if not documents:
        return "I don't know based on the provided documents."

    context = "\n\n".join(doc.page_content for doc in documents)

    return llm.invoke(
        RAG_PROMPT.format(
            context=context,
            question=question
        )
    )

