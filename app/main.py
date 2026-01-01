from app.ingestion.loader import load_documents
from ingestion.splitter import split_documents
from app.vectorstore.chroma_store import get_vectorstore
from app.llm.rag_chain import generate_answer


def run_cli(vectorstore):
    print("🟢 RAG Assistant is running!")
    print("Type your question. Press Ctrl+C to exit.\n")

    while True:
        try:
            query = input("🧠 You: ").strip()
            if not query:
                continue

            # --- MMR retrieval (diverse chunks) ---
            retrieved_docs = vectorstore.max_marginal_relevance_search(
                query=query,
                k=5,
                fetch_k=10,
                lambda_mult=0.5,
            )

            print("\n📄 Retrieved Chunks (MMR):")

            if not retrieved_docs:
                print("❌ No relevant chunks found.")
            else:
                for doc in retrieved_docs:
                    print("\n-----------------")
                    print(f"📄 Source: {doc.metadata.get('source', 'unknown')}")
                    print(doc.page_content[:400])
                    print("-----------------")

            # --- Generate answer using RAG ---
            try:
                answer = generate_answer(query, retrieved_docs)
            except Exception as e:
                print("\n⚠️ LLM error (safe fallback):")
                print(str(e))
                answer = "I don't know based on the provided documents."

            print("\n🤖 Assistant:")
            print(answer)
            print("\n" + "-" * 50 + "\n")

        except KeyboardInterrupt:
            print("\n👋 Exiting RAG Assistant.")
            break


def main():
    # Load documents
    documents = load_documents("data/documents")

    # Split documents
    chunks = split_documents(documents)

    # Load or create vector store
    vectorstore = get_vectorstore(chunks)

    # Start CLI
    run_cli(vectorstore)


if __name__ == "__main__":
    main()
