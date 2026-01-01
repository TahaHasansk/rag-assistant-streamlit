def retrieve_relevant_chunks(vector_store, query, k=3):
    """
    Takes a user query and retrieves the top-k
    most relevant document chunks from ChromaDB.
    """

    retriever = vector_store.as_retriever(
        search_kwargs={"k": k}
    )

    results = retriever.invoke(query)
    return results
