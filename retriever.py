def query_chroma_db(user_query, collection, embedding_function, top_n=5):
    """
    Perform a similarity search in ChromaDB for the given user query.

    Args:
        user_query (str): The query string provided by the user.
        collection: The ChromaDB collection to search in.
        embedding_function: The embedding function to generate query embeddings.
        top_n (int): Number of top results to return. Default is 5.

    Returns:
        list: A list of dictionaries containing the most relevant chunks and their metadata.
    """
    # Generate embedding for the user query
    query_embedding = embedding_function([user_query])[0]

    # Perform similarity search in ChromaDB
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_n,
        include=['metadatas', 'documents', 'distances']
    )

    # Format the results
    relevant_chunks = []
    if results and results.get('ids') and results['ids'][0]:
        for i, doc_id in enumerate(results['ids'][0]):
            relevant_chunks.append({
                "id": doc_id,
                "distance": results['distances'][0][i],
                "metadata": results['metadatas'][0][i],
                "document": results['documents'][0][i]
            })

    return relevant_chunks