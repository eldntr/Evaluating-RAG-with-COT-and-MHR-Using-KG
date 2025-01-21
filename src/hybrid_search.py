def hybrid_search(retriever, db, question):
    # Perform semantic search
    semantic_results = retriever.get_relevant_documents(question)
    
    # Perform keyword-based search
    keyword_results = db.similarity_search(question)
    
    # Combine results
    combined_results = semantic_results + keyword_results
    
    # Remove duplicates
    unique_results = {doc.page_content: doc for doc in combined_results}.values()
    
    return list(unique_results)
