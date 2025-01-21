def get_retriever(db):
    retriever = db.as_retriever(search_kwargs={"k": 4})
    return retriever
