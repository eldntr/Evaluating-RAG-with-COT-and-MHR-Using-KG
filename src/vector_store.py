from langchain_community.vectorstores import FAISS

def create_vector_store(docs, embeddings):
    db = FAISS.from_documents(docs, embeddings)
    return db
