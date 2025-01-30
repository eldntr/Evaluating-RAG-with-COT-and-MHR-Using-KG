import os
from langchain_community.vectorstores import FAISS

def create_vector_store(docs, embeddings, save_path="faiss_index"):
    """ Membuat atau memuat FAISS database dengan pengecekan error """
    if os.path.exists(save_path):
        try:
            print("Memuat FAISS index dari penyimpanan...")
            db = FAISS.load_local(save_path, embeddings)
            return db
        except Exception as e:
            print(f"⚠️  Gagal memuat FAISS index, membuat ulang. Error: {e}")
    
    print("Membuat FAISS index baru...")
    db = FAISS.from_documents(docs, embeddings)
    db.save_local(save_path)
    return db

