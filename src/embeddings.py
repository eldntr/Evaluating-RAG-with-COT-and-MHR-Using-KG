import hashlib
import torch
import numpy as np
from langchain_huggingface import HuggingFaceEmbeddings  # Perbaikan impor

def get_embeddings():
    model_path = "sentence-transformers/all-MiniLM-L6-v2"
    model_kwargs = {'device': "cuda" if torch.cuda.is_available() else "cpu"}
    encode_kwargs = {'normalize_embeddings': False}
    
    embeddings = HuggingFaceEmbeddings(
        model_name=model_path,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
    )
    
    return embeddings

def hash_text(text):
    return hashlib.sha256(text.encode()).hexdigest()

def save_embedding_to_cache(text, embedding, conn):
    """ Simpan embedding ke SQLite cache sebagai BLOB """
    embedding_blob = np.array(embedding).tobytes()  # Konversi list ke binary
    cur = conn.cursor()
    cur.execute("INSERT OR IGNORE INTO cache (text_hash, embedding) VALUES (?, ?)", (hash_text(text), embedding_blob))
    conn.commit()

def get_cached_embedding(text, conn):
    """ Ambil embedding dari cache dan konversi kembali ke list """
    cur = conn.cursor()
    cur.execute("SELECT embedding FROM cache WHERE text_hash=?", (hash_text(text),))
    row = cur.fetchone()
    return np.frombuffer(row[0]) if row else None  # Konversi binary ke array

def get_or_compute_embedding(text, embeddings, conn):
    """ Ambil embedding dari cache atau hitung baru jika tidak ada """
    cached_embedding = get_cached_embedding(text, conn)
    if cached_embedding:
        return cached_embedding
    new_embedding = embeddings.embed_query(text)
    save_embedding_to_cache(text, new_embedding, conn)
    return new_embedding