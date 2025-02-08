import numpy as np
import faiss
from sentence_transformers import SentenceTransformer, CrossEncoder
from rank_bm25 import BM25Okapi

# Fungsi untuk menyimpan embeddings, indeks Faiss, dan dokumen ke file
def save_embeddings_and_index(corpus_embeddings, index, documents, embeddings_path, index_path, documents_path):
    """
    Menyimpan embeddings, indeks Faiss, dan dokumen ke file.
    """
    np.save(embeddings_path, corpus_embeddings)
    faiss.write_index(index, index_path)
    np.save(documents_path, np.array(documents, dtype=object))

# Fungsi untuk memuat embeddings, indeks Faiss, dan dokumen dari file
def load_embeddings_and_index(embeddings_path, index_path, documents_path):
    """
    Memuat embeddings, indeks Faiss, dan dokumen dari file.
    """
    corpus_embeddings = np.load(embeddings_path)
    index = faiss.read_index(index_path)
    documents = np.load(documents_path, allow_pickle=True).tolist()
    return corpus_embeddings, index, documents

from nltk.tokenize import sent_tokenize

def split_long_documents(documents, max_sentences=5):
    chunked_documents = []
    for doc in documents:
        sentences = sent_tokenize(doc["text"])
        chunks = [sentences[i:i + max_sentences] for i in range(0, len(sentences), max_sentences)]
        for chunk in chunks:
            chunked_documents.append({"text": " ".join(chunk), "metadata": doc.get("metadata", {})})
    return chunked_documents

# Fungsi untuk membangun retriever dengan model yang lebih canggih
def build_retriever(documents, model_name="sentence-transformers/all-mpnet-base-v2", 
                    embeddings_path="corpus_embeddings.npy", index_path="faiss_index.index", documents_path="documents.npy"):
    """
    Membangun sistem retriever menggunakan SentenceTransformer dan Faiss.
    """
    # Memecah dokumen panjang menjadi chunks
    documents = split_long_documents(documents)
    # Menggunakan SentenceTransformer untuk menghasilkan embeddings
    model = SentenceTransformer(model_name)
    corpus_embeddings = model.encode(
        [doc["text"] for doc in documents],
        convert_to_tensor=False
    )
    corpus_embeddings = np.array(corpus_embeddings).astype('float32')
    # Membuat indeks Faiss
    index = faiss.IndexFlatL2(corpus_embeddings.shape[1])
    index.add(corpus_embeddings)
    # Simpan embeddings, indeks, dan dokumen
    save_embeddings_and_index(corpus_embeddings, index, documents, embeddings_path, index_path, documents_path)
    return model, index, documents

# Fungsi untuk membangun sistem retriever menggunakan BM25
def build_bm25_retriever(documents):
    """
    Membangun sistem retriever menggunakan BM25.
    """
    tokenized_documents = [doc["text"].split(" ") for doc in documents]
    bm25 = BM25Okapi(tokenized_documents)
    return bm25

# Fungsi untuk melakukan pencarian menggunakan BM25
def search_with_bm25(query, bm25, documents, k=5):
    """
    Melakukan pencarian menggunakan BM25.
    """
    tokenized_query = query.split(" ")
    scores = bm25.get_scores(tokenized_query)
    top_indices = np.argsort(scores)[::-1][:k]
    results = [{"document": documents[idx]["text"], "bm25_score": scores[idx]} for idx in top_indices]
    return results

# Fungsi untuk menggabungkan hasil dari DPR dan BM25
def combine_results(dpr_results, bm25_results, alpha=0.5):
    """
    Menggabungkan hasil dari DPR dan BM25 menggunakan bobot alpha.
    """
    combined_results = {}
    # Normalisasi skor DPR
    dpr_scores = np.array([result["rerank_score"] for result in dpr_results])
    dpr_scores = (dpr_scores - np.min(dpr_scores)) / (np.max(dpr_scores) - np.min(dpr_scores))
    # Normalisasi skor BM25
    bm25_scores = np.array([result["bm25_score"] for result in bm25_results])
    bm25_scores = (bm25_scores - np.min(bm25_scores)) / (np.max(bm25_scores) - np.min(bm25_scores))
    # Menggabungkan skor
    for i, result in enumerate(dpr_results):
        document = result["document"]
        combined_score = alpha * dpr_scores[i] + (1 - alpha) * bm25_scores[i]
        combined_results[document] = {
            "combined_score": combined_score,
            "distance": result.get("distance", None),
            "rerank_score": result.get("rerank_score", None)
        }
    # Mengurutkan hasil berdasarkan skor gabungan
    sorted_results = [
        {"document": doc, "combined_score": data["combined_score"], "distance": data["distance"], "rerank_score": data["rerank_score"]}
        for doc, data in sorted(combined_results.items(), key=lambda x: x[1]["combined_score"], reverse=True)
    ]
    return sorted_results

# Fungsi untuk melakukan re-ranking menggunakan Cross-Encoder
def rerank_with_cross_encoder(query, initial_results, reranker_model="cross-encoder/ms-marco-MiniLM-L-12-v2"):
    """
    Melakukan re-ranking menggunakan Cross-Encoder.
    """
    reranker = CrossEncoder(reranker_model)
    pairs = [(query, result["document"]) for result in initial_results]
    rerank_scores = reranker.predict(pairs)
    reranked_results = [
        {"document": initial_results[i]["document"], "rerank_score": score, "distance": initial_results[i]["distance"]}
        for i, score in enumerate(rerank_scores)
    ]
    reranked_results.sort(key=lambda x: x["rerank_score"], reverse=True)
    return reranked_results

# Fungsi untuk melakukan self-reflection
def self_reflection(query, retrieved_documents, reflection_model="cross-encoder/ms-marco-MiniLM-L-12-v2", threshold=0.3):
    """
    Melakukan self-reflection dengan ambang batas yang lebih ketat.
    """
    reranker = CrossEncoder(reflection_model)
    pairs = [(query, doc["document"]) for doc in retrieved_documents]
    reflection_scores = reranker.predict(pairs)
    
    reflected_results = [
        {
            "document": retrieved_documents[i]["document"],
            "reflection_score": score,
            "distance": retrieved_documents[i]["distance"]
        }
        for i, score in enumerate(reflection_scores) if score > threshold
    ]
    reflected_results.sort(key=lambda x: x["reflection_score"], reverse=True)
    return reflected_results

def search_with_filters(query, bm25, documents, filters=None, k=5):
    """
    Melakukan pencarian dengan filter tambahan.
    """
    tokenized_query = query.split(" ")
    scores = bm25.get_scores(tokenized_query)
    # Terapkan filter jika ada
    if filters:
        filtered_indices = [
            i for i, doc in enumerate(documents)
            if all(doc.get(key) == value for key, value in filters.items())
        ]
        scores = [scores[i] for i in filtered_indices]
        documents = [documents[i] for i in filtered_indices]
    # Ambil top-k hasil berdasarkan skor BM25
    top_indices = np.argsort(scores)[::-1][:k]
    results = [{"document": documents[idx]["text"], "bm25_score": scores[idx]} for idx in top_indices]
    return results

def search_hybrid(query, model, index, bm25, documents, reranker_model="cross-encoder/ms-marco-MiniLM-L-12-v2", 
                  reflection_model="cross-encoder/ms-marco-MiniLM-L-12-v2", k=5, alpha=0.5, filters=None):
    """
    Melakukan pencarian hybrid dengan penambahan metadata filtering.
    """
    # Pencarian awal menggunakan Faiss (DPR)
    query_embedding = model.encode(query, convert_to_tensor=False)
    query_embedding = np.array([query_embedding]).astype('float32')
    distances, indices = index.search(query_embedding, k * 2)
    initial_results = [{"document": documents[idx]["text"], "distance": distances[0][i]} for i, idx in enumerate(indices[0])]
    
    # Re-ranking menggunakan Cross-Encoder untuk DPR
    dpr_results = rerank_with_cross_encoder(query, initial_results, reranker_model)
    
    # Pencarian menggunakan BM25 dengan filter
    bm25_results = search_with_filters(query, bm25, documents, filters=filters, k=k * 2)
    
    # Menggabungkan hasil dari DPR dan BM25
    combined_results = combine_results(dpr_results, bm25_results, alpha=alpha)
    
    # Self-reflection untuk mengevaluasi relevansi dokumen
    reflected_results = self_reflection(query, combined_results, reflection_model, threshold=0.1)
    
    return reflected_results[:k]