import numpy as np
import faiss
from sentence_transformers import SentenceTransformer, CrossEncoder

def save_embeddings_and_index(corpus_embeddings, index, documents, embeddings_path, index_path, documents_path):
    np.save(embeddings_path, corpus_embeddings)
    faiss.write_index(index, index_path)
    np.save(documents_path, np.array(documents, dtype=object))

def load_embeddings_and_index(embeddings_path, index_path, documents_path):
    corpus_embeddings = np.load(embeddings_path)
    index = faiss.read_index(index_path)
    documents = np.load(documents_path, allow_pickle=True).tolist()
    return corpus_embeddings, index, documents

def build_retriever(documents, model_name="sentence-transformers/multi-qa-mpnet-base-dot-v1", embeddings_path="corpus_embeddings.npy", index_path="faiss_index.index", documents_path="documents.npy"):

    model = SentenceTransformer(model_name)
    corpus_embeddings = model.encode([doc["text"] for doc in documents], convert_to_tensor=False)
    corpus_embeddings = np.array(corpus_embeddings).astype('float32')

    index = faiss.IndexFlatL2(corpus_embeddings.shape[1])
    index.add(corpus_embeddings)

    # save_embeddings_and_index(corpus_embeddings, index, documents, embeddings_path, index_path, documents_path)
    return model, index, documents

def search(query, model, index, documents, reranker_model="cross-encoder/ms-marco-MiniLM-L-12-v2", k=5):

    query_embedding = model.encode(query, convert_to_tensor=False)
    query_embedding = np.array([query_embedding]).astype('float32')

    distances, indices = index.search(query_embedding, k * 2)
    initial_results = [{"document": documents[idx]["text"], "distance": distances[0][i]} for i, idx in enumerate(indices[0])]

    reranker = CrossEncoder(reranker_model)
    pairs = [(query, result["document"]) for result in initial_results]
    rerank_scores = reranker.predict(pairs)

    reranked_results = [
        {"document": initial_results[i]["document"], "rerank_score": score, "distance": initial_results[i]["distance"]}
        for i, score in enumerate(rerank_scores)
    ]

    reranked_results.sort(key=lambda x: x["rerank_score"], reverse=True)
    return reranked_results[:k]