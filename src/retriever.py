from rank_bm25 import BM25Okapi
from langchain_community.vectorstores import FAISS
from langchain_core.retrievers import BaseRetriever
from typing import List, Optional
from langchain_core.documents import Document
from pydantic import BaseModel, Field

class HybridRetriever(BaseRetriever, BaseModel):
    vectorstore: FAISS = Field(...)  # FAISS index
    documents: List[Document] = Field(...)  # Dokumen untuk BM25
    k: int = 4  # Jumlah dokumen untuk diambil
    bm25: Optional[BM25Okapi] = None  # Model BM25 opsional

    model_config = {
        "arbitrary_types_allowed": True  # Izinkan tipe non-standar seperti FAISS dan BM25
    }

    def __init__(self, vectorstore: FAISS, documents: List[Document], k: int = 2):
        super().__init__(vectorstore=vectorstore, documents=documents, k=k, bm25=None)

        if documents and all(hasattr(doc, "page_content") and isinstance(doc.page_content, str) for doc in documents):
            tokenized_corpus = [doc.page_content.split() for doc in documents]
            if tokenized_corpus:  # Pastikan dokumen tidak kosong
                object.__setattr__(self, "bm25", BM25Okapi(tokenized_corpus))
            else:
                print("⚠️  BM25 tidak dapat dibuat karena dokumen kosong.")
                object.__setattr__(self, "bm25", None)
        else:
            print("⚠️  BM25 tidak dapat dibuat karena dokumen tidak valid.")
            object.__setattr__(self, "bm25", None)

    def get_relevant_documents(self, query: str) -> List[Document]:
        """ Hybrid Retrieval dengan FAISS + BM25 dan bobot yang lebih seimbang """
        if not self.documents:
            print("⚠️  Tidak ada dokumen yang tersedia untuk retrieval!")
            return []

        # FAISS Semantic Search
        faiss_weight = 1  # Bobot untuk FAISS
        bm25_weight = 0  # Bobot untuk BM25
        k_faiss = int(self.k * faiss_weight)
        k_bm25 = self.k - k_faiss  # Sisa kuota untuk BM25

        faiss_results = self.vectorstore.similarity_search(query, k=k_faiss)

        # BM25 Keyword Search
        if self.bm25:
            bm25_scores = self.bm25.get_scores(query.split())
            bm25_indices = sorted(range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True)[:k_bm25]
            bm25_results = [self.documents[i] for i in bm25_indices]

            print(f"✅ FAISS retrieved {len(faiss_results)} documents, BM25 retrieved {len(bm25_results)} documents.")

            # Gabungkan hasil dan hilangkan duplikasi
            unique_results = {}
            for doc in (faiss_results + bm25_results):
                unique_results[doc.page_content] = doc

            return list(unique_results.values())[:self.k]  # Ambil hanya k dokumen unik

        else:
            print("⚠️  BM25 tidak tersedia, hanya menggunakan FAISS.")
            return faiss_results

    def retrieve(self, query: str) -> List[Document]:
        """ Wrapper untuk get_relevant_documents() agar kompatibel dengan LangChain """
        return self.invoke(query)

def get_hybrid_retriever(vectorstore, documents):
    """ Mengembalikan hybrid retriever """
    return HybridRetriever(vectorstore=vectorstore, documents=documents)
