from rank_bm25 import BM25Okapi
from langchain_community.vectorstores import FAISS
from langchain_core.retrievers import BaseRetriever
from typing import List, Optional
from langchain_core.documents import Document
from pydantic import BaseModel, Field

class HybridRetriever(BaseRetriever, BaseModel):  # Gunakan BaseModel dari Pydantic untuk validasi
    vectorstore: FAISS = Field(...)  # Field wajib untuk menyimpan FAISS index
    documents: List[Document] = Field(...)  # Field wajib untuk menyimpan dokumen
    k: int = 4  # Jumlah dokumen yang diambil dalam retrieval
    bm25: Optional[BM25Okapi] = None  # Field opsional untuk BM25

    model_config = {
        "arbitrary_types_allowed": True  # Izinkan tipe non-standar seperti FAISS dan BM25
    }

    def __init__(self, vectorstore: FAISS, documents: List[Document], k: int = 4):
        super().__init__(vectorstore=vectorstore, documents=documents, k=k, bm25=None)

        if documents and all(hasattr(doc, "page_content") and isinstance(doc.page_content, str) for doc in documents):
            object.__setattr__(self, "bm25", BM25Okapi([doc.page_content.split() for doc in documents]))
        else:
            print("⚠️  BM25 tidak dapat dibuat karena dokumen tidak valid atau kosong.")
            object.__setattr__(self, "bm25", None)

    def get_relevant_documents(self, query: str) -> List[Document]:
        """ Hybrid Retrieval dengan pengecekan dokumen kosong """
        if not self.documents:
            print("⚠️  Tidak ada dokumen yang tersedia untuk retrieval!")
            return []

        # FAISS Semantic Search
        faiss_results = self.vectorstore.similarity_search(query, k=self.k)

        # BM25 Keyword Search (hanya jika ada dokumen)
        if not self.bm25:
            print("⚠️  BM25 tidak tersedia, hanya menggunakan FAISS.")
            return faiss_results

        bm25_scores = self.bm25.get_scores(query.split())
        bm25_indices = sorted(range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True)[:self.k]
        bm25_results = [self.documents[i] for i in bm25_indices]

        # Gabungkan hasil tanpa duplikasi
        combined_results = list({doc.page_content: doc for doc in (faiss_results + bm25_results)}.values())

        return combined_results[:self.k]  # Batasi ke jumlah yang ditentukan

    def retrieve(self, query: str) -> List[Document]:
        """ Wrapper untuk get_relevant_documents() agar kompatibel dengan LangChain """
        return self.get_relevant_documents(query)

def get_hybrid_retriever(vectorstore, documents):
    """ Mengembalikan hybrid retriever """
    return HybridRetriever(vectorstore=vectorstore, documents=documents)
