from langchain.text_splitter import RecursiveCharacterTextSplitter
import numpy as np

def adaptive_chunk_size(docs, base_size=1000, min_size=500, max_size=2000):
    """ Menentukan ukuran chunk adaptif berdasarkan panjang rata-rata dokumen """
    avg_length = np.mean([len(doc.page_content) for doc in docs])
    chunk_size = max(min_size, min(int(avg_length * 0.5), max_size))  # Sesuaikan faktor 0.5 sesuai kebutuhan
    return chunk_size

def split_text(data):
    """ Membagi teks menjadi chunks dengan ukuran adaptif """
    if not data:
        return []

    chunk_size = adaptive_chunk_size(data)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=int(chunk_size * 0.15))
    docs = text_splitter.split_documents(data)
    return docs
