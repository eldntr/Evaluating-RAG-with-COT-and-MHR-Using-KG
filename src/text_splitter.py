from langchain.text_splitter import RecursiveCharacterTextSplitter
import numpy as np
from langchain_core.documents import Document
from nltk.tokenize import sent_tokenize
import nltk

# Download model tokenizer kalimat (hanya perlu dijalankan sekali)
nltk.download('punkt')

def adaptive_chunk_size(docs, base_size=1000, min_size=500, max_size=2000):
    """ Menentukan ukuran chunk adaptif berdasarkan panjang rata-rata dokumen """
    avg_length = np.mean([len(doc.page_content) for doc in docs])
    chunk_size = max(min_size, min(int(avg_length * 0.5), max_size))  # Sesuaikan faktor 0.5 sesuai kebutuhan
    return chunk_size

def split_text(data):
    """ Membagi teks menjadi chunks dengan 2 kalimat per chunk menggunakan NLTK """
    if not data:
        return []
    
    all_chunks = []
    
    for doc in data:
        text = doc.page_content
        sentences = sent_tokenize(text)  # Split kalimat dengan NLTK
        
        # Gabungkan setiap 2 kalimat
        for i in range(0, len(sentences), 2):
            chunk_sentences = sentences[i:i+2]
            chunk_text = ' '.join(chunk_sentences).strip()
            
            if chunk_text:
                new_doc = Document(
                    page_content=chunk_text,
                    metadata=doc.metadata.copy()  # Pertahankan metadata asli
                )
                all_chunks.append(new_doc)
    
    return all_chunks