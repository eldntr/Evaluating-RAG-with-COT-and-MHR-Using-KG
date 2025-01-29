import sys
import os
import sqlite3

# Tentukan direktori script
script_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else os.getcwd()

# Tambahkan src directory ke sys.path
src_path = os.path.join(script_dir, 'src')
if src_path not in sys.path:
    sys.path.append(src_path)

from src.data_loader import load_data
from src.text_splitter import split_text
from src.embeddings import get_embeddings, get_or_compute_embedding
from src.vector_store import create_vector_store
from src.retriever import get_hybrid_retriever
from src.pipeline import generate_answer, clean_answer

# **1. Inisialisasi Database Embedding Cache**
conn = sqlite3.connect("embedding_cache.db")
conn.execute("CREATE TABLE IF NOT EXISTS cache (text_hash TEXT PRIMARY KEY, embedding BLOB)")
conn.commit()

# **2. Load Dataset**
dataset_name = "databricks/databricks-dolly-15k"
page_content_column = "context"

print("Memuat data...")
data = load_data(dataset_name, page_content_column)

# **3. Split Teks Secara Adaptif**
print("Memecah teks menjadi chunks...")
docs = split_text(data)
print(f"✅ Total dokumen setelah pemrosesan: {len(docs)}")
print(f"✅ Contoh dokumen pertama: {docs[0].page_content if docs else 'Tidak ada data'}")

# **4. Load atau Hitung Embeddings dengan Cache**
print("Menginisialisasi embedding model...")
embeddings = get_embeddings()

# **5. Buat atau Load FAISS Vector Store**
print("Membuat/memuat FAISS Vector Store...")
db = create_vector_store(docs, embeddings)

# **6. Gunakan Hybrid Retriever (BM25 + FAISS)**
print("Menginisialisasi Hybrid Retriever...")
if not docs:
    print("⚠️  Tidak ada dokumen yang tersedia untuk Hybrid Retrieval!")
    sys.exit(1)  # Menghentikan program jika tidak ada dokumen

retriever = get_hybrid_retriever(db, docs)

# **7. Proses Pertanyaan & Hasilkan Jawaban**
question = "When did Virgin Australia start operating?"
chat_history = []  # Bisa diperluas jika ingin percakapan berkelanjutan

print(f"\n🔍 Query: {question}")
retrieved_docs = retriever.retrieve(question)

# **8. Gunakan Conversational Retrieval Chain untuk Jawaban**
raw_answer = generate_answer(retriever, chat_history, question)

# **9. Bersihkan Jawaban Sebelum Ditampilkan**
cleaned_answer = clean_answer(raw_answer)

print("\n📌 Answer:")
print(cleaned_answer)

# Tutup koneksi SQLite
conn.close()