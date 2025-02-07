from src.data_processing import load_and_process_data
from src.retriever import build_retriever, load_embeddings_and_index, search
from src.generation import generate_answer
from src.evaluation import evaluate_model, save_evaluation_results_to_csv  # Import fungsi untuk menyimpan hasil evaluasi ke CSV
from dotenv import load_dotenv
import os
import time

load_dotenv()
api_key = os.getenv("API_KEY")
if not api_key:
    raise ValueError("API_KEY not found in environment variables")

qa_dataset, documents = load_and_process_data()
documents = documents[:10]
qa_dataset = qa_dataset[:10]  

embeddings_path = "model/corpus_embeddings.npy"
index_path = "model/faiss_index.index"
documents_path = "model/documents.npy"

# try:
#     model, index, documents = load_embeddings_and_index(embeddings_path, index_path, documents_path)
#     print("Successfully loaded embeddings and index from file.")
# except FileNotFoundError:
#     print("Embeddings and index file not found. Start building...")
start_time = time.time()
model, index, documents = build_retriever(documents, embeddings_path=embeddings_path, index_path=index_path, documents_path=documents_path)
end_time = time.time()
print(f"Time taken to build retriever: {end_time - start_time} seconds")

validation_dataset = [entry for entry in qa_dataset if entry["subset"] == "validation"]
generation_model = "mistralai/mistral-small-24b-instruct-2501"
evaluation_results = evaluate_model(validation_dataset, model, generation_model, index, documents, api_key)

csv_path = "evaluation_results_RAG.csv"
save_evaluation_results_to_csv(evaluation_results, csv_path)
print(f"Evaluation results saved to {csv_path}")