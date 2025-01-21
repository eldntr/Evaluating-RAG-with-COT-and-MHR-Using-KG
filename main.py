import sys
import os

# Determine the script directory
script_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else os.getcwd()

# Add src directory to sys.path
src_path = os.path.join(script_dir, 'src')
if src_path not in sys.path:
    sys.path.append(src_path)

from src.data_loader import load_data
from src.text_splitter import split_text
from src.embeddings import get_embeddings
from src.vector_store import create_vector_store
from src.retriever import get_retriever
from src.pipeline import generate_answer
from src.hybrid_search import hybrid_search

dataset_name = "databricks/databricks-dolly-15k"
page_content_column = "context"

data = load_data(dataset_name, page_content_column)
docs = split_text(data)
embeddings = get_embeddings()
db = create_vector_store(docs, embeddings)
retriever = get_retriever(db)

question = "Who is Joko Widodo?"

# Output using only semantic search
semantic_results = retriever.get_relevant_documents(question)
semantic_answer = generate_answer(retriever, semantic_results, question)
print('Answer using semantic search: ', semantic_answer)

# Output using hybrid search
hybrid_results = hybrid_search(retriever, db, question)
hybrid_answer = generate_answer(retriever, hybrid_results, question)
print('Answer using hybrid search: ', hybrid_answer)