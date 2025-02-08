from src.relation_extraction import from_text_to_kb
from src.knowledge_graph import visualize_knowledge_graph, save_knowledge_graph, load_knowledge_graph
from src.knowledge_base import KB 
from dotenv import load_dotenv
from src.data_processing import load_and_process_data
from tqdm import tqdm  
from src.evaluation import evaluate_mhr, save_evaluation_results_to_csv
import os

load_dotenv()

api_key = os.getenv("API_KEY")
if not api_key:
    raise ValueError("API_KEY not found in environment variables")

qa_dataset, documents = load_and_process_data()
qa_dataset = qa_dataset[:100]
documents = documents[:100]

kg_filepath = "combined_kb.pkl"
if os.path.exists(kg_filepath): 
    print("Loading knowledge graph from file...")
    combined_kb = load_knowledge_graph(kg_filepath)
else:
    print("Processing texts to create knowledge graph...")
    combined_kb = KB()
    for i, text in enumerate(tqdm(documents, desc="Processing texts", unit="text")):  
        kb = from_text_to_kb(text["text"], span_length=1024, verbose=False)
        for relation in kb.relations:
            combined_kb.add_relation(relation)
    save_knowledge_graph(combined_kb, kg_filepath)

visualize_knowledge_graph(combined_kb.relations)

evaluation_results =  evaluate_mhr(qa_dataset, combined_kb, api_key)

csv_path = "evaluation_results_MHR.csv"
save_evaluation_results_to_csv(evaluation_results, csv_path)
print(f"Evaluation results saved to {csv_path}")