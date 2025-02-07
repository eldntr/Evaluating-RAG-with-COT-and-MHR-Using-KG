from dotenv import load_dotenv
from src.data_processing import load_and_process_data  
from src.evaluation import evaluate_default, save_evaluation_results_to_csv
import os

load_dotenv()

api_key = os.getenv("API_KEY")
if not api_key:
    raise ValueError("API_KEY not found in environment variables")

qa_dataset, documents = load_and_process_data()
qa_dataset = qa_dataset[:10]
documents = documents[:10]

evaluation_results =  evaluate_default(qa_dataset,api_key)

csv_path = "evaluation_results_Default.csv"
save_evaluation_results_to_csv(evaluation_results, csv_path)
print(f"Evaluation results saved to {csv_path}")