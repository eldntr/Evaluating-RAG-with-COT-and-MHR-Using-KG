import time  # Tambahkan impor modul time
import csv  # Tambahkan impor modul csv
from rouge import Rouge
from sentence_transformers import SentenceTransformer, util
from src.generation import generate_answer  
from tqdm import tqdm  # Tambahkan impor pustaka tqdm
from src.entity_extraction import extract_entities
from src.knowledge_graph import collect_kg_info, format_kg
from src.retriever import search_hybrid

semantic_model = SentenceTransformer("all-MiniLM-L6-v2")

def calculate_rouge_score(prediction, reference):
    rouge = Rouge()
    scores = rouge.get_scores(prediction, reference)
    return scores[0]['rouge-l']['f']

def calculate_semantic_similarity(prediction, reference):
    pred_embedding = semantic_model.encode(prediction)
    ref_embedding = semantic_model.encode(reference)
    similarity = util.cos_sim(pred_embedding, ref_embedding).item()
    return similarity

def calculate_keyword_score(prediction, reference):
    keywords = reference.split()
    prediction_lower = prediction.lower()
    match_count = sum(keyword.lower() in prediction_lower for keyword in keywords)
    return match_count / len(keywords) if keywords else 0

def save_evaluation_results_to_csv(results, csv_path):
    keys = results[0].keys()
    with open(csv_path, 'w', newline='', encoding='utf-8') as output_file:
        dict_writer = csv.DictWriter(output_file, fieldnames=keys)
        dict_writer.writeheader()
        dict_writer.writerows(results)

def evaluate_model(qa_dataset, embedding_model, generation_model, index, bm25, documents, api_key):
    results = []
    for qa_entry in tqdm(qa_dataset, desc="Evaluating"):  # Bungkus iterasi dengan tqdm
        question = qa_entry["question"]
        reference_answer = qa_entry["answer"]
        start_time = time.time()  
        generated_answer = generate_answer(question, embedding_model, generation_model, index, bm25, documents, api_key, prompt_type="cot")
        end_time = time.time() 
        generation_time = end_time - start_time  
        rouge_score = calculate_rouge_score(generated_answer, reference_answer)
        semantic_score = calculate_semantic_similarity(generated_answer, reference_answer)
        keyword_score = calculate_keyword_score(generated_answer, reference_answer)
        retrieved_docs = search_hybrid(question, embedding_model, index, bm25, documents)  
        results.append({
            "Question": question,
            "Actual Answer": reference_answer,
            "Generated Answer": generated_answer,
            "Retrieved Documents": retrieved_docs,
            "ROUGE-L F1 Score": rouge_score,
            "Semantic Similarity": semantic_score,
            "Keyword-Based Score": keyword_score,
            "Generation Time (seconds)": generation_time, 
        })
    return results

def evaluate_mhr(qa_dataset, combined_kb, api_key):
    results = []
    
    for qa_entry in tqdm(qa_dataset, desc="Evaluating MHR", unit="entry"):
        question = qa_entry["question"]
        reference_answer = qa_entry["answer"]
        
        start_time = time.time()
        entities = extract_entities(question, verbose=False)
        kg_info = collect_kg_info(entities, combined_kb)
        formatted_kg = format_kg(kg_info)

        generated_answer = generate_answer(
            question,
            api_key=api_key,
            kg_info=kg_info,
            prompt_type="mhr"
        )
        end_time = time.time()
        generation_time = end_time - start_time

        rouge_score = calculate_rouge_score(generated_answer, reference_answer)
        semantic_score = calculate_semantic_similarity(generated_answer, reference_answer)
        keyword_score = calculate_keyword_score(generated_answer, reference_answer)

        results.append({
            "Question": question,
            "Actual Answer": reference_answer,
            "Generated Answer": generated_answer,
            "Entities Extracted": entities,
            "Knowledge Graph Info": formatted_kg,
            "ROUGE-L F1 Score": rouge_score,
            "Semantic Similarity": semantic_score,
            "Keyword-Based Score": keyword_score,
            "Generation Time (seconds)": generation_time,
        })
    
    return results

def evaluate_default(qa_dataset, api_key):
    results = []
    
    for qa_entry in tqdm(qa_dataset, desc="Evaluating Default", unit="entry"):
        question = qa_entry["question"]
        reference_answer = qa_entry["answer"]
        
        start_time = time.time()
        generated_answer = generate_answer(
            question,
            api_key=api_key,
            prompt_type="default"
        )
        end_time = time.time()
        generation_time = end_time - start_time

        rouge_score = calculate_rouge_score(generated_answer, reference_answer)
        semantic_score = calculate_semantic_similarity(generated_answer, reference_answer)
        keyword_score = calculate_keyword_score(generated_answer, reference_answer)

        results.append({
            "Question": question,
            "Actual Answer": reference_answer,
            "Generated Answer": generated_answer,
            "ROUGE-L F1 Score": rouge_score,
            "Semantic Similarity": semantic_score,
            "Keyword-Based Score": keyword_score,
            "Generation Time (seconds)": generation_time,
        })
    
    return results