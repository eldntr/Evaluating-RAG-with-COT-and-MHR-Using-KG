import requests
import json
import re
from src.retriever import search  # Import the search function

DEFAULT_PROMPT = """
Answer the following question directly and concisely. Ensure the answer is grammatically correct and relevant to the question.

**Question**: {question}

**Answer**:
"""

COT_PROMPT = """
**Question**: {question}
    
**Context**: {context}
    
**Instructions**:
1. Analyze the question and identify key elements such as entities (e.g., names, organizations), dates, or relationships.
2. Extract ONLY the most relevant sentences or phrases from the context that directly address the identified key elements.
3. Formulate a concise answer using the extracted information. Ensure the answer is grammatically correct and directly answers the question.
4. Follow this exact format:
    - Final Answer: [A single sentence or phrase that directly answers the question. Avoid unnecessary details.]
    
**Answer for This Question**:
"""

MHR_PROMPT = """
**Instructions**:
Use the information provided in the Knowledge Graph to answer the question. Focus only on relevant details and provide a concise answer.

**Knowledge Graph Information**:
{formatted_kg}

**Question**: {question}
"""

def generate_answer(question, embedding_model=None, generation_model="mistralai/mistral-small-24b-instruct-2501", index=None, documents=None, api_key=None, kg_info=None, prompt_type="default"):
    if prompt_type == "cot":
        if embedding_model is None or index is None or documents is None:
            raise ValueError("Embedding model, index, and documents are required for CoT prompt.")
        results = search(question, embedding_model, index, documents)
        retrieved_docs = [doc["document"] for doc in results[:5]]
        context = "\n".join(retrieved_docs)
        formatted_prompt = COT_PROMPT.format(question=question, context=context)
    elif prompt_type == "default":
        formatted_prompt = DEFAULT_PROMPT.format(question=question)
    elif prompt_type == "mhr":
        if kg_info is None:
            raise ValueError("Knowledge graph information is required for MHR prompt.")
        formatted_kg = "\n".join([f"{k}: {v}" for k, v in kg_info.items()])
        formatted_prompt = MHR_PROMPT.format(question=question, formatted_kg=formatted_kg)
    else:
        raise ValueError("Invalid prompt type. Choose 'cot' or 'default'.")

    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    data = {
        "model": generation_model,
        "messages": [{"role": "user", "content": formatted_prompt}],
        "max_tokens": 100,
        "temperature": 0.1,
        "top_p": 0.95,
        "repetition_penalty": 1.1,
        "num_return_sequences": 1
    }

    try:
        response = requests.post(url, headers=headers, data=json.dumps(data))
        response.raise_for_status()
        api_response = response.json()
        generated_text = api_response["choices"][0]["message"]["content"]
        match = re.search(r"(?<=\*\*Answer for This Question\*\*:)\s*(.*?)(\n\n|\Z)", generated_text, re.DOTALL) if prompt_type == "cot" else re.search(r"(?<=\*\*Answer\*\*:)\s*(.*?)(\n\n|\Z)", generated_text, re.DOTALL)
        return match.group(1).strip() if match else generated_text.strip()
    except requests.exceptions.RequestException as e:
        print(f"API Error: {e}")
        return "Maaf, terjadi kesalahan saat menghubungi API."