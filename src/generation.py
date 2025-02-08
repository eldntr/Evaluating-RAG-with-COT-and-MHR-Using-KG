import requests
import json
import re
from src.retriever import search_hybrid  # Import the search function

DEFAULT_PROMPT = """
Answer the following question directly and concisely. Ensure the answer is grammatically correct and relevant to the question.

**Question**: {question}

**Answer**:
"""

COT_PROMPT = """
**Question**: {question}

**Context**: {context}

**Instructions**:
1. **Analyze the Question**:
   - Identify key elements such as entities (e.g., names, organizations), dates, or relationships.
   - Determine what information is needed to answer the question.
   
2. **Reasoning Process**:
   - Use the context to logically connect the key elements.
   - Show your step-by-step reasoning process. Explain how you arrive at each conclusion.
   
3. **Extract Relevant Information**:
   - Highlight ONLY the most relevant sentences or phrases from the context that support your reasoning.
   
4. **Formulate the Final Answer**:
   - Based on your reasoning, provide a concise and grammatically correct answer.
   - Ensure the answer directly addresses the question.
   
**Final Answer**:
[A single sentence or phrase that directly answers the question. Avoid unnecessary details.]
"""

MHR_PROMPT = """
**Instructions**:
Use the information provided in the Knowledge Graph (KG), which has been processed using Breadth-First Search (BFS) to identify relevant multi-hop paths. Answer the question based on the following steps:

**Step-by-Step Guidance**:
1. **Identify Key Entities**: Extract the main entities mentioned in the question.
2. **Locate Relevant Information**: Use the precomputed BFS paths in the KG to find relationships between the key entities.
3. **Validate Relevance**: Ensure all information used directly answers the question.

**Constraints**:
- The answer must be reached within a maximum of 3 hops.
- Only use information explicitly provided in the KG.

**Knowledge Graph Information**:
{formatted_kg}

**Question**: {question}

**Final Answer**:
[A single sentence or phrase that directly answers the question. Avoid unnecessary details.]
"""

def generate_answer(question, embedding_model=None, generation_model="mistralai/mistral-small-24b-instruct-2501", index=None, bm25=None, documents=None, api_key=None, kg_info=None, prompt_type="default"):
    if prompt_type == "cot":
        if embedding_model is None or index is None or documents is None:
            raise ValueError("Embedding model, index, and documents are required for CoT prompt.")
        results = search_hybrid(question, embedding_model, index, bm25, documents)
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
        "max_tokens": 300,
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
        # return match.group(1).strip() if match else generated_text.strip()
        return generated_text
    except requests.exceptions.RequestException as e:
        print(f"API Error: {e}")
        return "Maaf, terjadi kesalahan saat menghubungi API."