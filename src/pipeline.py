from transformers import pipeline
import torch

def generate_answer(retriever, hybrid_results, question):
    pipe = pipeline(
        "text-generation", 
        model="TinyLlama/TinyLlama-1.1B-Chat-v1.0", 
        torch_dtype=torch.bfloat16, 
        device_map="auto"
    )

    retrived_text = '\n\n'.join([doc.page_content for doc in hybrid_results])
    messages = [
        {
            "role": "system",
            "content": f'''Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer.
            context: {retrived_text}'''
        },
        {"role": "user", "content": f'{question}'}
    ]
    prompt = pipe.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    outputs = pipe(prompt, max_new_tokens=256, do_sample=True, temperature=0.7, top_k=50, top_p=0.95)
    return outputs[0]["generated_text"][len(prompt):]
