from langchain.chains import ConversationalRetrievalChain
from langchain_huggingface import HuggingFacePipeline
from langchain.prompts import PromptTemplate
from transformers import pipeline
import torch
import re

COT_PROMPT = PromptTemplate(
    input_variables=["question", "context"],
    template="""
    **Question**: {question}
    **Context**: {context}
    **Instructions**:
    1. Analyze the question and identify key elements (e.g., dates, locations, entities).
    2. Extract relevant information from the context.
    3. Formulate a concise answer using the identified information.
    **Example Format**:
    1. Step 1: Identified keywords "[key_term1]" and "[key_term2]".
    2. Step 2: Found relevant information: "[excerpt_from_context]".
    Final Answer: [Concise answer based on context]
    **Answer for This Question**:
    """
)

def generate_answer(retriever, history, question, prompt_type="default"):
    # Ambil dan format dokumen sesuai kebutuhan
    retrieved_docs = retriever.get_relevant_documents(question)
    
    if prompt_type == "cot":
        context = "\n".join([doc.page_content for doc in retrieved_docs[:3]])
        selected_prompt = COT_PROMPT
    else:
        context = "\n".join([doc.page_content for doc in retrieved_docs])
        selected_prompt = PromptTemplate.from_template("{question}\n\nContext: {context}")

    # Inisialisasi model
    pipe = pipeline(
        "text-generation",
        model="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    llm = HuggingFacePipeline(pipeline=pipe)

    # Gunakan ConversationalRetrievalChain dengan prompt template
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        combine_docs_chain_kwargs={"prompt": selected_prompt}  # <-- Gunakan objek PromptTemplate
    )

    # Eksekusi dengan question dan context yang benar
    response = qa_chain({"question": question, "chat_history": history})
    return response["answer"] if response["answer"] else "Maaf, saya tidak bisa menjawab."

def clean_answer(answer):
    answer = re.sub(r"(Jawaban Akhir:)", "\n\\1", answer, flags=re.IGNORECASE)
    answer = re.sub(r"\n{2,}", "\n", answer.strip())
    return answer