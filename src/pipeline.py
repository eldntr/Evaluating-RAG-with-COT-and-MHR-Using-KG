from langchain.chains import ConversationalRetrievalChain
from langchain_huggingface import HuggingFacePipeline  # Perbaikan impor
from transformers import pipeline
import torch
import re

def generate_answer(retriever, history, question):
    """ Conversational Retrieval Chain dengan cek jawaban kosong """
    pipe = pipeline(
        "text-generation",
        model="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )

    llm = HuggingFacePipeline(pipeline=pipe)  # Menggunakan versi baru

    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        return_source_documents=True
    )

    response = qa_chain({"question": question, "chat_history": history})

    if not response["answer"]:  # Cek jika jawaban kosong
        print("⚠️  Model tidak bisa memberikan jawaban yang valid.")
        return "Maaf, saya tidak bisa menjawab pertanyaan ini."

    return response["answer"]

def clean_answer(answer):
    """ Membersihkan jawaban dari redundansi dan format yang aneh """
    answer = re.sub(r"\n+", "\n", answer.strip())  # Hapus newline berlebih
    answer = re.sub(r"(Answer:|Response:)", "", answer, flags=re.IGNORECASE).strip()  # Hapus awalan berlebih
    return answer