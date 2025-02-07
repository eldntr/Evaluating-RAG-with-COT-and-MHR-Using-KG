from datasets import load_dataset

def load_and_process_data(dataset_name="hotpot_qa", subset_name="distractor"):
    dataset = load_dataset(dataset_name, subset_name)
    qa_dataset = []
    documents = []
    seen_texts = set()

    def process_subset(subset_data, subset_name):
        for entry in subset_data:
            question = entry["question"]
            answer = entry["answer"]
            qa_dataset.append({"question": question, "answer": answer, "subset": subset_name})
            for title, sentences in zip(entry["context"]["title"], entry["context"]["sentences"]):
                paragraph = " ".join(sentences)
                if paragraph not in seen_texts:
                    documents.append({"title": title, "text": paragraph})
                    seen_texts.add(paragraph)

    # process_subset(dataset['train'], "train")
    process_subset(dataset['validation'], "validation")

    return qa_dataset, documents
