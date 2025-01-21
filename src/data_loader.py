from langchain_community.document_loaders import HuggingFaceDatasetLoader

def load_data(dataset_name, page_content_column):
    loader = HuggingFaceDatasetLoader(dataset_name, page_content_column)
    data = loader.load()
    return data
