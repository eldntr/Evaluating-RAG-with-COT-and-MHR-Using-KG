from langchain_community.embeddings import HuggingFaceEmbeddings

def get_embeddings():
    modelPath = "sentence-transformers/all-MiniLM-L6-v2"
    model_kwargs = {'device': 0}
    encode_kwargs = {'normalize_embeddings': False}

    embeddings = HuggingFaceEmbeddings(
        model_name=modelPath,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
    )
    return embeddings
