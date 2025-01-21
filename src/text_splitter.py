from langchain.text_splitter import RecursiveCharacterTextSplitter

def split_text(data):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    docs = text_splitter.split_documents(data)
    return docs
