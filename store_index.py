import os
from dotenv import load_dotenv
load_dotenv()

from src.helper import download_embeddings, filter_to_minimal_docs, load_pdf_files, text_split

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
DATA_FOLDER = os.getenv("DATA_FOLDER")

extracted_data = load_pdf_files(DATA_FOLDER)
minimal_docs = filter_to_minimal_docs(extracted_data)
text_chunk = text_split(minimal_docs)

embedding = download_embeddings()

from pinecone import Pinecone

pinecone_api_key = PINECONE_API_KEY

pc = Pinecone(api_key=pinecone_api_key)

from pinecone import ServerlessSpec

index_name = "medical-chatbot"

if not pc.has_index(index_name):
    pc.create_index(
        name=index_name,
        dimension=384,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )

#index = pc.index(index_name)

index = pc.Index(index_name)

from langchain_pinecone import PineconeVectorStore

doc_search = PineconeVectorStore.from_documents(
    documents=text_chunk,
    embedding=embedding,
    index_name=index_name
)