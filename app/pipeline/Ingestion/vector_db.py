import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import ServerlessSpec

# Pinecone Client Import
from ...config.pinecone_client import get_pinecone_client

load_dotenv()

INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "pdf-rag-index")
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
EMBEDDING_DIMENSION = 384


def get_vectorstore_retriever(pdf_path: str, k: int = 4):
    # 1. Get the raw connection client
    pc = get_pinecone_client()

    # 2. Initialize local HuggingFace embeddings
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

    # 3. Check/Create Index via the client
    existing_indexes = [idx["name"] for idx in pc.list_indexes()]
    if INDEX_NAME not in existing_indexes:
        print(f"Creating Pinecone serverless index: {INDEX_NAME}...")
        pc.create_index(
            name=INDEX_NAME,
            dimension=EMBEDDING_DIMENSION,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1")
        )

    # 4. Check index stats to see if we need to seed data
    index = pc.Index(INDEX_NAME)
    stats = index.describe_index_stats()
    total_vectors = stats.get("total_vector_count", 0)

    if total_vectors == 0:
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"Target PDF '{pdf_path}' not found. Please provide a valid file.")
        
        print(f"Index is empty. Processing and uploading PDF to Pinecone: {pdf_path}...")
        loader = PyPDFLoader(pdf_path)
        docs = loader.load()

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, 
            chunk_overlap=200
        )
        splits = text_splitter.split_documents(docs)

        # Upload vectors to Pinecone cloud
        vectorstore = PineconeVectorStore.from_documents(
            documents=splits, 
            embedding=embeddings, 
            index_name=INDEX_NAME
        )
        print("Vectors successfully uploaded to Pinecone!")
    else:
        print(f"Connected to existing Pinecone index '{INDEX_NAME}' ({total_vectors} vectors found).")
        vectorstore = PineconeVectorStore(index_name=INDEX_NAME, embedding=embeddings)

    # 5. Return retriever instance
    return vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": k}
    )