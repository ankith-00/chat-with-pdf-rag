import os
from dotenv import load_dotenv
from pinecone import Pinecone

load_dotenv()

# Pinecone DB Client
def get_pinecone_client() -> Pinecone:
    api_key = os.getenv("PINECONE_API_KEY")
    if not api_key:
        raise ValueError("[404] PINECONE-API-KEY Not Found !")    
    return Pinecone(api_key=api_key)