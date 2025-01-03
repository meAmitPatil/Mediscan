import os
import uuid
from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

qdrant_client = QdrantClient(host="localhost", port=6333)
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
embedding_model = "text-embedding-ada-002"

def generate_embeddings(text):
    response = openai_client.embeddings.create(input=[text], model=embedding_model)
    return response.data[0].embedding

def ensure_collection_exists(collection_name, vector_size):
    collections = qdrant_client.get_collections().collections
    if any(col.name == collection_name for col in collections):
        print(f"Collection '{collection_name}' already exists.")
    else:
        # Create the collection if it doesn't exist
        qdrant_client.create_collection(
            collection_name=collection_name,
            vectors_config=qmodels.VectorParams(size=vector_size, distance="Cosine")
        )
        print(f"Collection '{collection_name}' created.")

def add_document_to_qdrant(text, metadata):
    try:
        embedding = generate_embeddings(text)
        
        collection_name = "medical_documents"
        ensure_collection_exists(collection_name, len(embedding))
        
        document_id = metadata.get("id") or str(uuid.uuid4())

        qdrant_client.upsert(
            collection_name=collection_name,
            points=[
                qmodels.PointStruct(
                    id=document_id,
                    vector=embedding,
                    payload={
                        "text": text,
                        **metadata
                    }
                )
            ]
        )
        
        print("Document successfully added to Qdrant.")
    except Exception as e:
        print(f"Error adding document to Qdrant: {e}")
