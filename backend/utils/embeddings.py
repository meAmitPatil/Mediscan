import os
import uuid
from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels
from sentence_transformers import SentenceTransformer
from openai import OpenAI
from dotenv import load_dotenv
load_dotenv()

# Initialize Qdrant client, OpenAI client, and embedding model
qdrant_client = QdrantClient(host="localhost", port=6333)
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
embedding_model = "text-embedding-3-small"  # or "text-embedding-3-large" if you prefer

def generate_embeddings(text):
    """Generate embeddings for a given text using OpenAI embeddings."""
    response = openai_client.embeddings.create(input=[text], model=embedding_model)
    return response.data[0].embedding

def ensure_collection_exists(collection_name, vector_size):
    """Ensure that the specified Qdrant collection exists, create it if it does not."""
    try:
        qdrant_client.get_collection(collection_name)
        print(f"Collection '{collection_name}' already exists.")
    except qmodels.CollectionNotFoundException:
        # Create the collection if it doesn't exist
        qdrant_client.create_collection(
            collection_name=collection_name,
            vectors_config=qmodels.VectorParams(size=vector_size, distance="Cosine")
        )
        print(f"Collection '{collection_name}' created.")

def add_document_to_qdrant(text, metadata):
    """Add a document with embeddings to the Qdrant instance."""
    try:
        # Generate embeddings
        embedding = generate_embeddings(text)
        
        # Ensure the collection exists
        collection_name = "medical_documents"
        ensure_collection_exists(collection_name, len(embedding))
        
        # Generate a unique ID if not provided in metadata
        document_id = metadata.get("id") or str(uuid.uuid4())

        # Insert document into Qdrant
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

