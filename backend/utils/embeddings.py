# utils/embeddings.py
import os
from dotenv import load_dotenv
import weaviate
from sentence_transformers import SentenceTransformer
from weaviate.classes.init import Auth

load_dotenv()

# Initialize Weaviate and Sentence Transformer
weaviate_client = weaviate.connect_to_weaviate_cloud(
    cluster_url=os.getenv("WEAVIATE_CLUSTER_URL"),
    auth_credentials=Auth.api_key(os.getenv("WEAVIATE_API_KEY")),
    skip_init_checks=True  # Optional: add this if you experience timeout issues
)
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

def generate_embeddings(text):
    return embedding_model.encode(text).tolist()

def add_document_to_weaviate(text, metadata):
    """Add a medical document to the Weaviate instance using the updated API."""
    try:
        # Check if collection exists and create if it doesn't
        collection_name = "MedicalDocument"
        
        try:
            collection = weaviate_client.collections.get(collection_name)
        except weaviate.exceptions.WeaviateCollectionNotFoundException:
            # Create collection if it doesn't exist
            collection = weaviate_client.collections.create(
                name=collection_name,
                properties=[
                    {
                        "name": "text",
                        "dataType": "text",
                        "description": "Medical document text"
                    }
                ],
                vectorizer_config={
                    "vectorizer": "text2vec-transformers"
                }
            )

        # Add the document
        collection.data.insert({
            "text": text,
            **metadata
        })
        
        print("Document successfully added to Weaviate.")
        
    except Exception as e:
        print(f"Error adding document to Weaviate: {e}")
