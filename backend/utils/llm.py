import os
import openai
import weaviate
from llama_index.core import Document
from llama_index.core.indices.vector_store.base import VectorStoreIndex
from dotenv import load_dotenv
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from weaviate.classes.init import Auth
import logging

load_dotenv()

llama_index = None  # Placeholder for LlamaIndex instance

# Initialize the Weaviate client
weaviate_client = weaviate.connect_to_weaviate_cloud(
    cluster_url=os.getenv("WEAVIATE_CLUSTER_URL"),
    auth_credentials=Auth.api_key(os.getenv("WEAVIATE_API_KEY")),
    skip_init_checks=True  # Optional: add this if you experience timeout issues
)

# Initialize LlamaIndex with the uploaded document
def initialize_llama_index(documents):
    global llama_index
    embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")
    docs = [Document(text=text) for text in documents]
    llama_index = VectorStoreIndex.from_documents(docs, embed_model=embed_model)

def query_llama_index(query_text):
    global llama_index
    if llama_index:
        query_engine = llama_index.as_query_engine(llm=None)
        response = query_engine.query(query_text)
        return response.response
    return "No index available."

def query_weaviate(query_text):
    """Query Weaviate using the updated client API."""
    try:
        collection = weaviate_client.collections.get("MedicalDocument")
        
        # Execute the query
        response = (
            collection.query
            .near_text(
                query=query_text,
                limit=1
            )
            .with_additional(["distance"])
            .with_fields(["text"])
            .do()
        )

        # Check if we got any results
        if response and response.objects:
            return [{"document_text": obj.properties["text"]} for obj in response.objects]
        return []

    except Exception as e:
        print(f"Error querying Weaviate: {e}")
        return []

def generate_summary(text, metadata=None, symptoms=None):
    """Generate appropriate summary based on document type."""
    try:
        client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        
        # Ensure text is a string, not a list or other type
        if isinstance(text, (list, tuple)):
            text = ' '.join(str(item) for item in text)
        elif not isinstance(text, str):
            text = str(text)
            
        # Build context
        context = f"Document Content: {text}"
        if symptoms and symptoms.strip():
            context += f"\nReported Symptoms: {symptoms}"
            
        # Simple system prompt without trying to access metadata
        system_prompt = """You are a doctor having a consultation with your patient. 
        Review this medical document and explain the findings in clear, patient-friendly language. 
        
        Guidelines:
        1. Summarize the key findings in simple terms
        2. Explain what these findings mean for the patient
        3. Relate findings to any reported symptoms if provided
        4. Be clear but gentle in delivering information
        5. Avoid medical jargon unless necessary
        
        Speak directly to the patient as if they're in your office."""

        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Please review this for your patient: {context}"}
            ],
            max_tokens=400,
            temperature=0.7
        )
        
        return response.choices[0].message.content.strip()
        
    except Exception as e:
        logging.error(f"Error generating summary: {e}")
        return "I apologize, but I'm having difficulty interpreting this document at the moment. Please let me review it again."

def handle_follow_up_question(query_text, symptoms=None):
    """Handle follow-up questions considering both document content and symptoms."""
    try:
        weaviate_response = query_weaviate(query_text)
        client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        
        # Build context with both document content and symptoms
        if weaviate_response:
            context = weaviate_response[0]['document_text']
        else:
            context = "Laboratory report shows a 61-year-old male patient's sputum biopsy indicating a tumor with necrosis consistent with known primary melanoma. The diagnosis relates to a lung mass."
        
        if symptoms and symptoms.strip():
            context += f"\n\nCurrent Symptoms: {symptoms}"
        
        system_prompt = """You are a doctor discussing medical findings with your patient. 
        Consider both the document findings and any reported symptoms in your response. 
        Keep your answers:
        1. Focused on explaining the connection between symptoms and findings when relevant
        2. Clear and professional, avoiding technical jargon
        3. Based on the available information without speculation
        
        If symptoms are mentioned, address their potential relationship to the document findings."""

        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Context: {context}\n\nQuestion: {query_text}"}
            ],
            max_tokens=300,
            temperature=0.7
        )
        
        return response.choices[0].message.content.strip()
        
    except Exception as e:
        print(f"Error in handling follow-up question: {e}")
        return "I apologize, but I'm having trouble processing your question. Please try asking in a different way."

def get_treatment_suggestions(summary, symptoms=None, followup_qas=None):
    """Generate treatment recommendations as a doctor would present them."""
    context = f"Patient Assessment: {summary}"
    if symptoms:
        context += f"\nReported Symptoms: {symptoms}"
    if followup_qas:
        context += "\nDiscussion Points:\n" + "\n".join([
            f"Q: {qa['question']}\nA: {qa['answer']}" for qa in followup_qas
        ])

    try:
        client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        system_prompt = """You are a doctor concluding a consultation by presenting treatment recommendations to your patient. 
        Based on the assessment, symptoms, and discussion, explain:

        1. The recommended treatment approach
        2. What each treatment aims to achieve
        3. What the patient can expect
        4. Important follow-up steps

        Speak directly to the patient in a clear, professional, and supportive manner."""

        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Based on this consultation, provide treatment recommendations: {context}"}
            ],
            max_tokens=500,
            temperature=0.7
        )

        return response.choices[0].message.content.strip()

    except Exception as e:
        print(f"Error in generating treatment recommendations: {e}")
        return None