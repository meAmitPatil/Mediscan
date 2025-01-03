import os
import openai
from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels
from llama_index.core import Document
from llama_index.core.indices.vector_store.base import VectorStoreIndex
from dotenv import load_dotenv
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
import logging

load_dotenv()

llama_index = None


qdrant_client = QdrantClient(host="localhost", port=6333)


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

def query_qdrant(query_text):
    try:
        embedding = generate_embeddings(query_text)
        
        response = qdrant_client.search(
            collection_name="medical_documents",
            query_vector=embedding,
            limit=1,
            with_payload=True,
            with_vectors=False
        )

        if response:
            return [{"document_text": hit.payload["text"]} for hit in response]
        return []

    except Exception as e:
        print(f"Error querying Qdrant: {e}")
        return []

def generate_embeddings(text):
    client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    response = client.embeddings.create(
        input=[text],
        model="text-embedding-ada-002"
    )
    return response.data[0].embedding

def generate_summary(text, metadata=None, symptoms=None):
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

def handle_follow_up_question(query_text, symptoms=None, context=None):
    """Handle follow-up questions considering document content, symptoms, and previous interactions."""
    try:
        client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        
        full_context = ""
        
        if context:
            full_context += f"Previous Interactions:\n{context}\n\n"
        
        if symptoms and symptoms.strip():
            full_context += f"Reported Symptoms: {symptoms}\n\n"
        
        full_context += f"Question: {query_text}"

        system_prompt = """You are a doctor in a follow-up conversation with a patient. The patient has already received an initial summary 
        of their medical document, and now they have additional questions for you.

        Your goal is to:
        1. Address their question clearly and compassionately, based on previous information and any symptoms they've reported.
        2. Relate your answer to the previous findings as much as possible, providing clarity without unnecessary jargon.
        3. Reassure the patient and make them feel comfortable with the information.

        Please respond in a way that simulates a patient-friendly and empathetic consultation."""

        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": full_context}
            ],
            max_tokens=200,
            temperature=0.7
        )
        
        return response.choices[0].message.content.strip()
        
    except Exception as e:
        print(f"Error in handling follow-up question: {e}")
        return "I apologize, but I'm having trouble processing your question. Please try asking in a different way."

def get_treatment_suggestions(summary, symptoms=None, followup_qas=None):
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
        Based on the findings, symptoms, and previous discussion, please provide:

        1. An overview of the recommended treatment approach.
        2. An explanation of what each treatment is expected to accomplish.
        3. Clear, supportive information on what the patient should expect next.
        4. Any follow-up actions the patient should take.

        Use a tone that is professional, empathetic, and supportive to ensure the patient feels reassured and well-informed."""

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
