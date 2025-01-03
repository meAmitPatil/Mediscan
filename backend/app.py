import streamlit as st
import os
import asyncio
from utils.extractors import extract_text_from_file
from utils.embeddings import add_document_to_qdrant
from utils.llm import generate_summary, handle_follow_up_question, get_treatment_suggestions
from lmnt.api import Speech
from dotenv import load_dotenv
from mem0 import MemoryClient

# Load environment variables
load_dotenv()
LMNT_API_KEY = os.getenv("LMNT_API_KEY")
MEM0_API_KEY = os.getenv("MEM0_API_KEY")

# Configure Mem0 with Qdrant for memory storage and retrieval
config = {
    "vector_store": {
        "provider": "qdrant",
        "config": {
            "collection_name": "medical_documents",
            "host": "localhost",
            "port": 6333,
        }
    }
}
memory_client = MemoryClient(api_key=MEM0_API_KEY)

# Define the async TTS function
async def read_treatment_suggestion(suggestion_text):
    async with Speech() as speech:
        synthesis = await speech.synthesize(suggestion_text, 'lily')
        audio_file = 'treatment_suggestion.mp3'
        with open(audio_file, 'wb') as f:
            f.write(synthesis['audio'])
    return audio_file

# Initialize session state for Streamlit
def initialize_session_state():
    if "processing_complete" not in st.session_state:
        st.session_state.processing_complete = False
    if "summary" not in st.session_state:
        st.session_state.summary = None
    if "followup_qas" not in st.session_state:
        st.session_state.followup_qas = []
    if "treatment_plan" not in st.session_state:
        st.session_state.treatment_plan = None

def process_uploaded_file(uploaded_file):
    """Process the uploaded file directly from memory."""
    try:
        bytes_data = uploaded_file.getvalue()
        document_text, metadata = extract_text_from_file(bytes_data, uploaded_file.name)
        return document_text, metadata
    except Exception as e:
        raise Exception(f"Error processing file: {e}")

def main():
    initialize_session_state()

    # Sidebar Guide
    st.sidebar.title("Guide")
    st.sidebar.markdown("""
    **How to use MediScan:**
    1. **Upload Document**: Share your medical document for analysis.
    2. **Describe Symptoms**: Enter any symptoms you have (optional).
    3. **Get Analysis**: Receive a summary of findings.
    4. **Ask Questions**: Ask follow-up questions for clarity.
    5. **Treatment Suggestions**: Review doctor-like treatment recommendations.
    """)

    # UI setup
    st.markdown("<div class='header'><h1>🏥 MediScan</h1><p>AI-Powered Medical Document Analysis</p></div>", unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Upload your medical document", type=["pdf", "png", "jpg", "jpeg", "docx"])
    symptoms = st.text_area("Enter patient symptoms (optional)", placeholder="Example: fever, headache, fatigue...")

    if uploaded_file and not st.session_state.processing_complete:
        try:
            with st.spinner("🔄 Doctor is reviewing your document..."):
                
                if uploaded_file.type in ["image/png", "image/jpeg", "image/jpg", "image/dicom"]:
                    st.image(uploaded_file, caption="Uploaded Medical Image", use_column_width=True)
                
                document_text, metadata = process_uploaded_file(uploaded_file)

                if document_text:
                    add_document_to_qdrant(document_text, metadata or {})
                    st.session_state.summary = generate_summary(document_text, symptoms=symptoms)
                    st.session_state.processing_complete = True
                    st.success("✅ Doctor has completed the review")
                else:
                    st.error("❌ Unable to properly review the document")

        except Exception as e:
            st.error(f"An error occurred during the consultation: {str(e)}")

    if st.session_state.processing_complete:
        st.markdown("### 📑 Document Summary")
        st.markdown(st.session_state.summary)

        question = st.text_input("Ask a question about the document", placeholder="Enter your question here...")

        if st.button("Ask Doctor"):
            if question:
                with st.spinner("Doctor is considering your question..."):
                    # Retrieve previous relevant memories
                    previous_interactions = memory_client.search(query=question, user_id="patient_id")
                    
                    # Construct context from previous interactions
                    context = "\n".join([mem.get("memory", "") for mem in previous_interactions])

                    # Generate an answer with context
                    answer = handle_follow_up_question(question, symptoms, context=context)
                    st.session_state.followup_qas.append({"question": question, "answer": answer})

                    # Save question and answer to Mem0 memory
                    messages = [
                        {"role": "user", "content": question},
                        {"role": "assistant", "content": answer}
                    ]
                    memory_client.add(messages, user_id="patient_id")

            else:
                st.warning("Please enter a question before asking the doctor.")

        if st.session_state.followup_qas:
            for qa in st.session_state.followup_qas:
                st.markdown(f"**Q:** {qa['question']}")
                st.markdown(f"**A:** {qa['answer']}")

        if st.button("💊 Get Treatment Suggestions"):
            with st.spinner("Doctor is preparing treatment recommendations..."):
                st.session_state.treatment_plan = get_treatment_suggestions(
                    st.session_state.summary, symptoms=symptoms, followup_qas=st.session_state.followup_qas
                )

        if st.session_state.treatment_plan:
            st.markdown("### 💊 Treatment Suggestions")
            st.markdown(st.session_state.treatment_plan)

            if st.button("🔊 Read Aloud"):
                with st.spinner("Preparing audio..."):
                    audio_file_path = asyncio.run(read_treatment_suggestion(st.session_state.treatment_plan))
                    with open(audio_file_path, "rb") as audio_file:
                        st.audio(audio_file.read(), format="audio/mp3")

    st.markdown("<div class='footer'><p>MediScan - AI-Powered Medical Document Analysis</p><p>© 2024 All rights reserved</p></div>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
