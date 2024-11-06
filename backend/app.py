import streamlit as st
import io
import os
from utils.extractors import extract_text_from_file
from utils.embeddings import add_document_to_weaviate
from utils.llm import generate_summary, handle_follow_up_question, get_treatment_suggestions
import asyncio
from lmnt.api import Speech
from dotenv import load_dotenv


# Load environment variables
load_dotenv()
LMNT_API_KEY = os.getenv("LMNT_API_KEY")

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

    # Custom CSS for improved UI
    st.markdown("""
        <style>
        body { background-color: #f4f7fa; font-family: 'Arial', sans-serif; }
        .header { text-align: center; padding: 2rem; background-color: #007BFF; color: white; border-radius: 10px; margin-bottom: 2rem; }
        .card { background-color: #ffffff; border-radius: 10px; padding: 20px; box-shadow: 0 4px 8px rgba(0,0,0,0.1); margin-bottom: 20px; }
        .button { background-color: #28a745; color: white; border-radius: 20px; height: 3em; width: 100%; border: none; cursor: pointer; transition: background-color 0.3s; }
        .button:hover { background-color: #218838; }
        .footer { text-align: center; padding: 20px; background-color: #007BFF; color: white; border-radius: 10px; margin-top: 20px; }
        .image { width: 100%; border-radius: 10px; margin-bottom: 20px; }
        </style>
    """, unsafe_allow_html=True)

    st.markdown("<div class='header'><h1>🏥 MediScan</h1><p>AI-Powered Medical Document Analysis</p></div>", unsafe_allow_html=True)

    col1, col2 = st.columns([2, 1])

    with col1:
        st.image("/Users/amit/Documents/Mediscan/backend/image.jpg", use_column_width=True, caption="Upload your medical documents for analysis.")

        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("### 📋 Document Upload")
        uploaded_file = st.file_uploader("Upload your medical document", type=["pdf", "png", "jpg", "jpeg", "docx"])
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("### 🤒 Symptoms")
        symptoms = st.text_area("Enter patient symptoms (optional)", placeholder="Example: fever, headache, fatigue...")
        st.markdown("</div>", unsafe_allow_html=True)

    with col2:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("### ℹ️ How it works")
        st.markdown("""
            <ol>
                <li><strong>Upload Document</strong>: Share your medical document</li>
                <li><strong>Add Symptoms</strong>: Describe any current symptoms</li>
                <li><strong>Get Analysis</strong>: Receive doctor-like interpretation</li>
                <li><strong>Ask Questions</strong>: Get clarification on findings</li>
                <li><strong>Review Plan</strong>: Get treatment suggestions</li>
            </ol>
        """, unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    if uploaded_file and not st.session_state.processing_complete:
        try:
            with st.spinner("🔄 Doctor is reviewing your document..."):
                document_text, metadata = process_uploaded_file(uploaded_file)

                if document_text:
                    add_document_to_weaviate(document_text, metadata or {})
                    if st.session_state.summary is None:
                        st.session_state.summary = generate_summary(document_text, symptoms=symptoms)
                    st.session_state.processing_complete = True
                    st.success("✅ Doctor has completed the review")
                else:
                    st.error("❌ Unable to properly review the document")

        except Exception as e:
            st.error(f"An error occurred during the consultation: {str(e)}")

    if st.session_state.processing_complete:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("### 📑 Document Summary")
        st.markdown(st.session_state.summary)
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("### ❓ Follow-up Questions")
        question = st.text_input("Ask a question about the document", placeholder="Enter your question here...")

        if st.button("Ask Doctor", key="ask_doctor"):
            if question:
                with st.spinner("Doctor is considering your question..."):
                    answer = handle_follow_up_question(question, symptoms)
                    st.session_state.followup_qas.append({"question": question, "answer": answer})
            else:
                st.warning("Please enter a question before asking the doctor.")

        if st.session_state.followup_qas:
            for qa in st.session_state.followup_qas:
                st.markdown(f"**Q:** {qa['question']}")
                st.markdown(f"**A:** {qa['answer']}")
        st.markdown("</div>", unsafe_allow_html=True)

        if st.button("💊 Get Treatment Suggestions"):
            with st.spinner("Doctor is preparing treatment recommendations..."):
                if st.session_state.treatment_plan is None:
                    st.session_state.treatment_plan = get_treatment_suggestions(
                        st.session_state.summary,
                        symptoms=symptoms,
                        followup_qas=st.session_state.followup_qas
                    )

        if st.session_state.treatment_plan:
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.markdown("### 💊 Treatment Suggestions")
            st.markdown(st.session_state.treatment_plan)

            # Add the "Read Aloud" button
            if st.button("🔊 Read Aloud", key="read_aloud"):
                with st.spinner("Preparing audio..."):
                    audio_file_path = asyncio.run(read_treatment_suggestion(st.session_state.treatment_plan))
                    
                    # Load and play the audio file using Streamlit's st.audio
                    try:
                        with open(audio_file_path, "rb") as audio_file:
                            st.audio(audio_file.read(), format="audio/mp3")
                    except FileNotFoundError:
                        st.error("Audio file not found. Please check the file path.")
            st.markdown("</div>", unsafe_allow_html=True)


    st.markdown("<div class='footer'><p>MediScan - AI-Powered Medical Document Analysis</p><p>© 2024 All rights reserved</p></div>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
