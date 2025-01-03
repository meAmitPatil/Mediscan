# MediScan 🏥 - AI-Powered Medical Document Analysis

[![Status](https://img.shields.io/badge/Status-Active-brightgreen)](https://github.com/meAmitPatil/MediScan)

MediScan is an AI-powered application designed to assist users in analyzing medical documents and images, providing patient-friendly summaries, answering follow-up questions, and suggesting treatment options. It leverages state-of-the-art AI models and integrates advanced vector search with Qdrant to store and retrieve insights.

---

## 🚀 **Features**
- **Document Analysis**: Upload PDFs, Word documents, or images for extraction and interpretation.
- **Medical Image Analysis**: Process X-rays, CT scans, and other medical images with automated insights.
- **AI-Powered Summaries**: Patient-friendly summaries generated using GPT-3.5.
- **Follow-Up Questions**: Context-aware Q&A for clarity on findings.
- **Treatment Suggestions**: Empathetic and informative recommendations based on analysis.
- **Text-to-Speech Integration**: Reads treatment suggestions aloud for accessibility.
- **Memory Integration**: Powered by Mem0 and Qdrant for personalized context retrieval.

---

## 🛠️ **Technologies Used**
- **Backend**: Python, FastAPI
- **Frontend**: Streamlit
- **AI Models**: OpenAI GPT-3.5 and HuggingFace Embeddings
- **Database**: Qdrant (Vector Database)
- **APIs**:
  - OpenAI for text analysis and summarization
  - Mem0 for memory context storage
  - LMNT for Text-to-Speech
- **Image Processing**: OpenCV, PyMuPDF, Tesseract OCR

---

## 📝 **Usage**
1. **Upload File**: Upload your medical document or image for analysis.
2. **Describe Symptoms**: Optionally input symptoms for enhanced analysis.
3. **Get Summary**: View AI-generated findings in simple, patient-friendly language.
4. **Ask Questions**: Type your questions to get clear and context-aware answers.
5. **Get Treatment Suggestions**: Review professional treatment options with empathetic explanations.

---

## 🧑‍💻 **Getting Started**

### Prerequisites
1. Python 3.8 or later.
2. `pip` for managing dependencies.
3. Qdrant server for vector storage.

### Installation
1. Clone the repository:
    ```bash
    git clone https://github.com/meAmitPatil/MediScan.git
    ```
2. Navigate to the project directory:
    ```bash
    cd MediScan
    ```
3. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```
4. Create a `.env` file in the root directory:
    ```plaintext
    OPENAI_API_KEY=<your_openai_api_key>
    LMNT_API_KEY=<your_lmnt_api_key>
    MEM0_API_KEY=<your_mem0_api_key>
    ```
5. Start the Qdrant server (requires Docker):
    ```bash
    docker run -p 6333:6333 qdrant/qdrant
    ```
6. Run the Streamlit app:
    ```bash
    streamlit run app.py
    ```

7. Access the app at `http://127.0.0.1:8501`.

---


## 🤝 **Contributing**
1. Fork the repository.
2. Create a new branch:
    ```bash
    git checkout -b feature-branch-name
    ```
3. Commit your changes:
    ```bash
    git commit -m "Add your message"
    ```
4. Push the branch:
    ```bash
    git push origin feature-branch-name
    ```
5. Open a Pull Request.

---

## ⚠️ **Disclaimer**
This tool is for **educational purposes only** and is **not intended for real medical diagnosis**. Consult a certified healthcare professional for accurate medical advice.

---

🎉 **Thank you for using MediScan!**
