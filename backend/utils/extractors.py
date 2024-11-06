# utils/extractors.py
import pymupdf  # PyMuPDF for PDF files
import pytesseract  # Tesseract OCR for image files
from tika import parser as tika_parser  # Apache Tika for various file types
from PIL import Image
import os
import pydicom  # For DICOM files
import numpy as np
import cv2  # For image processing
import logging
import io
import fitz  # PyMuPDF
import docx

def analyze_medical_image(image_array):
    """Analyze medical images using computer vision techniques."""
    try:
        # Basic image analysis
        mean_intensity = np.mean(image_array)
        std_intensity = np.std(image_array)
        
        # Edge detection for identifying structures
        edges = cv2.Canny(image_array, 100, 200)
        edge_density = np.sum(edges > 0) / edges.size
        
        # Basic structure detection
        features = {
            "density": mean_intensity / 255.0,
            "contrast": std_intensity / 255.0,
            "edge_density": edge_density,
            "size": image_array.shape
        }
        
        return features
    except Exception as e:
        logging.error(f"Error in image analysis: {e}")
        return None

def detect_document_type(image, text=None):
    """Detect if the document is a medical scan/image or a text-based report."""
    try:
        # Check if it's an image-based scan
        if isinstance(image, np.ndarray):
            # Characteristics of medical scans
            is_scan = False
            
            # 1. Check grayscale nature (most medical scans are grayscale)
            if len(image.shape) == 2 or (len(image.shape) == 3 and np.allclose(image[:,:,0], image[:,:,1], image[:,:,2])):
                is_scan = True
            
            # 2. Check intensity distribution (medical scans have specific histogram patterns)
            hist = cv2.calcHist([image], [0], None, [256], [0, 256])
            peaks = np.where(hist > np.mean(hist) + 2 * np.std(hist))[0]
            if len(peaks) >= 2:  # Medical scans often have distinct peaks
                is_scan = True
            
            # 3. Check for typical medical scan text markers in OCR
            if text:
                scan_keywords = [
                    'x-ray', 'xray', 'radiograph', 'ct scan', 'mri', 'ultrasound',
                    'lateral view', 'anterior', 'posterior', 'contrast',
                    'radiology', 'imaging', 'scan', 'radiological'
                ]
                if any(keyword in text.lower() for keyword in scan_keywords):
                    is_scan = True
            
            return "medical_scan" if is_scan else "photo_of_report"
        
        return "text_report"
        
    except Exception as e:
        logging.error(f"Error in document type detection: {e}")
        return "unknown"

def extract_text_from_file(file_bytes, filename):
    """Process file directly from memory."""
    try:
        # Get file extension
        file_type = filename.lower().split('.')[-1]
        
        # Handle image files
        if file_type in ["jpg", "jpeg", "png"]:
            # Convert bytes to numpy array
            nparr = np.frombuffer(file_bytes, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if image is None:
                raise ValueError("Could not read image file")
            
            # Convert to PIL Image for OCR
            image_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            ocr_text = pytesseract.image_to_string(image_pil)
            
            return ocr_text, {"type": "image"}
        
        # Handle PDF files
        elif file_type == "pdf":
            # Read PDF from memory
            with fitz.open(stream=file_bytes, filetype="pdf") as doc:
                text = ""
                for page in doc:
                    text += page.get_text()
            return text, {"type": "pdf"}
        
        # Handle DOCX files
        elif file_type == "docx":
            # Read DOCX from memory
            doc = docx.Document(io.BytesIO(file_bytes))
            text = "\n".join([paragraph.text for paragraph in doc.paragraphs])
            return text, {"type": "docx"}
        
        else:
            return None, {"type": "unknown"}
            
    except Exception as e:
        logging.error(f"Error processing file: {e}")
        return None, {"type": "error", "error": str(e)}