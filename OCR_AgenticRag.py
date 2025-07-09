import os
import re
import logging
import numpy as np
from PIL import Image
from sentence_transformers import SentenceTransformer
from google.cloud import vision
from google.oauth2 import service_account
import faiss
import asyncio
import platform

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Regex patterns for redaction
SENSITIVE_PATTERNS = {
    'name': r'\b[A-Z][a-z]+ [A-Z][a-z]+\b',
    'address': r'\d{1,5} [\w\s]+, [\w\s]+, [A-Z]{2} \d{5}',
    'phone': r'\b\d{3}-\d{3}-\d{4}\b|\b\d{10}\b',
    'email': r'\b[\w\.-]+@[\w\.-]+\.\w+\b',
    'bank_account': r'\b\d{8,12}\b',
    'amount': r'\$\d+\.\d{2}\b'
}


class InvoiceRAG:
    def __init__(self, credential_path: str):
        """Initialize RAG with Google Vision and SentenceTransformer."""
        try:
            credentials = service_account.Credentials.from_service_account_file(credential_path)
            self.vision_client = vision.ImageAnnotatorClient(credentials=credentials)
            self.model = SentenceTransformer('all-MiniLM-L6-v2')
            self.index = None
            self.text_chunks = []
            self.redacted_text = []
            logger.info("InvoiceRAG initialized with Google Cloud Vision OCR.")
        except Exception as e:
            logger.error(f"Initialization failed: {e}")
            raise

    def extract_text_from_image(self, image_path: str) -> str:
        """Use Google Cloud Vision API to extract text from image."""
        try:
            if not os.path.exists(image_path):
                raise FileNotFoundError(f"File not found: {image_path}")

            with open(image_path, "rb") as img_file:
                content = img_file.read()

            image = vision.Image(content=content)
            response = self.vision_client.text_detection(image=image)

            if response.error.message:
                raise Exception(f"Vision API error: {response.error.message}")

            text = response.full_text_annotation.text
            logger.info("Text successfully extracted from image.")
            return text
        except Exception as e:
            logger.error(f"Text extraction failed: {e}")
            return ""

    def redact_sensitive_info(self, text: str) -> str:
        """Redact known sensitive data using regex patterns."""
        try:
            redacted = text
            for key, pattern in SENSITIVE_PATTERNS.items():
                redacted = re.sub(pattern, f"[REDACTED_{key.upper()}]", redacted, flags=re.IGNORECASE)
            logger.info("Sensitive information redacted.")
            return redacted
        except Exception as e:
            logger.error(f"Redaction error: {e}")
            return text

    def chunk_text(self, text: str, chunk_size: int = 100) -> list:
        """Split text into manageable chunks for vector embedding."""
        try:
            words = text.split()
            return [' '.join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]
        except Exception as e:
            logger.error(f"Chunking error: {e}")
            return []

    def build_vector_store(self, chunks: list):
        """Create a FAISS index from text chunks."""
        try:
            embeddings = self.model.encode(chunks)
            self.index = faiss.IndexFlatL2(embeddings.shape[1])
            self.index.add(embeddings)
            self.text_chunks = chunks
            logger.info("FAISS vector store built.")
        except Exception as e:
            logger.error(f"Vector store build failed: {e}")
            raise

    async def process_invoice(self, image_path: str) -> bool:
        """Full pipeline: OCR -> redact -> chunk -> embed."""
        try:
            text = self.extract_text_from_image(image_path)
            if not text.strip():
                return False

            self.redacted_text = self.redact_sensitive_info(text).splitlines()
            chunks = self.chunk_text(text)
            if not chunks:
                return False

            self.build_vector_store(chunks)
            return True
        except Exception as e:
            logger.error(f"Invoice processing failed: {e}")
            return False

    async def query(self, query_text: str, top_k: int = 3) -> str:
        """Search the vector store and return top relevant chunks."""
        try:
            if not query_text or self.index is None:
                return "Error: Invalid query or no invoice processed."

            query_vector = self.model.encode([query_text])
            distances, indices = self.index.search(np.array(query_vector), top_k)
            results = [self.text_chunks[i] for i in indices[0] if i < len(self.text_chunks)]

            if not results:
                return "No relevant information found."

            # Custom logic for specific queries
            if "total" in query_text.lower():
                for chunk in results:
                    match = re.search(r'\$\d+\.\d{2}', chunk)
                    if match:
                        return f"Invoice total: {match.group()}"
                return "Total amount not found."

            if "issued to" in query_text.lower():
                return "Recipient information has been redacted."

            return "Relevant information:\n" + "\n".join(results)
        except Exception as e:
            logger.error(f"Query error: {e}")
            return f"Query failed: {str(e)}"
async def main():
    # Set your credential path and image
    credential_path = "ocr-test-460505-f7af38004fea.json"
    image_path = "invoice.jpg"

    rag = InvoiceRAG(credential_path)

    if not await rag.process_invoice(image_path):
        print("Failed to process invoice.")
        return

    queries = [
        "What is the invoice total?",
        "Who is the invoice issued to?",
        "What items are listed?",
        "When is the due date?",
        "",
    ]

    for q in queries:
        response = await rag.query(q)
        print(f"Query: {q}\nResponse: {response}\n")

if __name__ == "__main__":
    asyncio.run(main())