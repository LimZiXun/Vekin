# Import a model for embedding text into vectors
from sentence_transformers import SentenceTransformer

# Import a text generation pipeline and models from Hugging Face Transformers
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM

# FAISS is used for fast similarity search in dense vector spaces
import faiss

# NumPy is used for numerical operations, like handling vectors
import numpy as np

# ---------------------- STEP 1: Knowledge Base ----------------------

# Define your small "document corpus" — the knowledge base
documents = [
    "Python is a high-level programming language.",
    "RAG stands for Retrieval-Augmented Generation.",
    "FAISS is a library for efficient similarity search.",
    "Transformers are powerful models for NLP tasks.",
    "Hugging Face provides open-source NLP tools.",
]

# ---------------------- STEP 2: Embedding the Documents ----------------------

# Load a pre-trained model to convert text into dense vector embeddings
try:
    embedder = SentenceTransformer('all-MiniLM-L6-v2')  # Small and fast embedding model
except Exception as e:
    print(f"Failed to load embedding model: {e}")
    exit()
# Encode (embed) all documents into vector format
doc_embeddings = embedder.encode(documents)

# ---------------------- STEP 3: Indexing with FAISS ----------------------

# Get the dimensionality of embeddings (e.g., 384 dimensions)
dimension = doc_embeddings.shape[1]

# Create a FAISS index for fast L2 (Euclidean) distance search
index = faiss.IndexFlatL2(dimension)

# Add the document vectors into the FAISS index
index.add(np.array(doc_embeddings))

# ---------------------- STEP 4: Document Retriever ----------------------

# Define a function that retrieves top-k relevant documents given a query
def retrieve(query, k=2):
    # Convert the input query to a vector
    query_vec = embedder.encode([query])
    
    # Search the FAISS index for the top-k nearest neighbors (smallest distances)----->Important
    distances, indices = index.search(np.array(query_vec), k)
    
    # Return the top-k documents using the retrieved indices
    return [documents[i] for i in indices[0]]

# ---------------------- STEP 5: Text Generator ----------------------

# Load a pre-trained text-to-text generation model (FLAN-T5 is good at Q&A)
try:
    generator = pipeline("text2text-generation", model="google/flan-t5-base")
except Exception as e:
    print(f"Failed to load generator model: {e}")
    exit()

# ---------------------- STEP 6: RAG System (Retrieve + Generate) ----------------------

# This is the main RAG pipeline
def rag(query, k=2):
    # First retrieve top-k context documents
    context_docs = retrieve(query, k)
    
    # Combine them into a single string (you can also use separators if needed)
    context = " ".join(context_docs)
    
    # Format the prompt to feed into the generator: context + user question
    prompt = f"Context: {context} Question: {query}"
    
    # Generate an answer based on the prompt
    result = generator(prompt, max_new_tokens=100)[0]['generated_text']
    
    # Return the generated answer
    return result

# ---------------------- STEP 7: Test the System ----------------------

# Define a question to test your RAG system
query = "What is rag in NLP?"

# Call the RAG function with the query
answer = rag(query)

# Print the question and the system's answer
print("Question:", query)
print("Answer:", answer)
