"""
Configuration file for FYP Handbook RAG System
Contains all hyperparameters as per assignment requirements
"""

import os

# ============ Paths ============
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
EMBEDDINGS_DIR = os.path.join(DATA_DIR, "embeddings")
LOGS_DIR = os.path.join(BASE_DIR, "logs")
SRC_DIR = os.path.join(BASE_DIR, "src")

# Input/Output Files
HANDBOOK_PDF = os.path.join(DATA_DIR, "handbook.pdf")
CHUNKS_JSON = os.path.join(DATA_DIR, "chunks.json")
FAISS_INDEX = os.path.join(EMBEDDINGS_DIR, "faiss.index")
META_PKL = os.path.join(EMBEDDINGS_DIR, "meta.pkl")
MODEL_NAME_FILE = os.path.join(EMBEDDINGS_DIR, "model_name.txt")
PROMPT_LOG = os.path.join(LOGS_DIR, "prompt_log.txt")
PROMPT_TEMPLATE_FILE = os.path.join(SRC_DIR, "prompt_template.txt")

# ============ Chunking Parameters (As per Assignment) ============
CHUNK_SIZE_WORDS = 350  # Target: 250-400 words
OVERLAP_PERCENTAGE = 0.30  # Target: 20-40% overlap
MIN_CHUNK_SIZE = 50  # Minimum words per chunk

# ============ Embedding Model (As per Assignment) ============
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIMENSION = 384  # Dimension of all-MiniLM-L6-v2

# ============ Retrieval Parameters (As per Assignment) ============
TOP_K = 5  # Number of chunks to retrieve
SIMILARITY_THRESHOLD = 0.25  # Minimum cosine similarity score

# ============ Create Directories ============
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(EMBEDDINGS_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)