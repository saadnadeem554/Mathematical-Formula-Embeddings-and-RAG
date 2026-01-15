"""Configuration constants for the RAG system."""
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Paths
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
PDF_DIR = DATA_DIR / "pdfs"
CHROMA_DIR = DATA_DIR / "chroma_db"
IMAGE_DIR = DATA_DIR / "images"

# Create directories
for dir_path in [DATA_DIR, PDF_DIR, CHROMA_DIR, IMAGE_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# Groq API Configuration
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")

# Model Configuration
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
RERANKER_MODEL = "ms-marco-MiniLM-L-12-v2"

# Groq Models
VISION_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"  # For image analysis (Llama 4 Scout)
LLM_MODEL = "llama-3.3-70b-versatile"  # For text generation (fast on Groq)

# Chunking Configuration
CHUNK_SIZE = 500  # tokens
CHUNK_OVERLAP = 50  # tokens

# Retrieval Configuration
TOP_K_RETRIEVAL = 15
TOP_K_RERANK = 5

# ChromaDB Collection
COLLECTION_NAME = "rag_documents"

# System Prompt for LLM
SYSTEM_PROMPT = """You are a helpful assistant that answers questions based on the provided context.
Use ONLY the information from the context to answer. If the answer cannot be found in the context, say so.

IMPORTANT for mathematical formulas:
- When a formula appears in the context (marked with $$...$$), treat it as THE mathematical equation for the concept being discussed.
- Connect formulas to the variables and concepts described in text sources.
- Display formulas using LaTeX notation when presenting equations.

When referencing images, output the image path so the user can view it.
Be concise and accurate in your responses."""

# Image Analysis Prompt
IMAGE_ANALYSIS_PROMPT = """Analyze this image from a PDF document. 
If it's a chart or graph, describe the data trends and key values.
If it's a diagram, explain the components and their relationships.
If it contains text, extract the key information.
Provide a detailed 3-sentence summary."""
