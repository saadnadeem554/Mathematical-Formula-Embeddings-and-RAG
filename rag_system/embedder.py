"""Text embeddings using sentence-transformers."""
from typing import List
import numpy as np
from sentence_transformers import SentenceTransformer
from rich.console import Console

from .config import EMBEDDING_MODEL

console = Console()

# Global model instance (lazy loaded)
_model = None


def get_model() -> SentenceTransformer:
    """Get or initialize the embedding model."""
    global _model
    if _model is None:
        console.print(f"[blue]Loading embedding model:[/blue] {EMBEDDING_MODEL}")
        _model = SentenceTransformer(EMBEDDING_MODEL)
        console.print("[green]✓ Embedding model loaded[/green]")
    return _model


def embed_texts(texts: List[str]) -> List[List[float]]:
    """
    Generate embeddings for a list of texts.
    
    Args:
        texts: List of text strings to embed
        
    Returns:
        List of embedding vectors
    """
    model = get_model()
    embeddings = model.encode(texts, show_progress_bar=len(texts) > 10)
    return embeddings.tolist()


def embed_query(query: str) -> List[float]:
    """
    Generate embedding for a single query.
    
    Args:
        query: Query text
        
    Returns:
        Embedding vector
    """
    model = get_model()
    embedding = model.encode([query])[0]
    return embedding.tolist()
