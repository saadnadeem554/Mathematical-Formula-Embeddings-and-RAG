"""ChromaDB vector store operations."""
from typing import List, Dict, Any, Optional
import chromadb
from chromadb.config import Settings
from rich.console import Console

from .config import CHROMA_DIR, COLLECTION_NAME
from .chunker import Chunk
from .embedder import embed_texts

console = Console()

# Global client instance
_client = None
_collection = None


def get_collection():
    """Get or initialize the ChromaDB collection."""
    global _client, _collection
    
    if _collection is None:
        console.print(f"[blue]Initializing ChromaDB at:[/blue] {CHROMA_DIR}")
        _client = chromadb.PersistentClient(
            path=str(CHROMA_DIR),
            settings=Settings(anonymized_telemetry=False)
        )
        _collection = _client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"}
        )
        console.print(f"[green]✓ Collection ready:[/green] {_collection.count()} documents")
    
    return _collection


def add_chunks(chunks: List[Chunk]) -> int:
    """
    Add chunks to the vector store.
    
    Args:
        chunks: List of Chunk objects to add
        
    Returns:
        Number of chunks added
    """
    if not chunks:
        return 0
    
    collection = get_collection()
    
    # Prepare data
    texts = [c.content for c in chunks]
    ids = [f"{c.source_file}_{c.chunk_index}" for c in chunks]
    metadatas = []
    for c in chunks:
        meta = {
            "chunk_type": c.chunk_type,
            "source_file": c.source_file,
            "chunk_index": c.chunk_index,
            **c.metadata
        }
        # Add formula_latex for formula chunks (for rendering during retrieval)
        if c.formula_latex:
            meta["formula_latex"] = c.formula_latex
        metadatas.append(meta)
    
    # Generate embeddings
    console.print(f"[blue]Generating embeddings for {len(texts)} chunks...[/blue]")
    embeddings = embed_texts(texts)
    
    # Add to collection
    collection.add(
        documents=texts,
        embeddings=embeddings,
        metadatas=metadatas,
        ids=ids
    )
    
    console.print(f"[green]✓ Added {len(chunks)} chunks to vector store[/green]")
    return len(chunks)


def search(query_embedding: List[float], top_k: int = 20) -> List[Dict[str, Any]]:
    """
    Search for similar documents.
    
    Args:
        query_embedding: Query embedding vector
        top_k: Number of results to return
        
    Returns:
        List of results with document, metadata, and distance
    """
    collection = get_collection()
    
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=["documents", "metadatas", "distances"]
    )
    
    # Format results
    formatted = []
    for i in range(len(results["documents"][0])):
        formatted.append({
            "content": results["documents"][0][i],
            "metadata": results["metadatas"][0][i],
            "distance": results["distances"][0][i]
        })
    
    return formatted


def clear_collection():
    """Delete all documents from the collection."""
    global _collection
    
    if _client is not None:
        _client.delete_collection(COLLECTION_NAME)
        _collection = None
        console.print("[yellow]Collection cleared[/yellow]")
    
    # Recreate empty collection
    get_collection()


def get_stats() -> Dict[str, Any]:
    """Get collection statistics."""
    collection = get_collection()
    return {
        "total_documents": collection.count(),
        "collection_name": COLLECTION_NAME
    }
