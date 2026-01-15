"""Text chunking for RAG indexing."""
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field

from .config import CHUNK_SIZE, CHUNK_OVERLAP
from .parser import ParsedContent


@dataclass
class Chunk:
    """A chunk of text with metadata."""
    content: str
    chunk_type: str  # "text", "table", "image", "formula"
    source_file: str
    chunk_index: int
    metadata: Dict[str, Any] = field(default_factory=dict)
    formula_latex: Optional[str] = None  # Original LaTeX for formula chunks


def split_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    """
    Split text into overlapping chunks.
    
    Uses a simple word-based splitting to approximate token counts.
    """
    if not text.strip():
        return []
    
    words = text.split()
    chunks = []
    
    # Approximate: 1 token ≈ 0.75 words (so chunk_size tokens ≈ chunk_size * 0.75 words)
    words_per_chunk = int(chunk_size * 0.75)
    overlap_words = int(overlap * 0.75)
    
    start = 0
    while start < len(words):
        end = start + words_per_chunk
        chunk_words = words[start:end]
        chunks.append(" ".join(chunk_words))
        start = end - overlap_words
        
        if start >= len(words):
            break
    
    return chunks


def create_chunks(parsed: ParsedContent) -> List[Chunk]:
    """
    Create chunks from parsed PDF content.
    
    Args:
        parsed: ParsedContent from parser
        
    Returns:
        List of Chunk objects ready for embedding
    """
    chunks = []
    chunk_idx = 0
    
    # Chunk main text
    text_chunks = split_text(parsed.text)
    for text in text_chunks:
        chunks.append(Chunk(
            content=text,
            chunk_type="text",
            source_file=parsed.source_file,
            chunk_index=chunk_idx,
            metadata={"type": "text"}
        ))
        chunk_idx += 1
    
    # Add tables as individual chunks (preserve structure)
    for i, table in enumerate(parsed.tables):
        # Tables might be large, so chunk them too if needed
        if len(table.split()) > CHUNK_SIZE:
            table_chunks = split_text(table)
            for tc in table_chunks:
                chunks.append(Chunk(
                    content=f"[TABLE {i + 1}]\n{tc}",
                    chunk_type="table",
                    source_file=parsed.source_file,
                    chunk_index=chunk_idx,
                    metadata={"type": "table", "table_index": i + 1}
                ))
                chunk_idx += 1
        else:
            chunks.append(Chunk(
                content=f"[TABLE {i + 1}]\n{table}",
                chunk_type="table",
                source_file=parsed.source_file,
                chunk_index=chunk_idx,
                metadata={"type": "table", "table_index": i + 1}
            ))
            chunk_idx += 1
    
    # Add image descriptions with path reference
    for img in parsed.images:
        if img.get("description"):
            content = f"[IMAGE: {img['path']}]\n{img['description']}"
            chunks.append(Chunk(
                content=content,
                chunk_type="image",
                source_file=parsed.source_file,
                chunk_index=chunk_idx,
                metadata={
                    "type": "image",
                    "image_path": img["path"],
                    "image_index": img.get("index", 0)
                }
            ))
            chunk_idx += 1
    
    # NOTE: Formulas are now inlined in the text by pipeline.py before chunking
    # This means formulas appear in text chunks with their natural context
    # (e.g., "The LQI formula is: $$\Upsilon = ...$$")
    # We no longer need separate formula chunks
    
    return chunks
