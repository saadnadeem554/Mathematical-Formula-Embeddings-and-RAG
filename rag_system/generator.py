"""LLM generation using Groq API."""
from typing import List, Dict, Any, Generator

from groq import Groq
from rich.console import Console

from .config import GROQ_API_KEY, LLM_MODEL, SYSTEM_PROMPT

console = Console()

# Groq client
_client = None


def get_client() -> Groq:
    """Get or initialize the Groq client."""
    global _client
    if _client is None:
        if not GROQ_API_KEY:
            raise ValueError("GROQ_API_KEY not set. Please set it in .env file or environment.")
        _client = Groq(api_key=GROQ_API_KEY)
    return _client


def build_context(results: List[Dict[str, Any]]) -> str:
    """
    Build context string from search results.
    
    Args:
        results: Reranked search results
        
    Returns:
        Formatted context string
    """
    context_parts = []
    
    for i, result in enumerate(results, 1):
        content = result["content"]
        metadata = result.get("metadata", {})
        chunk_type = metadata.get("chunk_type", "text")
        
        # Add source info based on chunk type
        if chunk_type == "image":
            image_path = metadata.get("image_path", "unknown")
            context_parts.append(f"[Source {i} - Image: {image_path}]\n{content}")
        elif chunk_type == "table":
            context_parts.append(f"[Source {i} - Table]\n{content}")
        elif chunk_type == "formula":
            # Include both the description and the raw LaTeX for display
            raw_latex = metadata.get("raw_latex") or metadata.get("formula_latex", "")
            if raw_latex:
                # Format LaTeX in display math mode for rendering
                context_parts.append(
                    f"[Source {i} - Mathematical Formula]\n"
                    f"{content}\n"
                    f"Display: $${raw_latex}$$"
                )
            else:
                context_parts.append(f"[Source {i} - Formula]\n{content}")
        else:
            context_parts.append(f"[Source {i}]\n{content}")
    
    return "\n\n---\n\n".join(context_parts)


def generate_response(query: str, context: str) -> str:
    """
    Generate a response using Groq API.
    
    Args:
        query: User's question
        context: Retrieved and reranked context
        
    Returns:
        Generated response
    """
    console.print(f"[blue]Generating response with Groq ({LLM_MODEL})...[/blue]")
    
    prompt = f"""Context:
{context}

Question: {query}

Answer based on the context above:"""
    
    try:
        client = get_client()
        response = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt}
            ],
            max_tokens=1024,
            temperature=0.7
        )
        return response.choices[0].message.content
        
    except Exception as e:
        return f"Error generating response: {e}"


def stream_response(query: str, context: str) -> Generator[str, None, None]:
    """
    Stream a response using Groq API.
    
    Args:
        query: User's question
        context: Retrieved and reranked context
        
    Yields:
        Response chunks
    """
    prompt = f"""Context:
{context}

Question: {query}

Answer based on the context above:"""
    
    try:
        client = get_client()
        stream = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt}
            ],
            max_tokens=1024,
            temperature=0.7,
            stream=True
        )
        
        for chunk in stream:
            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content
                
    except Exception as e:
        yield f"Error generating response: {e}"
