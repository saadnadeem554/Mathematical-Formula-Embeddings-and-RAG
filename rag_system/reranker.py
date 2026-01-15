"""FlashRank reranking for CPU-optimized relevance scoring."""
from typing import List, Dict, Any
from flashrank import Ranker, RerankRequest
from rich.console import Console

from .config import RERANKER_MODEL, TOP_K_RERANK

console = Console()

# Global ranker instance
_ranker = None


def get_ranker() -> Ranker:
    """Get or initialize the FlashRank ranker."""
    global _ranker
    if _ranker is None:
        console.print(f"[blue]Loading reranker model:[/blue] {RERANKER_MODEL}")
        _ranker = Ranker(model_name=RERANKER_MODEL)
        console.print("[green]✓ Reranker loaded[/green]")
    return _ranker


def rerank(query: str, results: List[Dict[str, Any]], top_k: int = TOP_K_RERANK) -> List[Dict[str, Any]]:
    """
    Rerank search results using FlashRank.
    
    Args:
        query: The search query
        results: List of search results with 'content' and 'metadata' keys
        top_k: Number of top results to return after reranking
        
    Returns:
        Top-k reranked results
    """
    if not results:
        return []
    
    ranker = get_ranker()
    
    # Prepare passages for reranking
    passages = [
        {"id": i, "text": r["content"], "meta": r.get("metadata", {})}
        for i, r in enumerate(results)
    ]
    
    # Rerank
    rerank_request = RerankRequest(query=query, passages=passages)
    reranked = ranker.rerank(rerank_request)
    
    # Get top-k and restore original structure
    top_results = []
    for item in reranked[:top_k]:
        original_idx = item["id"]
        result = results[original_idx].copy()
        result["rerank_score"] = item["score"]
        top_results.append(result)
    
    return top_results
