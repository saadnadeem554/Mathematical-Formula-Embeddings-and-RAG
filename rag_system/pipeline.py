"""Main RAG pipeline orchestration."""
from pathlib import Path
from typing import Dict, Any
from rich.console import Console

from .config import TOP_K_RETRIEVAL, TOP_K_RERANK, IMAGE_DIR
from .parser import parse_pdf
from .image_describer import describe_all_images
from .chunker import create_chunks
from .embedder import embed_query
from .vector_store import add_chunks, search, clear_collection, get_stats
from .reranker import rerank
from .generator import generate_response, build_context, stream_response

console = Console()


def save_image_descriptions_to_markdown(pdf_path: Path, images: list) -> None:
    """Append AI-generated image descriptions to the markdown file."""
    if not images:
        return
    
    markdown_dir = IMAGE_DIR.parent / "markdown"
    markdown_path = markdown_dir / f"{pdf_path.stem}.md"
    
    if not markdown_path.exists():
        return
    
    # Build image descriptions section
    descriptions = ["\n\n---\n\n## AI-Generated Image Descriptions\n"]
    for img in images:
        img_path = img.get("path", "unknown")
        description = img.get("description", "No description available")
        descriptions.append(f"\n### Image: {Path(img_path).name}\n")
        descriptions.append(f"**Path:** `{img_path}`\n\n")
        descriptions.append(f"{description}\n")
    
    # Append to markdown file
    with open(markdown_path, "a", encoding="utf-8") as f:
        f.writelines(descriptions)
    
    console.print(f"  [green]✓[/green] Image descriptions added to markdown")


def ingest_pdf(pdf_path: str | Path) -> Dict[str, Any]:
    """
    Ingest a PDF into the RAG system.
    
    Steps:
    1. Check for vector formulas
    2. If vectors found: create marked PDF and extract formulas
    3. Parse PDF with Docling (marked PDF if applicable)
    4. Replace markers with LaTeX
    5. Process images
    6. Chunk content
    7. Embed and store in ChromaDB
    
    Args:
        pdf_path: Path to PDF file
        
    Returns:
        Ingestion statistics
    """
    pdf_path = Path(pdf_path)
    console.print(f"\n[bold blue]=== Ingesting PDF: {pdf_path.name} ===[/bold blue]\n")
    
    # Step 1: Check for vector formulas and prepare marked PDF
    console.print("[bold]Step 1/5: Checking for Vector Formulas[/bold]")
    marked_pdf_path = str(pdf_path)
    vector_formulas = []
    
    try:
        from .vector_formula_extractor import has_vector_formulas, process_pdf_with_markers
        
        if has_vector_formulas(pdf_path):
            console.print("  [cyan]Vector formulas detected - using marker method[/cyan]")
            marked_pdf_path, vector_formulas = process_pdf_with_markers(pdf_path)
            console.print(f"  [green]Prepared {len(vector_formulas)} formulas for replacement[/green]")
        else:
            console.print("  [dim]No vector formulas detected[/dim]")
    except Exception as e:
        console.print(f"  [yellow]Warning: Vector formula processing failed: {e}[/yellow]")
    
    # Step 2: Parse PDF (marked PDF if we have vector formulas)
    console.print("\n[bold]Step 2/5: Parsing PDF[/bold]")
    parsed = parse_pdf(
        marked_pdf_path, 
        use_marker_method=bool(vector_formulas), 
        vector_formulas=vector_formulas
    )
    
    # Step 3: Process images (classify as formula or regular image)
    console.print("\n[bold]Step 3/5: Processing Images[/bold]")
    formulas = []
    
    if parsed.images:
        from .image_describer import process_images
        
        # Process images - classify as formula or regular image
        regular_images, image_formulas = process_images(parsed.images, parsed.text)
        
        # Keep only regular images (not formulas)
        parsed.images = regular_images
        
        if image_formulas:
            console.print(f"  [cyan]Found {len(image_formulas)} formulas in extracted images[/cyan]")
            formulas.extend(image_formulas)
    else:
        console.print("  No extracted images to process")
    
    # Save image descriptions for regular images
    if parsed.images:
        save_image_descriptions_to_markdown(pdf_path, parsed.images)
    
    # Step 4: Create chunks
    console.print("\n[bold]Step 4/5: Creating Chunks[/bold]")
    chunks = create_chunks(parsed)
    console.print(f"  Created {len(chunks)} chunks")
    
    # Show breakdown by type
    chunk_types = {}
    for c in chunks:
        chunk_types[c.chunk_type] = chunk_types.get(c.chunk_type, 0) + 1
    for ctype, count in chunk_types.items():
        console.print(f"    - {ctype}: {count}")
    
    # Step 5: Store in vector DB
    console.print("\n[bold]Step 5/5: Storing in Vector Database[/bold]")
    num_added = add_chunks(chunks)
    
    stats = get_stats()
    console.print(f"\n[bold green]Ingestion complete![/bold green]")
    console.print(f"  Total documents in store: {stats['total_documents']}")
    
    return {
        "file": str(pdf_path),
        "text_length": len(parsed.text),
        "tables": len(parsed.tables),
        "images": len(parsed.images),
        "vector_formulas": len(vector_formulas),
        "formulas_from_images": len(formulas),
        "chunks_created": len(chunks),
        "chunks_added": num_added,
        "total_in_store": stats["total_documents"]
    }


def query(question: str, stream: bool = False) -> str | None:
    """
    Query the RAG system.
    
    Steps:
    1. Embed query
    2. Retrieve Top-K from ChromaDB
    3. Rerank with FlashRank
    4. Generate response with Llama-3.2
    
    Args:
        question: User's question
        stream: If True, stream the response
        
    Returns:
        Generated response (or None if streaming)
    """
    console.print(f"\n[bold blue]═══ Processing Query ═══[/bold blue]\n")
    console.print(f"[italic]{question}[/italic]\n")
    
    # Step 1: Embed query
    console.print("[bold]Step 1/4: Embedding Query[/bold]")
    query_embedding = embed_query(question)
    console.print("  [green]✓[/green] Query embedded")
    
    # Step 2: Retrieve from vector store
    console.print(f"\n[bold]Step 2/4: Retrieving Top-{TOP_K_RETRIEVAL} Results[/bold]")
    results = search(query_embedding, top_k=TOP_K_RETRIEVAL)
    console.print(f"  [green]✓[/green] Retrieved {len(results)} results")
    
    if not results:
        console.print("[yellow]No relevant documents found. Try ingesting some PDFs first.[/yellow]")
        return "No relevant documents found in the knowledge base."
    
    # Show top 5 BEFORE reranking (ChromaDB similarity)
    console.print("\n[bold cyan]Before Reranking (ChromaDB Similarity - lower distance = better):[/bold cyan]")
    for i, r in enumerate(results[:5], 1):
        chunk_type = r.get("metadata", {}).get("chunk_type", "text")
        distance = r.get("distance", 0)
        preview = r["content"][:80].replace("\n", " ")
        console.print(f"  {i}. [{chunk_type}] (dist: {distance:.4f}) {preview}...")
    
    # Step 3: Rerank
    console.print(f"\n[bold]Step 3/4: Reranking to Top-{TOP_K_RERANK}[/bold]")
    top_results = rerank(question, results, top_k=TOP_K_RERANK)
    console.print(f"  [green]✓[/green] Reranked to {len(top_results)} results")
    
    # Show AFTER reranking (FlashRank scores - higher = better)
    console.print("\n[bold green]After Reranking (FlashRank Score - higher = better):[/bold green]")
    for i, r in enumerate(top_results, 1):
        chunk_type = r.get("metadata", {}).get("chunk_type", "text")
        score = r.get("rerank_score", 0)
        preview = r["content"][:80].replace("\n", " ")
        console.print(f"  {i}. [{chunk_type}] (score: {score:.4f}) {preview}...")
    
    # Step 4: Generate response
    console.print(f"\n[bold]Step 4/4: Generating Response[/bold]")
    context = build_context(top_results)
    
    if stream:
        console.print("\n[bold green]Response:[/bold green]")
        full_response = ""
        for chunk in stream_response(question, context):
            console.print(chunk, end="")
            full_response += chunk
        console.print("\n")
        return full_response
    else:
        response = generate_response(question, context)
        console.print(f"\n[bold green]Response:[/bold green]\n{response}\n")
        return response


def reset():
    """Clear all data from the vector store."""
    console.print("[yellow]Clearing vector store...[/yellow]")
    clear_collection()
    console.print("[green]✓ Vector store reset[/green]")
