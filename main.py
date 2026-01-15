"""Terminal CLI for the CPU-Based Multimodal RAG System."""
import typer
from pathlib import Path
from rich.console import Console
from rich.panel import Panel

from rag_system.pipeline import ingest_pdf, query, reset
from rag_system.vector_store import get_stats

app = typer.Typer(
    name="rag",
    help="CPU-Based Multimodal RAG System",
    add_completion=False
)
console = Console()


@app.command()
def ingest(pdf_path: str = typer.Argument(..., help="Path to PDF file to ingest")):
    """Ingest a PDF document into the RAG system."""
    path = Path(pdf_path)
    if not path.exists():
        console.print(f"[red]Error: File not found: {pdf_path}[/red]")
        raise typer.Exit(1)
    
    if not path.suffix.lower() == ".pdf":
        console.print(f"[red]Error: File must be a PDF: {pdf_path}[/red]")
        raise typer.Exit(1)
    
    try:
        stats = ingest_pdf(path)
        console.print(Panel(
            f"[green]Successfully ingested:[/green] {path.name}\n"
            f"Text: {stats['text_length']} chars | "
            f"Tables: {stats['tables']} | "
            f"Images: {stats['images']} | "
            f"Chunks: {stats['chunks_created']}",
            title="Ingestion Complete"
        ))
    except Exception as e:
        console.print(f"[red]Error during ingestion: {e}[/red]")
        raise typer.Exit(1)


@app.command()
def ask(
    question: str = typer.Argument(..., help="Question to ask"),
    stream: bool = typer.Option(False, "--stream", "-s", help="Stream the response")
):
    """Ask a question about the ingested documents."""
    try:
        response = query(question, stream=stream)
    except Exception as e:
        console.print(f"[red]Error during query: {e}[/red]")
        raise typer.Exit(1)


@app.command()
def clear():
    """Clear all documents from the vector store."""
    confirm = typer.confirm("Are you sure you want to clear all documents?")
    if confirm:
        reset()
        console.print("[green]Vector store cleared.[/green]")
    else:
        console.print("[yellow]Operation cancelled.[/yellow]")


@app.command()
def stats():
    """Show statistics about the vector store."""
    info = get_stats()
    console.print(Panel(
        f"[bold]Collection:[/bold] {info['collection_name']}\n"
        f"[bold]Total Documents:[/bold] {info['total_documents']}",
        title="Vector Store Statistics"
    ))


@app.command()
def chat():
    """Start an interactive chat session."""
    console.print(Panel(
        "[bold]CPU-Based Multimodal RAG System[/bold]\n\n"
        "Commands:\n"
        "  [cyan]/ingest <path>[/cyan] - Ingest a PDF\n"
        "  [cyan]/stats[/cyan]         - Show statistics\n"
        "  [cyan]/clear[/cyan]         - Clear vector store\n"
        "  [cyan]/quit[/cyan]          - Exit\n\n"
        "Or just type your question!",
        title="Interactive Chat"
    ))
    
    while True:
        try:
            user_input = console.input("\n[bold cyan]You:[/bold cyan] ").strip()
            
            if not user_input:
                continue
            
            if user_input.lower() in ["/quit", "/exit", "/q"]:
                console.print("[yellow]Goodbye![/yellow]")
                break
            
            if user_input.lower() == "/stats":
                info = get_stats()
                console.print(f"Documents in store: {info['total_documents']}")
                continue
            
            if user_input.lower() == "/clear":
                reset()
                continue
            
            if user_input.lower().startswith("/ingest "):
                pdf_path = user_input[8:].strip()
                if pdf_path:
                    try:
                        ingest_pdf(pdf_path)
                    except Exception as e:
                        console.print(f"[red]Error: {e}[/red]")
                continue
            
            # Regular question
            query(user_input, stream=True)
            
        except KeyboardInterrupt:
            console.print("\n[yellow]Use /quit to exit[/yellow]")
        except EOFError:
            break


if __name__ == "__main__":
    app()
