"""PDF parsing with Docling."""
from pathlib import Path
from typing import List, Dict, Any
from dataclasses import dataclass, field
import shutil

from docling.document_converter import DocumentConverter
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import PdfFormatOption
from rich.console import Console

from .config import IMAGE_DIR
from .latex_normalizer import normalize_latex, create_formula_description

console = Console()


@dataclass
class ParsedContent:
    """Container for parsed PDF content."""
    text: str
    tables: List[str]  # Markdown formatted tables
    images: List[Dict[str, Any]]  # {path: str, page: int, description: str}
    formulas: List[Dict[str, Any]] = field(default_factory=list)  # {raw: str, normalized: str, description: str}
    source_file: str = ""


def parse_pdf(pdf_path: str | Path, use_marker_method: bool = False, 
               vector_formulas: List[Dict[str, Any]] = None) -> ParsedContent:
    """
    Parse a PDF file using Docling.
    
    Extracts:
    - Text content
    - Tables (converted to Markdown)
    - Image locations
    - Mathematical formulas (as LaTeX)
    
    Args:
        pdf_path: Path to the PDF file (or marked PDF if using marker method)
        use_marker_method: If True, replace formula markers in output
        vector_formulas: Pre-extracted formulas with markers (for marker method)
        
    Returns:
        ParsedContent with extracted data
    """
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")
    
    console.print(f"[blue]Parsing PDF:[/blue] {pdf_path.name}")
    
    # Configure pipeline for image and formula extraction
    pipeline_options = PdfPipelineOptions()
    pipeline_options.generate_page_images = True
    pipeline_options.generate_picture_images = True
    pipeline_options.do_formula_enrichment = True  # Enable LaTeX formula extraction
    
    converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
        }
    )
    
    # Convert document
    result = converter.convert(pdf_path)
    doc = result.document
    
    # Extract text as markdown
    text_content = doc.export_to_markdown()
    
    # Replace markers with LaTeX if using marker method
    if use_marker_method and vector_formulas:
        from .vector_formula_extractor import replace_markers_with_latex
        console.print("  [blue]Replacing formula markers with LaTeX...[/blue]")
        text_content = replace_markers_with_latex(text_content, vector_formulas)
    
    # Extract tables separately (already in markdown from export)
    tables = []
    for table in doc.tables:
        table_md = table.export_to_markdown()
        if table_md:
            tables.append(table_md)
    
    # Extract images
    images = []
    doc_image_dir = IMAGE_DIR / pdf_path.stem
    doc_image_dir.mkdir(parents=True, exist_ok=True)
    
    for idx, picture in enumerate(doc.pictures):
        # Save picture to disk
        image_filename = f"image_{idx + 1}.png"
        image_path = doc_image_dir / image_filename
        
        # Get the image if available
        if hasattr(picture, 'image') and picture.image is not None:
            picture.image.pil_image.save(str(image_path))
            images.append({
                "path": str(image_path),
                "index": idx + 1,
                "description": ""  # Will be filled by image describer
            })
            console.print(f"  [green]Extracted image:[/green] {image_filename}")
    
    # Extract mathematical formulas (these are from Docling's native extraction)
    formulas = []
    if hasattr(doc, 'equations') and doc.equations:
        for idx, equation in enumerate(doc.equations):
            raw_latex = equation.latex if hasattr(equation, 'latex') else str(equation)
            normalized = normalize_latex(raw_latex)
            description = create_formula_description(normalized)
            formulas.append({
                "raw": raw_latex,
                "normalized": normalized,
                "description": description,
                "index": idx + 1
            })
            console.print(f"  [cyan]Extracted formula {idx + 1}:[/cyan] {raw_latex[:50]}...")
    
    # Also extract inline formulas from text using regex patterns
    import re
    inline_patterns = [
        r'\$\$(.+?)\$\$',  # Display math
        r'\$(.+?)\$',      # Inline math
        r'\\\[(.+?)\\\]',  # Display math alt
        r'\\\((.+?)\\\)',  # Inline math alt
    ]
    
    def is_meaningful_formula(latex: str) -> bool:
        """Check if a formula is complex enough to be worth embedding separately."""
        cleaned = latex.strip()
        # Skip empty or very short formulas
        if len(cleaned) < 3:
            return False
        # Skip single variables (a, b, x, y, alpha, etc.)
        if re.match(r'^[a-zA-Z]$', cleaned):
            return False
        # Skip single Greek letters
        greek = ['alpha', 'beta', 'gamma', 'delta', 'epsilon', 'theta', 'lambda', 
                 'mu', 'nu', 'pi', 'sigma', 'tau', 'phi', 'psi', 'omega']
        if cleaned.lower() in greek or cleaned.replace('\\', '').lower() in greek:
            return False
        # Should have at least one operator, function, or structural element
        meaningful_patterns = [
            r'[+\-*/=<>]',  # Operators
            r'\\frac', r'\\int', r'\\sum', r'\\prod', r'\\lim',  # LaTeX structures
            r'\\sqrt', r'\\log', r'\\sin', r'\\cos', r'\\tan',  # Functions
            r'\^', r'_',  # Subscripts/superscripts
            r'\\[A-Z]',  # Capital LaTeX commands like \Upsilon
        ]
        for pattern in meaningful_patterns:
            if re.search(pattern, cleaned):
                return True
        # If it has multiple tokens/words, consider it meaningful
        if len(cleaned.split()) >= 2 or len(cleaned) > 10:
            return True
        return False
    
    for pattern in inline_patterns:
        for match in re.finditer(pattern, text_content, re.DOTALL):
            raw_latex = match.group(1).strip()
            # Skip if already extracted
            if any(f['raw'] == raw_latex for f in formulas):
                continue
            # Skip trivial formulas (single variables)
            if not is_meaningful_formula(raw_latex):
                continue
            normalized = normalize_latex(raw_latex)
            description = create_formula_description(normalized)
            formulas.append({
                "raw": raw_latex,
                "normalized": normalized,
                "description": description,
                "index": len(formulas) + 1
            })
    
    console.print(f"  [green]Text extracted ({len(text_content)} chars)[/green]")
    console.print(f"  [green]{len(tables)} tables found[/green]")
    console.print(f"  [green]{len(images)} images extracted[/green]")
    console.print(f"  [green]{len(formulas)} formulas extracted[/green]")
    if use_marker_method and vector_formulas:
        replaced = sum(1 for f in vector_formulas if f.get("latex"))
        console.print(f"  [cyan]{replaced} vector formulas replaced from markers[/cyan]")
    
    # Save markdown to file
    markdown_dir = IMAGE_DIR.parent / "markdown"
    markdown_dir.mkdir(parents=True, exist_ok=True)
    markdown_path = markdown_dir / f"{pdf_path.stem}.md"
    markdown_path.write_text(text_content, encoding="utf-8")
    console.print(f"  [green]Markdown saved to: {markdown_path}[/green]")
    
    return ParsedContent(
        text=text_content,
        tables=tables,
        images=images,
        formulas=formulas,
        source_file=str(pdf_path)
    )

