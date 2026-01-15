"""
Vector formula detection and extraction using PyMuPDF (fitz).

This module uses the "Marker" method:
1. Pre-process PDF with fitz to insert unique markers at formula locations
2. Run Docling on the marked PDF - markers appear in markdown
3. Replace markers with LaTeX extracted from formula images
"""
import os
import fitz  # PyMuPDF
import tempfile
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from rich.console import Console

from .config import IMAGE_DIR
from .image_describer import get_client, encode_image_base64
from .latex_normalizer import normalize_latex

console = Console()

# Marker format for formulas
FORMULA_MARKER_PREFIX = "##FORMULA_"
FORMULA_MARKER_SUFFIX = "##"

# Prompt for extracting LaTeX from a formula image
VECTOR_FORMULA_PROMPT = """This image contains a mathematical formula or equation from a technical document.

Extract the COMPLETE LaTeX representation of this formula.
Be precise - include ALL:
- Fractions (\\frac{}{})
- Subscripts and superscripts
- Greek letters (\\alpha, \\beta, etc.)
- Special symbols (\\cdot, \\times, \\sum, \\int, etc.)
- Parentheses and brackets

Respond with ONLY the LaTeX code, nothing else. Do not wrap in $$ or code blocks."""


def has_vector_formulas(pdf_path: str | Path, sample_pages: int = 3) -> bool:
    """
    Quickly detect if a PDF contains vector-based formulas.
    
    Checks the first few pages for vector drawing clusters that look like formulas,
    filtering out tables, headers, and footers.
    
    Args:
        pdf_path: Path to the PDF file
        sample_pages: Number of pages to sample (default: 3)
        
    Returns:
        True if vector formulas are likely present
    """
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        return False
    
    try:
        doc = fitz.open(pdf_path)
        pages_to_check = min(sample_pages, len(doc))
        
        formula_candidates = 0
        
        for page_num in range(pages_to_check):
            page = doc[page_num]
            
            # Find tables to exclude
            tabs = page.find_tables()
            table_bboxes = [tab.bbox for tab in tabs.tables]
            
            # Find vector drawing clusters
            clusters = page.cluster_drawings(x_tolerance=30, y_tolerance=4)
            
            for rect in clusters:
                # Skip if overlaps with table
                overlap_area = 0
                for table_bbox in table_bboxes:
                    intersect = rect & table_bbox
                    if not intersect.is_empty:
                        overlap_area += intersect.get_area()
                
                if overlap_area > (rect.get_area() * 0.5):
                    continue
                
                # Skip header/footer regions
                if rect.y1 < 70 or rect.y0 > (page.rect.height - 70):
                    continue
                
                # Check if it's large enough to be a formula
                width = rect.x1 - rect.x0
                height = rect.y1 - rect.y0
                
                # Formulas are typically wider than tall and reasonably sized
                if width > 30 and height > 10 and width < page.rect.width * 0.9:
                    formula_candidates += 1
        
        doc.close()
        
        return formula_candidates > 0
        
    except Exception as e:
        console.print(f"[yellow]Warning: Could not check for vector formulas: {e}[/yellow]")
        return False


def create_marked_pdf_and_extract_formulas(pdf_path: str | Path) -> Tuple[str, List[Dict[str, Any]]]:
    """
    Create a copy of the PDF with text markers inserted at formula locations,
    and extract formula images.
    
    For each formula found:
    1. Insert a text marker like "##FORMULA_001##" at the formula location
    2. Extract the formula as a PNG image
    
    Args:
        pdf_path: Path to the original PDF file
        
    Returns:
        Tuple of (path_to_marked_pdf, list_of_formula_info)
        formula_info contains: marker, image_path, page_num
    """
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")
    
    # Create output directory for formula images
    formulas_dir = IMAGE_DIR / pdf_path.stem / "vector_formulas"
    formulas_dir.mkdir(parents=True, exist_ok=True)
    
    # Open PDF for modification
    doc = fitz.open(pdf_path)
    
    formulas = []
    formula_idx = 0
    
    console.print(f"  [blue]Scanning {len(doc)} pages for vector formulas...[/blue]")
    
    for page_num in range(len(doc)):
        page = doc[page_num]
        
        # Find tables to exclude
        tabs = page.find_tables()
        table_bboxes = [tab.bbox for tab in tabs.tables]
        
        # Find vector drawing clusters
        clusters = page.cluster_drawings(x_tolerance=30, y_tolerance=4)
        
        for rect in clusters:
            # Skip if overlaps with table (>50%)
            overlap_area = 0
            for table_bbox in table_bboxes:
                intersect = rect & table_bbox
                if not intersect.is_empty:
                    overlap_area += intersect.get_area()
            
            if overlap_area > (rect.get_area() * 0.5):
                continue
            
            # Skip header/footer regions
            if rect.y1 < 70 or rect.y0 > (page.rect.height - 70):
                continue
            
            # Skip very small items
            width = rect.x1 - rect.x0
            height = rect.y1 - rect.y0
            if width < 30 or height < 10:
                continue
            
            # Extract formula as image BEFORE adding marker
            mat = fitz.Matrix(4.0, 4.0)  # 4x scale for better quality
            clip_rect = rect + (-3, -3, 3, 3)  # Small padding
            pix = page.get_pixmap(matrix=mat, clip=clip_rect)
            
            # Skip near-blank images
            if pix.color_count() < 3:
                continue
            
            # Create unique marker
            marker = f"{FORMULA_MARKER_PREFIX}{formula_idx:03d}{FORMULA_MARKER_SUFFIX}"
            
            # Save formula image
            image_filename = f"formula_{formula_idx:03d}.png"
            image_path = formulas_dir / image_filename
            pix.save(str(image_path))
            
            # Insert marker text at the formula location
            # Position it at the top-left of the formula rectangle
            # Use a very small font so it doesn't disturb layout much
            text_point = fitz.Point(rect.x0, rect.y0 + 10)  # Slightly below top
            
            # Insert the marker as text
            # Use a small font and place it at the start of the formula area
            page.insert_text(
                text_point,
                marker,
                fontsize=8,
                color=(1, 1, 1),  # White text (invisible on white background)
                overlay=True
            )
            
            formulas.append({
                "marker": marker,
                "image_path": str(image_path),
                "page_num": page_num + 1,
                "rect": (rect.x0, rect.y0, rect.x1, rect.y1),
                "index": formula_idx
            })
            
            formula_idx += 1
    
    console.print(f"  [green]Found {len(formulas)} vector formulas[/green]")
    
    if not formulas:
        doc.close()
        return str(pdf_path), []
    
    # Save marked PDF to temp file
    temp_dir = Path(tempfile.gettempdir())
    marked_pdf_path = temp_dir / f"{pdf_path.stem}_marked.pdf"
    doc.save(str(marked_pdf_path))
    doc.close()
    
    console.print(f"  [green]Created marked PDF: {marked_pdf_path.name}[/green]")
    
    return str(marked_pdf_path), formulas


def convert_formula_image_to_latex(image_path: str | Path) -> Optional[str]:
    """
    Send a formula image to the vision model and extract LaTeX.
    
    Args:
        image_path: Path to the formula PNG image
        
    Returns:
        LaTeX string, or None if extraction failed
    """
    from .config import VISION_MODEL
    
    image_path = Path(image_path)
    if not image_path.exists():
        return None
    
    try:
        client = get_client()
        
        image_data = encode_image_base64(image_path)
        
        response = client.chat.completions.create(
            model=VISION_MODEL,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": VECTOR_FORMULA_PROMPT},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{image_data}"
                            }
                        }
                    ]
                }
            ],
            max_tokens=500
        )
        
        latex = response.choices[0].message.content.strip()
        
        # Clean up common artifacts
        latex = latex.replace("```latex", "").replace("```", "").strip()
        latex = latex.strip("$")
        
        return latex if latex else None
        
    except Exception as e:
        console.print(f"[yellow]Warning: Could not convert formula image: {e}[/yellow]")
        return None


def convert_all_formulas_to_latex(formulas: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Convert all formula images to LaTeX.
    
    Args:
        formulas: List of formula dicts with 'image_path' key
        
    Returns:
        Updated list with 'latex' key added
    """
    console.print(f"  [blue]Converting {len(formulas)} formulas to LaTeX...[/blue]")
    
    for i, formula_info in enumerate(formulas):
        image_path = formula_info["image_path"]
        console.print(f"    Processing formula {i + 1}/{len(formulas)}...")
        
        latex = convert_formula_image_to_latex(image_path)
        
        if latex:
            formula_info["latex"] = latex
            formula_info["normalized"] = normalize_latex(latex)
            console.print(f"      [cyan]OK[/cyan] {latex[:50]}...")
        else:
            formula_info["latex"] = None
            console.print(f"      [yellow]Failed to extract LaTeX[/yellow]")
    
    success_count = sum(1 for f in formulas if f.get("latex"))
    console.print(f"\n  [green]Successfully converted {success_count}/{len(formulas)} formulas[/green]")
    
    return formulas


def replace_markers_with_latex(markdown_text: str, formulas: List[Dict[str, Any]]) -> str:
    """
    Replace formula markers in markdown with actual LaTeX.
    
    Args:
        markdown_text: Markdown text containing markers like ##FORMULA_001##
        formulas: List of formula dicts with 'marker' and 'latex' keys
        
    Returns:
        Markdown with markers replaced by LaTeX
    """
    result = markdown_text
    replaced_count = 0
    
    for formula_info in formulas:
        marker = formula_info.get("marker", "")
        latex = formula_info.get("latex")
        
        if marker and latex and marker in result:
            # Replace marker with display math
            replacement = f"\n\n$${latex}$$\n\n"
            result = result.replace(marker, replacement)
            replaced_count += 1
            console.print(f"    [cyan]Replaced[/cyan] {marker} -> {latex[:40]}...")
    
    console.print(f"  [green]Replaced {replaced_count} markers with LaTeX[/green]")
    
    return result


def process_pdf_with_markers(pdf_path: str | Path) -> Tuple[str, List[Dict[str, Any]]]:
    """
    Main entry point: Process PDF with marker method.
    
    This function:
    1. Creates a marked PDF with formula markers
    2. Extracts formula images
    3. Converts images to LaTeX
    
    The caller should:
    1. Run Docling on the marked PDF
    2. Call replace_markers_with_latex() on the markdown output
    
    Args:
        pdf_path: Path to the original PDF file
        
    Returns:
        Tuple of (path_to_marked_pdf, formulas_with_latex)
    """
    pdf_path = Path(pdf_path)
    
    console.print(f"\n[bold cyan]Processing PDF with marker method: {pdf_path.name}[/bold cyan]")
    
    # Step 1: Create marked PDF and extract formula images
    marked_pdf_path, formulas = create_marked_pdf_and_extract_formulas(pdf_path)
    
    if not formulas:
        console.print("  [yellow]No vector formulas found[/yellow]")
        return str(pdf_path), []
    
    # Step 2: Convert formula images to LaTeX
    formulas = convert_all_formulas_to_latex(formulas)
    
    return marked_pdf_path, formulas


# Keep old functions for backwards compatibility
def extract_vector_formulas(pdf_path: str | Path) -> List[Dict[str, Any]]:
    """DEPRECATED: Use process_pdf_with_markers instead."""
    _, formulas = create_marked_pdf_and_extract_formulas(pdf_path)
    return formulas


def extract_and_convert_all_formulas(pdf_path: str | Path) -> List[Dict[str, Any]]:
    """DEPRECATED: Use process_pdf_with_markers instead."""
    _, formulas = process_pdf_with_markers(pdf_path)
    return formulas
