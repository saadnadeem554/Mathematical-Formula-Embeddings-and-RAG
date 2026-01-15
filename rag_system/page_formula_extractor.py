"""Extract formulas from PDF page images using vision model."""
import fitz  # PyMuPDF
from pathlib import Path
from typing import List, Dict, Any, Tuple
from rich.console import Console

from .config import IMAGE_DIR
from .image_describer import get_client, encode_image_base64
from .latex_normalizer import normalize_latex

console = Console()

# Prompt for finding formulas on a page
PAGE_FORMULA_PROMPT = """Analyze this PDF page image carefully.

Find ALL mathematical formulas, equations, or expressions on this page.
For each formula found:
1. Extract the complete LaTeX representation
2. Note approximately where it appears (e.g., "after paragraph about heat loss", "in the middle of the page")

Respond in this format for EACH formula found:
FORMULA_START
LOCATION: [brief description of where on page]
LATEX: [the LaTeX code]
FORMULA_END

If there are NO mathematical formulas on this page, respond with:
NO_FORMULAS

Be precise with the LaTeX - include all symbols, subscripts, superscripts, fractions, Greek letters, integrals, etc."""


def extract_page_images(pdf_path: Path, dpi: int = 150) -> List[Dict[str, Any]]:
    """
    Extract images of each page from a PDF.
    
    Args:
        pdf_path: Path to the PDF file
        dpi: Resolution for rendering pages (higher = better quality but larger)
        
    Returns:
        List of dicts with 'page_num', 'image_path'
    """
    pdf_path = Path(pdf_path)
    doc = fitz.open(pdf_path)
    
    # Create directory for page images
    pages_dir = IMAGE_DIR / pdf_path.stem / "pages"
    pages_dir.mkdir(parents=True, exist_ok=True)
    
    page_images = []
    
    console.print(f"  [blue]Extracting {len(doc)} page images...[/blue]")
    
    for page_num in range(len(doc)):
        page = doc[page_num]
        
        # Render page to image
        mat = fitz.Matrix(dpi / 72, dpi / 72)  # Scale factor
        pix = page.get_pixmap(matrix=mat)
        
        # Save as PNG
        image_path = pages_dir / f"page_{page_num + 1}.png"
        pix.save(str(image_path))
        
        page_images.append({
            "page_num": page_num + 1,
            "image_path": str(image_path)
        })
    
    doc.close()
    console.print(f"  [green]✓ Extracted {len(page_images)} page images[/green]")
    
    return page_images


def extract_formulas_from_page(image_path: str, page_num: int) -> List[Dict[str, Any]]:
    """
    Use vision model to find and extract formulas from a page image.
    
    Args:
        image_path: Path to the page image
        page_num: Page number for reference
        
    Returns:
        List of formulas found: [{latex, location, page_num}]
    """
    from .config import VISION_MODEL
    
    try:
        client = get_client()
        
        # Encode image
        image_data = encode_image_base64(image_path)
        
        response = client.chat.completions.create(
            model=VISION_MODEL,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": PAGE_FORMULA_PROMPT},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{image_data}"
                            }
                        }
                    ]
                }
            ],
            max_tokens=2000
        )
        
        result = response.choices[0].message.content
        
        # Parse the response
        if "NO_FORMULAS" in result.upper():
            return []
        
        formulas = []
        
        # Parse FORMULA_START blocks
        blocks = result.split("FORMULA_START")
        for block in blocks[1:]:  # Skip first empty split
            if "FORMULA_END" in block:
                content = block.split("FORMULA_END")[0].strip()
                
                location = ""
                latex = ""
                
                for line in content.split("\n"):
                    line = line.strip()
                    if line.upper().startswith("LOCATION:"):
                        location = line[9:].strip()
                    elif line.upper().startswith("LATEX:"):
                        latex = line[6:].strip()
                
                if latex:
                    # Clean up common artifacts
                    latex = latex.replace("```latex", "").replace("```", "").strip()
                    latex = latex.strip("$")
                    
                    formulas.append({
                        "latex": latex,
                        "normalized": normalize_latex(latex),
                        "location": location,
                        "page_num": page_num
                    })
        
        return formulas
        
    except Exception as e:
        console.print(f"[yellow]Warning: Error processing page {page_num}: {e}[/yellow]")
        return []


def extract_all_page_formulas(pdf_path: Path, max_pages: int = None) -> List[Dict[str, Any]]:
    """
    Extract all formulas from a PDF by analyzing page images.
    
    Args:
        pdf_path: Path to the PDF file
        max_pages: Optional limit on number of pages to process
        
    Returns:
        List of all formulas found across all pages
    """
    pdf_path = Path(pdf_path)
    
    console.print(f"\n[bold cyan]Extracting formulas from page images...[/bold cyan]")
    
    # Extract page images
    page_images = extract_page_images(pdf_path)
    
    if max_pages:
        page_images = page_images[:max_pages]
    
    all_formulas = []
    
    console.print(f"  [blue]Analyzing pages for formulas...[/blue]")
    
    for page_info in page_images:
        page_num = page_info["page_num"]
        image_path = page_info["image_path"]
        
        console.print(f"    Processing page {page_num}...")
        
        formulas = extract_formulas_from_page(image_path, page_num)
        
        if formulas:
            console.print(f"      [cyan]✓ Found {len(formulas)} formulas[/cyan]")
            all_formulas.extend(formulas)
        else:
            console.print(f"      No formulas found")
    
    console.print(f"\n  [green]✓ Total formulas extracted: {len(all_formulas)}[/green]")
    
    return all_formulas


def insert_formulas_into_markdown(markdown_text: str, formulas: List[Dict[str, Any]]) -> str:
    """
    Insert extracted formulas into markdown text based on context matching.
    
    This attempts to find formula labels like "(Formule 2.3)" and replace with actual LaTeX.
    
    Args:
        markdown_text: Original markdown text
        formulas: List of extracted formulas
        
    Returns:
        Updated markdown with actual formulas
    """
    import re
    
    result = markdown_text
    
    # Find formula placeholders like "$$(F o r m u l e 2.3)$$" or "$$(Formule 2.3)$$"
    placeholder_pattern = r'\$\$\s*\(\s*[Ff]\s*o?\s*r?\s*m?\s*u?\s*l?\s*e?\s*[\s\\]*(\d+)\s*\.?\s*(\d+)\s*\)\s*\$\$'
    
    # Build a mapping of formula numbers to actual LaTeX
    # This is a heuristic - we match based on order of appearance
    formula_idx = 0
    
    def replace_placeholder(match):
        nonlocal formula_idx
        if formula_idx < len(formulas):
            formula = formulas[formula_idx]
            formula_idx += 1
            latex = formula["latex"]
            return f"\n$$\n{latex}\n$$\n"
        return match.group(0)
    
    result = re.sub(placeholder_pattern, replace_placeholder, result)
    
    return result
