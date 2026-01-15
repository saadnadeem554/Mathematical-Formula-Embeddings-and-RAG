"""Image description using Groq's Llama 3.2 Vision."""
import base64
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional

from groq import Groq
from rich.console import Console

from .config import GROQ_API_KEY, VISION_MODEL, IMAGE_ANALYSIS_PROMPT
from .latex_normalizer import normalize_latex, create_formula_description

console = Console()

# Groq client
_client = None

# Prompt for extracting LaTeX from formula images
FORMULA_EXTRACTION_PROMPT = """Analyze this image carefully. 

If this image contains a mathematical formula, equation, or expression:
1. First, respond with "FORMULA: YES"
2. Then on the next line, provide the LaTeX representation of the formula/equation

If this image does NOT contain a mathematical formula (e.g., it's a diagram, photo, chart, etc.):
1. Respond with "FORMULA: NO"
2. Then provide a brief description of what the image shows

Be precise with the LaTeX - include all symbols, subscripts, superscripts, fractions, etc. exactly as shown."""

# Prompt for enriching formula with document context
FORMULA_CONTEXT_PROMPT = """You are given:
1. A mathematical formula in LaTeX: {latex}
2. Surrounding text from a document: {context}

Based on this information, create a rich semantic description that:
1. Identifies what this formula represents (e.g., "This is the LQI Quality Index formula")
2. Explains what each variable means based on the context
3. Describes the purpose of the formula

Write a clear, searchable description that would help someone find this formula when asking questions like "what is the formula for X?" or "show me the equation for Y".

Keep the response concise (3-5 sentences) and include the LaTeX formula at the end."""


def enrich_formula_with_context(latex: str, document_text: str) -> str:
    """
    Use LLM to create a rich description of a formula based on document context.
    
    Args:
        latex: The extracted LaTeX formula
        document_text: Surrounding text from the document
        
    Returns:
        Enriched description that explains what the formula represents
    """
    from .config import LLM_MODEL
    
    try:
        client = get_client()
        
        # Extract relevant context (limit to ~1000 chars around formulas)
        context = document_text[:2000] if len(document_text) > 2000 else document_text
        
        prompt = FORMULA_CONTEXT_PROMPT.format(latex=latex, context=context)
        
        response = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "user", "content": prompt}
            ],
            max_tokens=300,
            temperature=0.3
        )
        
        enriched = response.choices[0].message.content.strip()
        return enriched
        
    except Exception as e:
        console.print(f"[yellow]Warning: Could not enrich formula context: {e}[/yellow]")
        # Fall back to basic description
        return f"Mathematical formula: {latex}"


def get_client() -> Groq:
    """Get or initialize the Groq client."""
    global _client
    if _client is None:
        if not GROQ_API_KEY:
            raise ValueError("GROQ_API_KEY not set. Please set it in .env file or environment.")
        _client = Groq(api_key=GROQ_API_KEY)
    return _client


def encode_image_base64(image_path: str | Path) -> str:
    """Encode an image file to base64."""
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def get_image_media_type(image_path: str | Path) -> str:
    """Get the media type for an image based on its extension."""
    ext = Path(image_path).suffix.lower()
    media_types = {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".gif": "image/gif",
        ".webp": "image/webp"
    }
    return media_types.get(ext, "image/png")


def extract_formula_from_image(image_path: str | Path) -> Tuple[bool, Optional[str], str]:
    """
    Detect if an image contains a formula and extract LaTeX.
    
    Args:
        image_path: Path to the image file
        
    Returns:
        Tuple of (is_formula, latex_string, description)
        - is_formula: True if image contains a mathematical formula
        - latex_string: Extracted LaTeX if is_formula, None otherwise
        - description: Text description or LaTeX representation
    """
    image_path = Path(image_path)
    if not image_path.exists():
        return False, None, f"[Image not found: {image_path}]"
    
    try:
        client = get_client()
        
        # Encode image to base64
        image_data = encode_image_base64(image_path)
        media_type = get_image_media_type(image_path)
        
        response = client.chat.completions.create(
            model=VISION_MODEL,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": FORMULA_EXTRACTION_PROMPT
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:{media_type};base64,{image_data}"
                            }
                        }
                    ]
                }
            ],
            max_tokens=500
        )
        
        result = response.choices[0].message.content
        
        # Parse the response to check if it's a formula
        if "FORMULA: YES" in result.upper():
            # Extract the LaTeX part (everything after the first line)
            lines = result.strip().split('\n')
            latex_lines = [l for l in lines[1:] if l.strip() and "FORMULA:" not in l.upper()]
            raw_latex = '\n'.join(latex_lines).strip()
            
            # Clean up common artifacts
            raw_latex = raw_latex.replace('```latex', '').replace('```', '').strip()
            
            # Normalize the LaTeX
            normalized = normalize_latex(raw_latex)
            description = create_formula_description(normalized)
            
            return True, raw_latex, description
        else:
            # Not a formula - extract description
            lines = result.strip().split('\n')
            desc_lines = [l for l in lines if "FORMULA:" not in l.upper()]
            description = '\n'.join(desc_lines).strip()
            return False, None, description
            
    except Exception as e:
        console.print(f"[yellow]Warning: Could not analyze image {image_path.name}: {e}[/yellow]")
        return False, None, f"[Image: {image_path.name}]"


def describe_image(image_path: str | Path) -> str:
    """
    Generate a text description of an image using Groq's Llama 3.2 Vision.
    
    Args:
        image_path: Path to the image file
        
    Returns:
        Text description of the image
    """
    image_path = Path(image_path)
    if not image_path.exists():
        return f"[Image not found: {image_path}]"
    
    try:
        client = get_client()
        
        # Encode image to base64
        image_data = encode_image_base64(image_path)
        media_type = get_image_media_type(image_path)
        
        response = client.chat.completions.create(
            model=VISION_MODEL,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": IMAGE_ANALYSIS_PROMPT
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:{media_type};base64,{image_data}"
                            }
                        }
                    ]
                }
            ],
            max_tokens=500
        )
        return response.choices[0].message.content
        
    except Exception as e:
        console.print(f"[yellow]Warning: Could not describe image {image_path.name}: {e}[/yellow]")
        return f"[Image: {image_path.name}]"

def process_images(images: List[Dict[str, Any]], document_text: str = "") -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Process all images: classify as formula or regular image.
    
    For formulas: extract LaTeX and enrich with document context
    For images: generate description
    
    Args:
        images: List of image dicts with 'path' and 'index' keys
        document_text: Full document text for context enrichment
        
    Returns:
        Tuple of (regular_images, formulas)
        - regular_images: Only non-formula images with 'description'
        - formulas: List of formulas with 'latex', 'enriched_description', 'image_index'
    """
    if not images:
        return [], []
    
    console.print(f"[blue]Processing {len(images)} images...[/blue]")
    
    regular_images = []
    formulas = []
    
    for i, img in enumerate(images):
        img_index = img.get("index", i + 1)
        console.print(f"  Analyzing image {img_index}...")
        
        # Try to detect if it's a formula
        is_formula, latex, description = extract_formula_from_image(img["path"])
        
        if is_formula and latex:
            console.print(f"    [cyan]✓ Formula detected:[/cyan] {latex[:40]}...")
            
            # Enrich with document context
            enriched = enrich_formula_with_context(latex, document_text)
            
            formulas.append({
                "latex": latex,
                "normalized": normalize_latex(latex),
                "enriched_description": enriched,
                "image_index": img_index,
                "source_image": img["path"]
            })
        else:
            # Regular image - keep with description
            img["description"] = description
            regular_images.append(img)
            console.print(f"    [green]✓ Image description:[/green] {len(description)} chars")
    
    console.print(f"  [green]✓ Found {len(formulas)} formulas, {len(regular_images)} regular images[/green]")
    
    return regular_images, formulas


def inline_formulas_in_markdown(markdown_text: str, formulas: List[Dict[str, Any]]) -> str:
    """
    Replace image placeholders in markdown with LaTeX formulas.
    
    Docling exports images as `<!-- image -->` comments in markdown.
    This function replaces those placeholders with the corresponding LaTeX.
    
    Args:
        markdown_text: Original markdown with image placeholders
        formulas: List of formula dicts with 'latex' and 'image_index'
        
    Returns:
        Updated markdown with formulas inlined as $$LaTeX$$
    """
    import re
    
    if not formulas:
        return markdown_text
    
    # Find all image placeholders
    # Docling uses <!-- image --> or similar patterns
    placeholder_pattern = r'<!--\s*image\s*-->'
    
    # Find all placeholder positions
    placeholders = list(re.finditer(placeholder_pattern, markdown_text, re.IGNORECASE))
    
    if not placeholders:
        console.print("[yellow]No image placeholders found in markdown[/yellow]")
        return markdown_text
    
    # Sort formulas by image_index to match placeholder order
    sorted_formulas = sorted(formulas, key=lambda f: f.get("image_index", 0))
    
    # Build mapping of placeholder index to formula
    # Assumption: formulas appear in the same order as their source images
    formula_by_index = {f["image_index"]: f for f in sorted_formulas}
    
    # Replace placeholders from end to beginning (to preserve positions)
    result = markdown_text
    for i, match in enumerate(reversed(placeholders), 1):
        placeholder_idx = len(placeholders) - i + 1
        
        if placeholder_idx in formula_by_index:
            formula = formula_by_index[placeholder_idx]
            latex = formula["latex"]
            # Replace with display math block
            replacement = f"\n\n$${latex}$$\n\n"
            result = result[:match.start()] + replacement + result[match.end():]
            console.print(f"  [cyan]✓ Inlined formula at position {placeholder_idx}[/cyan]")
    
    return result


# Keep the old function name for backwards compatibility but delegate to new one
def describe_all_images(images: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    DEPRECATED: Use process_images() instead.
    This is kept for backwards compatibility.
    """
    return process_images(images, "")
