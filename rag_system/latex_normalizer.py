"""LaTeX formula normalization for consistent embeddings."""
import re
from typing import List, Tuple


def normalize_latex(latex: str) -> str:
    """
    Normalize LaTeX formula for consistent embedding.
    
    Applies various transformations to ensure semantically 
    similar formulas result in similar vector positions.
    
    Args:
        latex: Raw LaTeX string
        
    Returns:
        Normalized LaTeX string
    """
    if not latex:
        return ""
    
    normalized = latex.strip()
    
    # Remove surrounding $ or $$ delimiters if present
    normalized = re.sub(r'^\$\$?|\$\$?$', '', normalized).strip()
    
    # Normalize whitespace
    normalized = re.sub(r'\s+', ' ', normalized)
    
    # Apply normalization rules
    normalized = _normalize_fractions(normalized)
    normalized = _normalize_functions(normalized)
    normalized = _normalize_operators(normalized)
    normalized = _normalize_roots(normalized)
    normalized = _normalize_subscripts_superscripts(normalized)
    normalized = _normalize_greek_letters(normalized)
    normalized = _normalize_brackets(normalized)
    
    return normalized.strip()


def _normalize_fractions(latex: str) -> str:
    """Convert simple fractions to \\frac notation."""
    # Handle simple single character fractions: a/b -> \frac{a}{b}
    # Match a single char followed by / followed by a single char
    pattern = r'(?<![\\a-zA-Z])([a-zA-Z0-9])/([a-zA-Z0-9])(?![a-zA-Z])'
    result = re.sub(pattern, r'\\frac{\1}{\2}', latex)
    
    # Handle parenthesized fractions: (expr)/(expr) -> \frac{expr}{expr}
    pattern = r'\(([^()]+)\)/\(([^()]+)\)'
    result = re.sub(pattern, r'\\frac{\1}{\2}', result)
    
    return result


def _normalize_functions(latex: str) -> str:
    """Ensure standard function notation."""
    # List of common math functions that should use backslash
    functions = [
        'sin', 'cos', 'tan', 'cot', 'sec', 'csc',
        'sinh', 'cosh', 'tanh', 'coth',
        'arcsin', 'arccos', 'arctan',
        'asin', 'acos', 'atan',
        'log', 'ln', 'exp', 'lg',
        'lim', 'max', 'min', 'sup', 'inf',
        'det', 'dim', 'ker', 'deg',
        'gcd', 'lcm', 'mod',
        'arg', 'sgn', 'abs'
    ]
    
    result = latex
    for func in functions:
        # Match function name not preceded by backslash
        pattern = r'(?<!\\)\b' + func + r'\b'
        replacement = '\\' + func
        result = re.sub(pattern, lambda m: replacement, result)
    
    return result


def _normalize_operators(latex: str) -> str:
    """Normalize mathematical operators."""
    result = latex
    
    # Multiplication: * -> \cdot
    result = re.sub(r'\*', r' \\cdot ', result)
    
    # Comparison operators
    result = re.sub(r'>=', r'\\geq', result)
    result = re.sub(r'<=', r'\\leq', result)
    result = re.sub(r'!=', r'\\neq', result)
    result = re.sub(r'~=', r'\\approx', result)
    
    # Arrows
    result = re.sub(r'->', r'\\rightarrow', result)
    result = re.sub(r'<-', r'\\leftarrow', result)
    result = re.sub(r'=>', r'\\Rightarrow', result)
    
    # Infinity
    result = re.sub(r'\binfinity\b', r'\\infty', result)
    
    # Plus/minus
    result = re.sub(r'\+/-', r'\\pm', result)
    result = re.sub(r'-/\+', r'\\mp', result)
    
    return result


def _normalize_roots(latex: str) -> str:
    """Normalize square root and nth root notation."""
    # sqrt(x) -> \sqrt{x}
    result = re.sub(r'(?<!\\)sqrt\(([^)]+)\)', r'\\sqrt{\1}', latex)
    
    # cbrt(x) -> \sqrt[3]{x}
    result = re.sub(r'(?<!\\)cbrt\(([^)]+)\)', r'\\sqrt[3]{\1}', result)
    
    # nthroot(n, x) -> \sqrt[n]{x}
    result = re.sub(r'(?<!\\)nthroot\((\d+),\s*([^)]+)\)', r'\\sqrt[\1]{\2}', result)
    
    return result


def _normalize_subscripts_superscripts(latex: str) -> str:
    """Ensure consistent subscript/superscript notation."""
    # x_ab -> x_{ab} (multi-character subscripts need braces)
    result = re.sub(r'_([a-zA-Z0-9]{2,})(?![{}])', r'_{\1}', latex)
    
    # x^ab -> x^{ab} (multi-character superscripts need braces)
    result = re.sub(r'\^([a-zA-Z0-9]{2,})(?![{}])', r'^{\1}', result)
    
    # Normalize common superscripts
    result = re.sub(r'\^2(?![{}0-9])', r'^{2}', result)
    result = re.sub(r'\^3(?![{}0-9])', r'^{3}', result)
    result = re.sub(r'\^n(?![{}a-zA-Z])', r'^{n}', result)
    
    return result


def _normalize_greek_letters(latex: str) -> str:
    """Ensure Greek letters use proper LaTeX commands."""
    greek_letters = [
        ('alpha', r'\alpha'), ('beta', r'\beta'), ('gamma', r'\gamma'),
        ('delta', r'\delta'), ('epsilon', r'\epsilon'), ('zeta', r'\zeta'),
        ('eta', r'\eta'), ('theta', r'\theta'), ('iota', r'\iota'),
        ('kappa', r'\kappa'), ('mu', r'\mu'),
        ('nu', r'\nu'), ('xi', r'\xi'), 
        ('rho', r'\rho'), ('sigma', r'\sigma'), ('tau', r'\tau'),
        ('upsilon', r'\upsilon'), ('phi', r'\phi'), ('chi', r'\chi'),
        ('psi', r'\psi'), ('omega', r'\omega'),
        ('pi', r'\pi'),  # Put after upsilon to avoid matching 'pi' in 'upsilon'
        ('lambda', r'\lambda'),  # Handle lambda separately
        # Uppercase
        ('Gamma', r'\Gamma'), ('Delta', r'\Delta'), ('Theta', r'\Theta'),
        ('Lambda', r'\Lambda'), ('Xi', r'\Xi'), ('Pi', r'\Pi'),
        ('Sigma', r'\Sigma'), ('Phi', r'\Phi'), ('Psi', r'\Psi'),
        ('Omega', r'\Omega')
    ]
    
    result = latex
    for name, cmd in greek_letters:
        # Only replace if not already a LaTeX command
        # Use a function replacement to avoid escape issues
        pattern = r'(?<!\\)\b' + name + r'\b'
        result = re.sub(pattern, lambda m: cmd, result)
    
    return result


def _normalize_brackets(latex: str) -> str:
    """Normalize bracket notation."""
    # Use \left and \right for auto-sizing brackets when appropriate
    # This is a simple heuristic - complex expressions benefit from auto-sizing
    
    # Already has \left or \right, skip
    if '\\left' in latex or '\\right' in latex:
        return latex
    
    return latex


def extract_formulas_from_text(text: str) -> List[Tuple[str, int, int]]:
    """
    Extract LaTeX formulas from text.
    
    Args:
        text: Text containing LaTeX formulas
        
    Returns:
        List of tuples: (formula, start_pos, end_pos)
    """
    formulas = []
    
    # Match $...$ (inline) and $$...$$ (display) formulas
    patterns = [
        r'\$\$(.+?)\$\$',  # Display math
        r'\$(.+?)\$',      # Inline math
        r'\\\[(.+?)\\\]',  # Display math alt
        r'\\\((.+?)\\\)',  # Inline math alt
    ]
    
    for pattern in patterns:
        for match in re.finditer(pattern, text, re.DOTALL):
            formulas.append((match.group(0), match.start(), match.end()))
    
    return formulas


def create_formula_description(latex: str) -> str:
    """
    Create a natural language description of a formula for embedding.
    
    This helps the embedding model understand the semantic meaning
    of the mathematical expression.
    
    Args:
        latex: Normalized LaTeX string
        
    Returns:
        Description string combining LaTeX and natural language
    """
    description_parts = ["Mathematical formula:"]
    
    # Detect formula type and add description
    if '\\frac' in latex:
        description_parts.append("fraction/ratio")
    if '\\sum' in latex:
        description_parts.append("summation")
    if '\\int' in latex:
        description_parts.append("integral")
    if '\\prod' in latex:
        description_parts.append("product")
    if '\\lim' in latex:
        description_parts.append("limit")
    if any(f in latex for f in ['\\sin', '\\cos', '\\tan']):
        description_parts.append("trigonometric")
    if '\\sqrt' in latex:
        description_parts.append("square root")
    if '=' in latex:
        description_parts.append("equation")
    if any(op in latex for op in ['\\geq', '\\leq', '>', '<']):
        description_parts.append("inequality")
    if '\\partial' in latex:
        description_parts.append("partial derivative")
    if "\\nabla" in latex or "\\grad" in latex:
        description_parts.append("gradient")
    if '\\matrix' in latex or '\\begin{matrix}' in latex:
        description_parts.append("matrix")
    
    # Combine description with the normalized LaTeX
    desc = " ".join(description_parts)
    return f"{desc}\nLaTeX: {latex}"
