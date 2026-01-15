# 📚 Local RAG System with Hybrid Formula Extraction

A fully local, CPU-based multimodal Retrieval-Augmented Generation (RAG) system that processes PDF documents with text, tables, images, and **mathematical formulas** using a hybrid PyMuPDF + Docling approach.

## ✨ Features

- **🔍 PDF Parsing** - Extract text, tables and images using Docling
- **🖼️ Multimodal Support** - Image analysis using Llama 4 Scout vision model
- **📐 Hybrid Formula Extraction** - Combines PyMuPDF (fitz) vector detection with vision model LaTeX extraction
- **🧠 Semantic Search** - BGE embeddings with ChromaDB vector store
- **⚡ Fast Reranking** - FlashRank for improved retrieval accuracy
- **💬 Interactive Chat** - CLI-based chat interface with streaming responses
- **🆓 Free to Run** - Uses Groq API (free tier available) for LLM inference

## 🔬 How Formula Extraction Works

This system uses a **hybrid approach** combining PyMuPDF (fitz) and Docling for accurate mathematical formula extraction:

### The Marker Method

1. **Vector Detection (fitz)**: PyMuPDF scans PDF pages for vector drawing clusters that represent mathematical formulas (filtering out tables, headers, and footers)

2. **Formula Image Extraction**: Each detected formula region is extracted as a high-resolution PNG image (4x scale for quality)

3. **Marker Insertion**: Unique text markers (e.g., `##FORMULA_001##`) are inserted at formula locations in a temporary PDF copy

4. **Docling Parsing**: The marked PDF is processed by Docling, which preserves the markers in the markdown output

5. **Vision Model LaTeX Extraction**: Each formula image is sent to a vision model (Llama 4 Scout) to extract the LaTeX representation

6. **Marker Replacement**: The markers in the markdown are replaced with the extracted LaTeX formulas (wrapped in `$$...$$`)

### Why This Approach?

- **Docling's limitation**: Docling may output formula labels (e.g., "Formule 2.3") instead of actual equations for vector-based formulas
- **PyMuPDF's strength**: Excellent at detecting and extracting vector graphics regions
- **Vision model bridge**: Converts visual formula images to semantic LaTeX text

## ⚠️ Important: Vision Model Quality

> **The accuracy of extracted LaTeX formulas depends heavily on the vision model used.**

The current implementation uses `meta-llama/llama-4-scout-17b-16e-instruct` via Groq API. While functional, this may produce:

- **Approximate LaTeX**: Symbols might be slightly off (e.g., `\lambda` vs `\Lambda`)
- **Missing subscripts/superscripts**: Complex nested formulas may lose some detail
- **Formatting variations**: Different but mathematically equivalent representations

### For More Accurate Embeddings

To improve formula extraction accuracy, consider:

1. **Better Vision Models**: Use more capable vision models like GPT-4V, Claude 3.5 Sonnet, or Gemini Pro Vision
2. **Specialized OCR**: Integrate dedicated LaTeX OCR tools like Mathpix or LaTeX-OCR
3. **Fine-tuned Models**: Use vision models fine-tuned specifically for mathematical notation

The embedding quality directly correlates with LaTeX accuracy - more precise formulas lead to better semantic matching during retrieval.

## 🏗️ Architecture

```
PDF Document
     │
     ├──────────────────────────────────────────────────────┐
     │                                                      │
     ▼                                                      ▼
┌─────────────────┐                              ┌──────────────────┐
│  PyMuPDF (fitz) │                              │     Docling      │
│                 │                              │                  │
│ • Detect vector │     Insert Markers           │ • Parse text     │
│   formula regions ─────────────────────────────► • Extract tables │
│ • Extract as PNG │                             │ • Process images │
└────────┬────────┘                              └────────┬─────────┘
         │                                                │
         ▼                                                ▼
┌─────────────────┐                              ┌──────────────────┐
│  Vision Model   │                              │    Markdown      │
│  (Llama 4 Scout)│      Replace Markers         │  with ##MARKERS##│
│                 │ ◄────────────────────────────┤                  │
│ Image → LaTeX   │                              │                  │
└────────┬────────┘                              └──────────────────┘
         │
         ▼
┌──────────────────────────────────────────────────────────────────┐
│                    Final Markdown with LaTeX                      │
│                                                                   │
│  Text Chunks ──→ Embeddings ──┐                                  │
│  Tables ───────→ Embeddings ──┼──→ ChromaDB ──→ Query/Retrieve   │
│  Formulas ─────→ Embeddings ──┘                                  │
│  Images ───────→ Vision Desc → Embeddings                        │
└──────────────────────────────────────────────────────────────────┘
```

## 🛠️ Tech Stack

| Component | Technology |
|-----------|------------|
| PDF Parsing | [Docling](https://github.com/DS4SD/docling) |
| Vector Formula Detection | [PyMuPDF (fitz)](https://pymupdf.readthedocs.io/) |
| Formula LaTeX Extraction | Vision Model (Llama 4 Scout 17B) |
| Embeddings | [sentence-transformers](https://www.sbert.net/) (BGE-small) |
| Vector Store | [ChromaDB](https://www.trychroma.com/) |
| Reranking | [FlashRank](https://github.com/PrithivirajDamodaran/FlashRank) |
| LLM | [Groq](https://groq.com/) (Llama 3.3 70B) |
| CLI | [Typer](https://typer.tiangolo.com/) + [Rich](https://rich.readthedocs.io/) |

## 📦 Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/saadnadeem554/Local-Rag-using-Docling.git
   cd Local-Rag-using-Docling
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv venv
   
   # Windows
   venv\Scripts\activate
   
   # Linux/Mac
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**
   ```bash
   cp .env.example .env
   ```
   
   Edit `.env` and add your Groq API key:
   ```
   GROQ_API_KEY=your_groq_api_key_here
   ```
   
   Get your free API key from [console.groq.com/keys](https://console.groq.com/keys)

## 🚀 Usage

### Interactive Chat Mode
```bash
python main.py chat
```

In chat mode, you can:
- `/ingest <path>` - Ingest a PDF document
- `/stats` - View vector store statistics
- `/clear` - Clear all documents
- `/quit` - Exit the chat

Or just type your question to query the documents!

### CLI Commands

**Ingest a PDF:**
```bash
python main.py ingest path/to/document.pdf
```

**Ask a question:**
```bash
python main.py ask "What is the main topic of the document?"
```

**Stream the response:**
```bash
python main.py ask "Summarize the key points" --stream
```

**View statistics:**
```bash
python main.py stats
```

**Clear vector store:**
```bash
python main.py clear
```

## ⚙️ Configuration

Edit `rag_system/config.py` to customize:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `CHUNK_SIZE` | 500 | Token size for text chunks |
| `CHUNK_OVERLAP` | 50 | Overlap between chunks |
| `TOP_K_RETRIEVAL` | 15 | Documents to retrieve |
| `TOP_K_RERANK` | 5 | Documents after reranking |
| `EMBEDDING_MODEL` | BGE-small-en | Embedding model |
| `LLM_MODEL` | llama-3.3-70b | Generation model |
| `VISION_MODEL` | llama-4-scout-17b | Vision model for formula/image analysis |

## 📁 Project Structure

```
├── main.py                 # CLI entry point
├── requirements.txt        # Dependencies
├── .env.example           # Environment template
├── rag_system/
│   ├── config.py          # Configuration
│   ├── parser.py          # PDF parsing with Docling
│   ├── vector_formula_extractor.py  # PyMuPDF vector formula detection & marker method
│   ├── page_formula_extractor.py    # Page-level formula extraction
│   ├── latex_normalizer.py # LaTeX formula normalization
│   ├── chunker.py         # Text chunking
│   ├── embedder.py        # Embedding generation
│   ├── vector_store.py    # ChromaDB operations
│   ├── reranker.py        # FlashRank reranking
│   ├── image_describer.py # Vision model integration
│   ├── generator.py       # LLM response generation
│   └── pipeline.py        # Main RAG pipeline
└── data/                  # Generated data (gitignored)
    ├── pdfs/              # Stored PDFs
    ├── images/            # Extracted images & formula PNGs
    │   └── <pdf_name>/
    │       ├── pages/          # Full page renders
    │       └── vector_formulas/ # Extracted formula images
    ├── markdown/          # Generated markdown with LaTeX
    └── chroma_db/         # Vector database
```

## 📄 License

MIT License

## 🤝 Contributing

Contributions are welcome! Feel free to open an issue or submit a pull request.
