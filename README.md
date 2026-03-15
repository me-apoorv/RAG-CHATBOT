# RAG Chatbot

## Project overview
This project implements a simple Retrieval-Augmented Generation (RAG) chatbot that runs locally on CPU. It ingests PDF documents, builds a chunked text index with semantic embeddings stored in FAISS, and serves an interactive Streamlit chat UI that streams responses from a local Phi‑3 GGUF model.

## Architecture
ASCII diagram:

```
PDF
  |
  v
 ingest.py (extract & clean)
  |
  v
 chunks/ (chunks.json)
  |
  v
 FAISS (vectordb/index.faiss)
  |
  v
 retriever (src/retriever.py)
  |
  v
 pipeline (src/pipeline.py)
  |
  v
 Streamlit UI (app.py)
```

## Tech stack

| Component | Tool | Reason |
|---|---|---|
| PDF parsing | `pdfplumber` | Reliable PDF text extraction for many layouts |
| Text chunking | `langchain` (text splitters) | Robust, prebuilt text-splitting utilities |
| Embeddings | `sentence-transformers` (`all-MiniLM-L6-v2`) | Fast, compact embeddings suitable for CPU |
| Vector DB | `faiss-cpu` | Efficient similarity search on local machine |
| LLM runtime | `llama-cpp-python` (GGUF) | Run Phi‑3 Mini GGUF models locally on CPU |
| UI | `streamlit` | Lightweight web UI for chat interfaces |
| Model hosting | `huggingface_hub` | Convenient download of GGUF models |

## Setup instructions

1. Clone the repository

```bash
git clone <your-repo-url>
cd RAG
```

2. Install Python dependencies

```bash
python3 -m pip install -r requirements.txt
```

3. Download the Phi‑3 GGUF model

Place the GGUF model file in `./models/` with the filename `Phi-3-mini-4k-instruct-q4.gguf`.

You can download it manually from the model provider and copy it into `models/`, or use a small Python helper:

```python
from huggingface_hub import hf_hub_download

# Example (replace repo_id and filename with the actual HF repo path)
hf_hub_download(repo_id='your-gguf-repo', filename='Phi-3-mini-4k-instruct-q4.gguf', cache_dir='./models')
```

4. Ingest your PDFs (generate cleaned text and chunks)

Put one or more PDFs into the `data/` folder, then run:

```bash
python3 src/ingest.py
```

This will create `data/cleaned.txt`, `chunks/chunks.json`, and (after embedding) `vectordb/` files.

5. Run the Streamlit app

```bash
streamlit run app.py
```

Open the URL Streamlit provides (usually `http://localhost:8501`) to interact with the chatbot.

## Sample queries

- "Summarize the key takeaways from the document." 
- "What steps are recommended for deployment in the paper?"
- "Give me a short definition of the main concept introduced in the PDF."

## Known limitations

- Hallucinations: The model can fabricate details not present in the context; always verify critical facts against sources.
- CPU performance: Running embedding generation, FAISS indexing, and local GGUF inference on CPU can be slow compared to GPU setups.
- Context window: The local Phi‑3 Mini model has a finite context window (configured at 4096 tokens); extremely long contexts will be truncated or require smarter retrieval strategies.

## Folder structure

```
RAG/
├── app.py                  # Streamlit app
├── requirements.txt        # Pinned dependencies
├── README.md
├── models/                 # Put Phi-3 GGUF model here
│   └── Phi-3-mini-4k-instruct-q4.gguf
├── data/                   # Input PDFs and cleaned.txt
│   └── cleaned.txt
├── chunks/                 # chunked text (chunks.json)
│   └── chunks.json
├── vectordb/               # FAISS index and metadata
│   ├── index.faiss
│   └── metadata.json
└── src/
    ├── ingest.py
    ├── retriever.py
    ├── generator.py
    ├── pipeline.py
    └── (other helpers)
```
