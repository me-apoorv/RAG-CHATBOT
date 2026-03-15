
import os
import re
import json
import pdfplumber
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter


def find_pdf_in_data(data_dir):
    if not os.path.isdir(data_dir):
        return None
    for name in os.listdir(data_dir):
        if name.lower().endswith('.pdf'):
            return os.path.join(data_dir, name)
    return None


def extract_pages_text(pdf_path):
    pages = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text() or ''
            # split into lines so we can analyze headers/footers
            lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
            pages.append(lines)
    return pages


def detect_repeated_lines(pages, head_tail_lines=2, threshold_ratio=0.5):
    # Collect candidate header/footer lines from first/last N lines of each page
    counts = {}
    total = len(pages) or 1
    for lines in pages:
        # first head_tail_lines lines
        for ln in lines[:head_tail_lines]:
            counts[ln] = counts.get(ln, 0) + 1
        # last head_tail_lines lines
        for ln in lines[-head_tail_lines:]:
            counts[ln] = counts.get(ln, 0) + 1

    repeated = set()
    for ln, c in counts.items():
        if c / total >= threshold_ratio:
            repeated.add(ln)
    return repeated


def remove_page_numbers_and_misc(lines):
    cleaned = []
    for ln in lines:
        # Skip lines that are just page numbers or 'Page X' or 'X / Y'
        if re.fullmatch(r"\d+", ln):
            continue
        if re.fullmatch(r"[Pp]age\s+\d+", ln):
            continue
        if re.fullmatch(r"\d+\s*/\s*\d+", ln):
            continue
        if re.fullmatch(r"[Pp]age\s+\d+\s+of\s+\d+", ln):
            continue
        cleaned.append(ln)
    return cleaned


def clean_text_from_pages(pages):
    # Detect common headers/footers
    repeated = detect_repeated_lines(pages, head_tail_lines=2, threshold_ratio=0.5)

    page_texts = []
    for lines in pages:
        # remove repeated header/footer lines
        lines = [ln for ln in lines if ln not in repeated]
        # remove page numbers or small misc lines
        lines = remove_page_numbers_and_misc(lines)
        # join lines for the page
        page_texts.append(' '.join(lines))

    full = '\n\n'.join(p for p in page_texts if p)

    # Remove non-standard special characters but keep common punctuation
    # Keep letters, numbers, whitespace and . , ; : - ? ! ' "
    full = re.sub(r"[^0-9A-Za-z\s\.\,\;\:\-\?\!\'\"()\[\]]+", ' ', full)

    # Normalize whitespace (including newlines -> single space) and strip
    full = re.sub(r"\s+", ' ', full).strip()

    return full


def save_cleaned_text(text, out_path):
    out_dir = os.path.dirname(out_path)
    if out_dir and not os.path.isdir(out_dir):
        os.makedirs(out_dir, exist_ok=True)
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write(text)


def word_count(text):
    if not text:
        return 0
    return len(text.split())


def chunk_text(doc_name=None):
    """Read cleaned text from data/cleaned.txt, split into chunks using
    LangChain RecursiveCharacterTextSplitter (chunk_size=500, chunk_overlap=150),
    save chunks as list of dicts to <project_root>/chunks/chunks.json, and
    print the total number of chunks created. Each chunk includes document name.
    """
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    data_dir = os.path.join(project_root, 'data')
    cleaned_path = os.path.join(data_dir, 'cleaned.txt')

    if not os.path.isfile(cleaned_path):
        print('Cleaned file not found at', cleaned_path)
        return 0

    with open(cleaned_path, 'r', encoding='utf-8') as f:
        text = f.read()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=150,
        separators=["\n\n", "\n", ". ", "? ", "! ", " ", ""],
        # Tries paragraph → sentence → word before character split
    )
    chunks = splitter.split_text(text)

    chunk_dicts = []
    for i, c in enumerate(chunks, start=1):
        chunk_dicts.append({
            'chunk_id': i,
            'text': c,
            'word_count': len(c.split()),
            'document': doc_name or 'unknown'
        })

    out_dir = os.path.join(project_root, 'chunks')
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, 'chunks.json')
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(chunk_dicts, f, ensure_ascii=False, indent=2)

    total = len(chunk_dicts)
    print(f'Chunks saved to: {out_path}')
    print(f'Total chunks created: {total}')
    return total


def build_vectordb():
    """Load chunks from chunks/chunks.json, generate embeddings using
    sentence-transformers 'all-MiniLM-L6-v2', store in separate FAISS indices
    per document, and save under vectordb/. Prints indices created.
    """
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    chunks_path = os.path.join(project_root, 'chunks', 'chunks.json')

    if not os.path.isfile(chunks_path):
        print('Chunks file not found at', chunks_path)
        return 0

    with open(chunks_path, 'r', encoding='utf-8') as f:
        chunks = json.load(f)

    if not chunks:
        print('No chunks found')
        return 0

    # Load sentence-transformers model
    model = SentenceTransformer('all-MiniLM-L6-v2')

    # Group chunks by document
    docs_chunks = {}
    for chunk in chunks:
        doc_name = chunk.get('document', 'unknown')
        if doc_name not in docs_chunks:
            docs_chunks[doc_name] = []
        docs_chunks[doc_name].append(chunk)

    # Create vectordb directory
    vectordb_root = os.path.join(project_root, 'vectordb')
    os.makedirs(vectordb_root, exist_ok=True)

    # Create per-document indices
    index_registry = {}
    total_vectors = 0

    for doc_name, doc_chunks in docs_chunks.items():
        texts = [c.get('text', '') for c in doc_chunks]
        
        # Generate embeddings
        embeddings = model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
        embeddings = np.asarray(embeddings, dtype=np.float32)
        if embeddings.ndim == 1:
            embeddings = embeddings.reshape(1, -1)

        dim = embeddings.shape[1]

        # Build FAISS index for this document
        index = faiss.IndexFlatL2(dim)
        index.add(embeddings)

        # Create safe filename from document name
        safe_name = "".join(c for c in doc_name if c.isalnum() or c in ('-', '_')).rstrip()
        safe_name = safe_name or 'document'

        # Save index
        index_path = os.path.join(vectordb_root, f'{safe_name}_index.faiss')
        faiss.write_index(index, index_path)

        # Save metadata with document source
        metadata = [
            {
                'chunk_id': c.get('chunk_id'),
                'text': c.get('text'),
                'document': doc_name
            }
            for c in doc_chunks
        ]
        meta_path = os.path.join(vectordb_root, f'{safe_name}_metadata.json')
        with open(meta_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)

        index_registry[doc_name] = {
            'index_path': index_path,
            'metadata_path': meta_path,
            'vector_count': index.ntotal
        }

        total_vectors += index.ntotal
        print(f'  Document "{doc_name}": {index.ntotal} vectors')

    # Save registry
    registry_path = os.path.join(vectordb_root, 'index_registry.json')
    with open(registry_path, 'w', encoding='utf-8') as f:
        json.dump(index_registry, f, ensure_ascii=False, indent=2)

    print(f'Vector DB built successfully. Total vectors: {total_vectors}')
    print(f'Registry saved to: {registry_path}')
    return total_vectors


def main():
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    data_dir = os.path.join(project_root, 'data')
    pdf_path = find_pdf_in_data(data_dir)
    if not pdf_path:
        print('No PDF file found in', data_dir)
        return

    # Extract document name from PDF filename (without extension)
    doc_name = os.path.splitext(os.path.basename(pdf_path))[0]

    pages = extract_pages_text(pdf_path)
    cleaned = clean_text_from_pages(pages)

    out_path = os.path.join(data_dir, 'cleaned.txt')
    save_cleaned_text(cleaned, out_path)

    wc = word_count(cleaned)
    print(f'Cleaned text saved to: {out_path}')
    print(f'Document: "{doc_name}"')
    print(f'Total word count after cleaning: {wc}')
    # Create chunks from the cleaned text for downstream RAG pipeline
    chunk_text(doc_name=doc_name)


if __name__ == '__main__':
    main()
