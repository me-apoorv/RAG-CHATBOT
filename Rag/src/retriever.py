import json
import numpy as np
import faiss
import os
from sentence_transformers import SentenceTransformer


class Retriever:
    def __init__(self, vectordb_dir: str = './vectordb'):
        """Load per-document FAISS indices from registry.
        
        Paths are relative to current working directory.
        """
        self.vectordb_dir = vectordb_dir
        self.indices = {}  # dict: doc_name -> {'index': index, 'metadata': [...]}
        
        # Load registry
        registry_path = os.path.join(vectordb_dir, 'index_registry.json')
        try:
            with open(registry_path, 'r', encoding='utf-8') as f:
                registry = json.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"Index registry not found: {registry_path}")

        # Load each document's index and metadata
        for doc_name, info in registry.items():
            try:
                index_path = info['index_path']
                metadata_path = info['metadata_path']
                
                # Load FAISS index
                index = faiss.read_index(index_path)
                
                # Load metadata
                with open(metadata_path, 'r', encoding='utf-8') as f:
                    metadata = json.load(f)
                
                self.indices[doc_name] = {
                    'index': index,
                    'metadata': metadata,
                    'ntotal': index.ntotal
                }
            except Exception as e:
                print(f"Warning: Failed to load index for document '{doc_name}': {e}")
                continue

        if not self.indices:
            raise RuntimeError("No indices loaded. Check vectordb directory.")

        # Load embedding model
        try:
            self.model = SentenceTransformer('all-MiniLM-L6-v2')
        except Exception as e:
            raise RuntimeError(f"Failed to load sentence-transformers model: {e}")

    def search(self, query: str, top_k: int = 3) -> list[dict]:
        """Embed the query, search all document indices, and return matching chunk texts
        with document source.
        
        Returns a list of dicts: [{'text': str, 'document': str}, ...]
        ordered by similarity across all documents.
        """
        if not query or not self.indices:
            return []

        # Embed query
        emb = self.model.encode([query], convert_to_numpy=True)
        emb = np.asarray(emb, dtype=np.float32)
        if emb.ndim == 1:
            emb = emb.reshape(1, -1)

        # Search each document's index and collect results
        all_results = []
        for doc_name, doc_data in self.indices.items():
            index = doc_data['index']
            metadata = doc_data['metadata']
            ntotal = doc_data['ntotal']
            
            k = min(top_k, max(1, ntotal))
            
            try:
                distances, indices_arr = index.search(emb, k)
                
                # Collect results from this document
                for idx in indices_arr[0]:
                    if idx < 0:
                        continue
                    try:
                        chunk_data = metadata[idx]
                        text = chunk_data.get('text', '')
                        distance = float(distances[0][list(indices_arr[0]).index(idx)])
                        
                        all_results.append({
                            'text': text,
                            'document': doc_name,
                            'distance': distance
                        })
                    except (IndexError, KeyError):
                        continue
            except Exception as e:
                print(f"Warning: Search failed in document '{doc_name}': {e}")
                continue

        # Sort by distance (lower is better for L2) and return top-k
        all_results.sort(key=lambda x: x['distance'])
        return all_results[:top_k]

    def chunk_count(self) -> int:
        """Return total number of indexed chunks across all documents."""
        return sum(doc_data['ntotal'] for doc_data in self.indices.values())


if __name__ == '__main__':
    # Quick smoke test — will raise if files missing
    try:
        r = Retriever()
        print('Total chunk count:', r.chunk_count())
        print('Documents:', list(r.indices.keys()))
        # Example search
        res = r.search('What is RAG?', top_k=3)
        print('Top results:')
        for r_item in res:
            print(f"  - From '{r_item['document']}': {r_item['text'][:80]}...")
    except Exception as e:
        print('Retriever initialization/search failed:', e)
