from src.retriever import Retriever
from src.generator import Generator


class RAGPipeline:
    """Simple RAG pipeline that wires a Retriever and a Generator.

    Usage:
        pipeline = RAGPipeline()
        token_gen, sources = pipeline.ask('What is RAG?')
        for token in token_gen:
            print(token, end='', flush=True)
        # sources contains the list of source chunk texts
    """

    def __init__(self, retriever: Retriever | None = None, generator: Generator | None = None):
        # Allow dependency injection for easier testing
        self.retriever = retriever or Retriever()
        self.generator = generator or Generator()

    def ask(self, query: str, top_k: int = 5):
        """Return (token_generator, source_chunks).

        - token_generator: an iterator yielding token text from the generator
        - source_chunks: list[str] of retrieved chunk texts used as context
        - top_k: number of chunks to retrieve (default: 5)
        """
        # Retrieve relevant chunks (as strings)
        source_chunks = self.retriever.search(query, top_k=top_k)

        # Create token generator from the generator component
        token_generator = self.generator.stream_response(query, source_chunks)

        return token_generator, source_chunks


if __name__ == '__main__':
    # Quick smoke test (won't actually stream unless models/index exist)
    try:
        p = RAGPipeline()
        tg, srcs = p.ask('Explain retrieval-augmented generation in simple terms')
        print('Source chunks:', len(srcs))
        # To stream tokens, iterate tg (example commented):
        # for t in tg:
        #     print(t, end='', flush=True)
    except Exception as e:
        print('RAGPipeline initialization failed:', e)
