import os




class Generator:
    def __init__(self, model_path: str | None = None):
        # Resolve default model path relative to project root if not provided
        if model_path is None:
            project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
            model_path = os.path.join(project_root, 'models', 'Phi-3-mini-4k-instruct-q4.gguf')

        # Try to initialize Llama. If llama_cpp isn't installed yet, keep
        # initialization lazy so the rest of the app can run (uploader/retriever).
        self.model_path = model_path
        self.llm = None
        try:
            import llama_cpp  # local import to avoid hard dependency at module import time

            self.llm = llama_cpp.Llama(
                model_path=model_path,
                n_ctx=4096,
                n_threads=6,
                n_batch=512,
                verbose=False,
            )
        except Exception:
            # Leave self.llm as None; stream_response will raise a helpful error
            self.llm = None

    def build_prompt(self, query: str, chunks: list) -> str:
        """Format the prompt using the Phi-3 chat template.

        Chunks can be strings or dicts with 'text' and 'document' keys.
        Template:
        <|system|> ... <|end|>
        <|user|> Context: {chunks} Question: {query} <|end|>
        <|assistant|>
        """
        system = "<|system|> You are a helpful assistant. Answer only based on the provided context. <|end|>"

        # Extract text from chunks (handle both string and dict formats)
        chunk_texts = []
        for chunk in chunks:
            if isinstance(chunk, dict):
                text = chunk.get('text', '')
                doc = chunk.get('document', 'unknown')
                chunk_texts.append(f"[From {doc}] {text}")
            else:
                chunk_texts.append(str(chunk))
        
        # Join chunks with separators to preserve boundaries
        joined = '\n\n'.join(chunk_texts) if chunk_texts else '(No relevant context found)'
        user = f"<|user|> Context: {joined} Question: {query} <|end|>\n<|assistant|>"
        return f"{system}\n{user}"

    def stream_response(self, query, chunks):
        """Generator that yields token text pieces from the model stream.

        Calls the underlying Llama instance with stream=True, max_tokens=512,
        temperature=0.2 and yields token pieces from
        token["choices"][0]["text"].
        """
        prompt = self.build_prompt(query, chunks)

        # Ensure the model is initialized
        if self.llm is None:
            # Try one more time to import and initialize, to handle late installs
            try:
                import llama_cpp
                self.llm = llama_cpp.Llama(
                    model_path=self.model_path,
                    n_ctx=4096,
                    n_threads=6,
                    n_batch=512,
                    verbose=False,
                )
            except Exception as e:
                raise RuntimeError(
                    "Local LLM runtime not available. Install 'llama-cpp-python' and ensure the GGUF model exists at the configured path."
                ) from e

        # llama-cpp-python supports calling the instance with streaming
        for token in self.llm(prompt=prompt, stream=True, max_tokens=512, temperature=0.2):
            # Each token is expected to be a dict with 'choices'
            try:
                text = token["choices"][0].get("text", "")
            except Exception:
                # Skip malformed pieces
                continue

            if text:
                yield text


def create_default_generator():
    return Generator()


if __name__ == '__main__':
    # Quick manual test (won't run model here unless model exists)
    gen = Generator()
    # Example usage:
    # for piece in gen.stream_response('What is RAG?', ['Doc chunk 1', 'Doc chunk 2']):
    #     print(piece, end='', flush=True)
    print('Generator initialized with model at', gen.llm.model_path)
