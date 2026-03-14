import os
import json
import streamlit as st
from src.pipeline import RAGPipeline


st.set_page_config(page_title='RAG Chatbot')


def init_session():
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    if 'pipeline' not in st.session_state:
        st.session_state.pipeline = None


init_session()


st.title('RAG Chatbot')

# Sidebar: upload + settings
with st.sidebar:
    st.header('Settings')
    st.write('Model name: **Phi-3 Mini Q4 (CPU)**')

    uploaded_file = st.file_uploader('Upload a PDF to ingest', type=['pdf'])
    if uploaded_file is not None:
        # Save uploaded file to data/
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '.'))
        data_dir = os.path.join(project_root, 'data')
        os.makedirs(data_dir, exist_ok=True)
        save_path = os.path.join(data_dir, uploaded_file.name)
        with open(save_path, 'wb') as f:
            f.write(uploaded_file.getbuffer())
        st.success(f'Saved uploaded file to {save_path}')
        if st.button('Ingest uploaded document'):
            # Lazy-import ingest here so Streamlit can start even if pdfplumber
            # or other heavy deps are not available at import time.
            try:
                from src import ingest
            except Exception as e:
                st.error(f'Failed to import ingest module: {e}')
                ingest = None

            # Run ingestion and build vector DB
            with st.spinner('Extracting, cleaning and chunking...'):
                try:
                    if ingest is None:
                        raise RuntimeError('Ingest module not available')
                    ingest.main()
                except Exception as e:
                    st.error(f'Ingest failed: {e}')
            with st.spinner('Building vector DB (may take a while)...'):
                try:
                    if ingest is None:
                        raise RuntimeError('Ingest module not available')
                    ingest.build_vectordb()
                except Exception as e:
                    st.error(f'Build vectordb failed: {e}')

            # Initialize pipeline after vectordb is ready
            try:
                st.session_state.pipeline = RAGPipeline()
                st.success('Document ingested and pipeline initialized')
            except Exception as e:
                st.error(f'Pipeline init error: {e}')

    st.markdown('---')
    st.subheader('Download GGUF model from Hugging Face (optional)')
    hf_repo = st.text_input('Hugging Face repo id (e.g. owner/model-name)', '')
    hf_filename = st.text_input('Filename in repo (default: Phi-3-mini-4k-instruct-q4.gguf)', 'Phi-3-mini-4k-instruct-q4.gguf')
    hf_token = st.text_input('HF token (paste if repo is private)', type='password')
    if st.button('Download model'):
        if not hf_repo:
            st.error('Please provide a Hugging Face repo id')
        else:
            with st.spinner('Downloading model from Hugging Face...'):
                try:
                    # Import here to avoid adding dependency at top-level when not needed
                    from huggingface_hub import hf_hub_download
                    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '.'))
                    models_dir = os.path.join(project_root, 'models')
                    os.makedirs(models_dir, exist_ok=True)
                    out = hf_hub_download(repo_id=hf_repo, filename=hf_filename, cache_dir=models_dir, token=hf_token or None, repo_type='model')
                    st.success(f'Downloaded model to: {out}')
                    st.info('Model saved to ./models/. You can now initialize the pipeline to use the local LLM.')
                except Exception as e:
                    st.error(f'Model download failed: {e}')

    # Show chunk count if retriever available
    try:
        count = st.session_state.pipeline.retriever.chunk_count() if st.session_state.pipeline else 0
    except Exception:
        # Fallback: try to read metadata file
        try:
            meta_path = os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__), '.')), 'vectordb', 'metadata.json')
            if os.path.exists(meta_path):
                with open(meta_path, 'r', encoding='utf-8') as f:
                    meta = json.load(f)
                count = len(meta)
            else:
                count = 0
        except Exception:
            count = 0
    st.write(f'Indexed chunks: **{count}**')

    if st.button('Clear Chat'):
        st.session_state.messages = []

# Render chat history
for msg in st.session_state.messages:
    role = msg.get('role', 'user')
    content = msg.get('content', '')
    with st.chat_message(role):
        st.write(content)


def stream_with_fallback(token_generator):
    """Try to use st.write_stream if available, otherwise fallback to incremental updates."""
    # Prefer built-in streaming if present
    if hasattr(st, 'write_stream'):
        try:
            # st.write_stream expects an iterable of strings
            st.write_stream(token_generator)
            return ''.join([])  # write_stream handles display; return empty placeholder
        except Exception:
            pass

    # Fallback: accumulate tokens and update a placeholder in a chat message
    response_text = ''
    placeholder = st.empty()
    for token in token_generator:
        response_text += token
        placeholder.write(response_text)
    return response_text


# Chat input
query = st.chat_input('Ask a question')
if query:
    # Show user message immediately
    st.session_state.messages.append({'role': 'user', 'content': query})
    with st.chat_message('user'):
        st.write(query)

    if not st.session_state.pipeline:
        with st.chat_message('assistant'):
            st.write('Pipeline not available. Check sidebar for errors.')
    else:
        # Ask pipeline for token generator and source chunks
        try:
            token_gen, source_chunks = st.session_state.pipeline.ask(query)
        except Exception as e:
            with st.chat_message('assistant'):
                st.write(f'Error during retrieval/generation: {e}')
            source_chunks = []
            token_gen = []

        # Stream assistant response
        with st.chat_message('assistant'):
            # Try using st.write_stream, otherwise fallback
            full_response = stream_with_fallback(token_gen)

        # If fallback returned the full text, use it; otherwise try to collect from token_gen
        if not full_response:
            # If write_stream handled display, we still want to record full text by joining tokens
            try:
                # token_gen may be exhausted; attempt to join if it's a list
                full_response = ''.join(list(token_gen))
            except Exception:
                full_response = ''

        # Save assistant message to session history
        st.session_state.messages.append({'role': 'assistant', 'content': full_response})

        # Show sources used
        with st.expander('Sources used'):
            if not source_chunks:
                st.write('No sources returned.')
            else:
                for i, src in enumerate(source_chunks, start=1):
                    # Handle both dict and string formats for backward compatibility
                    if isinstance(src, dict):
                        doc = src.get('document', 'unknown')
                        text = src.get('text', '')
                        st.write(f'{i}. **From {doc}**: {text}')
                    else:
                        st.write(f'{i}. {src}')
