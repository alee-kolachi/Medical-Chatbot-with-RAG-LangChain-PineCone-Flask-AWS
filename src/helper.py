from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from prompt import system_msg
from dotenv import load_dotenv
load_dotenv()

#Extract text from pdf file
def load_pdf_files(data_path):
    loader = DirectoryLoader(
        data_path,
        glob="*.pdf",
        loader_cls=PyPDFLoader
    )

    documents = loader.load()
    return documents


from typing import List
from langchain.schema import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

def filter_to_minimal_docs(docs: List[Document]) -> List[Document]:
    minimal_docs: List[Document] = []
    for doc in docs:
        src = doc.metadata.get("source")
        minimal_docs.append(
            Document (
                page_content=doc. page_content,
                metadata={"source": src}
            )
        )
    return minimal_docs

#split document in smaller chunk
def text_split(minimal_docs):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=20
    )

    text_chunk = text_splitter.split_documents(minimal_docs)
    return text_chunk


from langchain_huggingface import HuggingFaceEmbeddings

def download_embeddings():
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    embeddings = HuggingFaceEmbeddings(
        model_name=model_name
    )
    return embeddings



import os
from groq import Groq

# --- Optional: token counting (preferred) ---
# Try to use tiktoken for accurate token-based truncation; fallback to char truncation.
try:
    import tiktoken

    def count_tokens(text, model_name=None):
        # Try to use model-specific encoding if available, otherwise fall back
        try:
            if model_name:
                enc = tiktoken.encoding_for_model(model_name)
            else:
                enc = tiktoken.get_encoding("cl100k_base")
        except Exception:
            enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(text))

except Exception:
    # fallback if tiktoken not installed
    def count_tokens(text, model_name=None):
        return len(text) // 4  # rough average (4 chars ≈ 1 token)


# --- Initialize Groq client ---
client = Groq(api_key=os.getenv("GROQ_API_KEY"))
model_name = os.getenv("MODEL_NAME") or "llama3-8b-8192"

# --- Example: build robust extractor for your retrieved documents ---
def get_doc_text(doc):
    """
    Accepts a LangChain/Pinecone-style Document or dict and returns plain text.
    """
    # common attributes for LangChain Document
    for attr in ("page_content", "content", "text", "body"):
        if hasattr(doc, attr):
            value = getattr(doc, attr)
            if isinstance(value, str) and value.strip():
                return value.strip()
    # dict-like
    if isinstance(doc, dict):
        for key in ("page_content", "content", "text", "body"):
            if key in doc and isinstance(doc[key], str):
                return doc[key].strip()
    # Fall back
    return str(doc)

def get_doc_source(doc, idx):
    """
    Extract metadata source or id for citation. Falls back to index.
    """
    if hasattr(doc, "metadata") and isinstance(doc.metadata, dict):
        md = doc.metadata
        return md.get("source") or md.get("id") or md.get("filename") or md.get("title")
    if isinstance(doc, dict):
        md = doc.get("metadata") or {}
        return md.get("source") or md.get("id") or md.get("filename") or md.get("title")
    return f"doc_{idx+1}"

# --- Build context with token-aware truncation ---
def build_context(retrieved_docs, query, max_tokens_for_context=4000, model_name_for_count=None):
    """
    Build a context string from top retrieved docs. Truncates older/longer docs
    so the total token count of context stays below max_tokens_for_context.
    """
    pieces = []
    total_tokens = 0
    # iterate and add until we hit token budget; prefer adding highest-scored docs first
    for i, doc in enumerate(retrieved_docs):
        text = get_doc_text(doc)
        src = get_doc_source(doc, i)
        header = f"[Document {i+1} | source={src}]"
        piece = f"{header}\n{text}\n"
        tcount = count_tokens(piece, model_name_for_count)
        # if adding this piece would exceed budget, try to truncate piece from the front
        if total_tokens + tcount > max_tokens_for_context:
            # allowed tokens left
            remaining = max(0, max_tokens_for_context - total_tokens)
            if remaining <= 10:
                break
            # crude truncation by characters if tiktoken not fine-grained; else approximate
            if tiktoken:
                enc = tiktoken.encoding_for_model(model_name_for_count) if model_name_for_count in tiktoken.list_models() else tiktoken.get_encoding("cl100k_base")
                tokens = enc.encode(piece)
                piece = enc.decode(tokens[-remaining:])  # keep last tokens
            else:
                # fallback: keep last (remaining * 4) characters
                keep_chars = remaining * 4
                piece = piece[-keep_chars:]
            tcount = count_tokens(piece, model_name_for_count)
            if tcount == 0:
                break
            pieces.append(piece)
            total_tokens += tcount
            break
        else:
            pieces.append(piece)
            total_tokens += tcount

    context = "\n\n---\n\n".join(pieces)
    return context

# --- Example usage with your Pinecone retriever ---
# Suppose you already have: retriever = doc_search.as_retriever(search_type="similarity", search_kwargs={"k":3})
# and used: retrieved_docs = retriever.invoke("What are the special treatments available?")
# We'll assume `retrieved_docs` is available here.

def rag_answer(retrieved_docs, user_question):
    # Build context: set token budget so model + prompt + context fit model limit.
    # Adjust max_tokens_for_context according to your model's context window.
    # E.g., for models with 16k tokens you might set 12000; for 8k models maybe 3500-4000.
    max_context_tokens = 4000
    context = build_context(retrieved_docs, user_question, max_tokens_for_context=max_context_tokens, model_name_for_count=model_name)

    system_msg = system_msg

    user_msg = (
        f"Context:\n{context}\n\n"
        f"Question: {user_question}\n\n"
        "Task: Provide a detailed and paraphrased answer in your own words using information from the context. "
        "Return a JSON object with keys: 'answer' and 'sources'."
        "'sources' (list of source labels you used). If you don't find the answer in the context, "
        "set 'answer' to \"I don't know\" and 'sources' to an empty list."
    )

    messages = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_msg}
    ]

    resp = client.chat.completions.create(
        model=model_name,
        messages=messages,
        # optional: control length of generated reply if API supports it (e.g., max_tokens param)
        # max_tokens=512
    )
    return resp.choices[0].message.content

