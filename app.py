# app.py
import os
import logging
import re
from collections import defaultdict
from threading import Lock

from flask import Flask, render_template, request, jsonify
from dotenv import load_dotenv

# your project-specific imports (adjust if your module paths differ)
from langchain_pinecone import PineconeVectorStore
from src.helper import download_embeddings, filter_to_minimal_docs, load_pdf_files, rag_answer, text_split

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
DATA_FOLDER = os.getenv("DATA_FOLDER")
INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "medical-chatbot")
PORT = int(os.getenv("PORT", 8000))

if not PINECONE_API_KEY:
    logger.error("PINECONE_API_KEY is not set. Exiting.")
    raise SystemExit("PINECONE_API_KEY is required")

if not DATA_FOLDER:
    logger.error("DATA_FOLDER is not set. Exiting.")
    raise SystemExit("DATA_FOLDER is required and should point to PDFs")

app = Flask(__name__, static_folder='static', template_folder='templates')

# ---- global placeholders and thread-safe init ----
_embedding = None
_doc_search = None
_retriever = None
_init_lock = Lock()

# ---- simple deterministic extraction helpers ----
NAME_PATTERNS = [
    r"(Dr\.?\s+[A-Z][A-Za-z\.\-]+(?:\s+[A-Z][A-Za-z\.\-]+)?)",
    r"(Mr\.?\s+[A-Z][A-Za-z\.\-]+(?:\s+[A-Z][A-Za-z\.\-]+)?)",
    r"(Ms\.?\s+[A-Z][A-Za-z\.\-]+(?:\s+[A-Z][A-Za-z\.\-]+)?)",
    r"([A-Z][A-Za-z]+\s+[A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+)?)"
]

TITLE_KEYWORDS = [
    "CENTRE FOR MEDICAL EDUCATION",
    "CENTRE FOR MEDICAL EDUCATION AND TECHNOLOGY",
    "CENTRE FOR MEDICAL EDUCATION & TECHNOLOGY",
    "K.L. WIG"
]

def extract_candidate_names(text):
    names = set()
    if not text:
        return names
    for pat in NAME_PATTERNS:
        for m in re.findall(pat, text):
            nm = m.strip().rstrip('.,;:')
            if len(nm) < 3:
                continue
            # avoid capturing short all-caps tokens
            if nm.isupper() and len(nm) <= 4:
                continue
            names.add(nm)
    return names

def center_mentioned(text, query_title=None):
    if not text:
        return False
    t = text.upper()
    for kw in TITLE_KEYWORDS:
        if kw.upper() in t:
            return True
    if query_title and query_title.upper() in t:
        return True
    return False

def postprocess_retrieval(docs, query_title):
    """
    Build deterministic answer + contacts + sources + confidence from docs.
    """
    found_names = set()
    source_map = defaultdict(list)
    sources_seen = []
    center_hits = set()

    for d in docs:
        # support both object and dict-like docs
        text = getattr(d, "page_content", None) or (d.get("page_content") if isinstance(d, dict) else "")
        meta = getattr(d, "metadata", None) or (d.get("metadata") if isinstance(d, dict) else {}) or {}
        src = meta.get("source") or meta.get("doc_id") or meta.get("source_path") or "unknown"
        if src not in sources_seen:
            sources_seen.append(src)

        names = extract_candidate_names(text)
        for n in names:
            found_names.add(n)
            source_map[src].append(n)

        if center_mentioned(text, query_title=query_title):
            center_hits.add(src)

    # sort contacts: honorifics first, then alphabetical
    def sort_key(n):
        return (0 if re.match(r"^(Dr|Mr|Ms|Mrs)\.?", n) else 1, n)
    contacts = sorted(found_names, key=sort_key)

    # simple confidence heuristic
    if contacts:
        overlap = any(source_map[s] for s in center_hits)
        confidence = "high" if overlap else "medium"
    else:
        confidence = "low"

    if contacts:
        answer = f"Contacts related to Centre for Medical Education and Technology: {', '.join(contacts)}."
        note = None
        if not center_hits:
            note = "Names were found in related documents, but no document explicitly links them to the exact center title."
        else:
            note = f"Center/title appears in document(s): {', '.join([s.split('/')[-1] for s in center_hits])}"
    else:
        answer = f"No contact person names for \"{query_title}\" were found in the retrieved documents."
        note = None

    prioritized = [s for s in sources_seen if source_map.get(s) or s in center_hits]
    if not prioritized:
        prioritized = sources_seen

    return {
        "answer": answer,
        "contacts": contacts,
        "sources": prioritized,
        "confidence": confidence,
        "note": note
    }

# ---- initialization: build embeddings, text chunks, and index ----
def initialize_resources():
    global _embedding, _doc_search, _retriever
    with _init_lock:
        if _doc_search is not None:
            return

        logger.info("Downloading embeddings...")
        _embedding = download_embeddings()

        logger.info("Loading PDFs from %s", DATA_FOLDER)
        extracted_data = load_pdf_files(DATA_FOLDER)
        minimal_docs = filter_to_minimal_docs(extracted_data)
        text_chunks = text_split(minimal_docs)

        logger.info("Building/connecting to Pinecone index '%s'...", INDEX_NAME)
        _doc_search = PineconeVectorStore.from_documents(
            documents=text_chunks,
            embedding=_embedding,
            index_name=INDEX_NAME
        )

        try:
            _retriever = _doc_search.as_retriever(search_type="similarity", search_kwargs={"k": 5})
        except Exception:
            _retriever = _doc_search.as_retriever()

        logger.info("Initialization done.")

# ---- routes ----
@app.route("/")
def index():
    # serve the HTML template (ensure templates/chat.html exists)
    return render_template("chat.html")

@app.route("/get", methods=["POST"])
def chat():
    """
    POST /get
    Request JSON: { "msg": "<user query>" }
    Response JSON:
    {
      "answer": "<concise human-readable string>",   # what should be shown in the chat bubble
      "contacts": [...],                             # list of names (if any)
      "sources": [...],                              # for debug / details toggle (not shown by default)
      "confidence": "low|medium|high",
      "note": "...",                                 # optional
      "explanation": "..."                           # LLM verbose explanation (developer only)
    }
    """
    data = request.get_json(silent=True)
    if not data or "msg" not in data:
        return jsonify({"error": "Missing 'msg' parameter"}), 400

    msg = (data.get("msg") or "").strip()
    if not msg:
        return jsonify({"error": "'msg' must be non-empty"}), 400

    # Ensure resources initialized
    if _retriever is None:
        try:
            initialize_resources()
        except Exception:
            logger.exception("Initialization failed")
            return jsonify({"error": "server initialization failed"}), 500

    logger.info("Chat request (len=%d): %s", len(msg), msg[:200])

    try:
        # Retrieve relevant documents (adapt if your retriever API differs)
        if hasattr(_retriever, "get_relevant_documents"):
            docs = _retriever.get_relevant_documents(msg)
        elif hasattr(_retriever, "retrieve"):
            docs = _retriever.retrieve(msg)
        else:
            docs = _retriever(msg)

        # Deterministic extraction (must be implemented in postprocess_retrieval)
        processed = postprocess_retrieval(docs, msg)
        # processed contains: answer (concise string), contacts (list), sources (list), confidence, note

        # Run the LLM/RAG explanation (optional) but treat it as explanation only.
        explanation = None
        try:
            explanation = rag_answer(docs, msg)
            if explanation is not None and not isinstance(explanation, str):
                explanation = str(explanation)
        except Exception:
            # Do not fail if rag_answer errors
            logger.debug("rag_answer failed, continuing without explanation", exc_info=True)
            explanation = None

        # Choose final_answer:
        # - If extractor found contacts -> use processed['answer'] (deterministic authoritative)
        # - Else if LLM produced something -> use LLM explanation (trimmed)
        # - Else fallback to processed['answer'] (which will be a "not found" message)
        if processed.get("contacts"):
            final_answer = processed["answer"]
        else:
            if explanation and isinstance(explanation, str) and explanation.strip():
                final_answer = explanation.strip()
            else:
                final_answer = processed["answer"]

        # IMPORTANT: ensure final_answer NEVER contains raw file paths. processed['answer']
        # should be a concise human string (postprocess_retrieval should not include paths).
        # If your postprocess currently builds answers with the full path, update it to use filenames only.

        response = {
            "answer": final_answer,
            "contacts": processed.get("contacts", []),
            "sources": processed.get("sources", []),    # kept for the details toggle only
            "confidence": processed.get("confidence", "low"),
            "note": processed.get("note"),
            "explanation": explanation
        }

        logger.debug("Outgoing response: answer=%s contacts=%s", (response["answer"][:200] + '...') if response["answer"] else "", response["contacts"])
        return jsonify(response)

    except Exception:
        logger.exception("Error during retrieval/answering")
        return jsonify({"error": "internal error"}), 500



# ---- run ----
if __name__ == "__main__":
    # initialize at process start (helpful for dev); in production manage worker initialization appropriately
    try:
        initialize_resources()
    except Exception:
        logger.exception("Initialization failed at startup; continuing and will try on first request")

    app.run(host="0.0.0.0", port=PORT, debug=False)
