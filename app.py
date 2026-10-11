# =============================================================================
# PDF RAG Lab  –  app.py
# =============================================================================
# A beginner-friendly Retrieval-Augmented Generation demo for AI students.
#
# RAG Pipeline
# ────────────
#   PDF Upload → Text Extraction → Chunking → TF-IDF Indexing
#   → Cosine-Similarity Retrieval → Optional Free-Tier LLM Answer
#
# Strictly free:
#   • All retrieval is local (scikit-learn TF-IDF + cosine similarity).
#   • Optional LLM: Google Gemini API via "gemini-1.5-flash" — genuinely
#     free tier as of mid-2025 (15 req/min, 1 500 req/day, no card needed).
#     Always check https://ai.google.dev/pricing for current limits.
#   • Zero paid services, zero paid libraries, zero paid hosting required.
#
# Security
# ────────
#   API keys are read from st.secrets / environment variables ONLY.
#   No key is ever written into this source file.
# =============================================================================

from __future__ import annotations

import hashlib
import io
import os
import re
import textwrap
import zipfile
from typing import Any

import streamlit as st
from pypdf import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ── Try importing the Gemini SDK; keep optional ───────────────────────────────
try:
    import google.generativeai as genai  # type: ignore
    _GEMINI_AVAILABLE = True
except ImportError:
    _GEMINI_AVAILABLE = False

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────
APP_TITLE        = "PDF RAG Lab"
APP_SUBTITLE     = "Upload a document, build a knowledge base, and explore how Retrieval-Augmented Generation works."
GEMINI_MODEL     = "gemini-3.5-flash"   # Free tier model; ultra-fast & high quality
CANDIDATE_MODELS = [
    "gemini-3.5-flash",
    "gemini-3.8-flash",
    "gemini-flash-latest",
    "gemini-2.5-flash-lite",
]
MAX_PDF_MB       = 20
DEFAULT_CHUNK_CHARS = 900
DEFAULT_OVERLAP_CHARS = 150
DEFAULT_TOP_K    = 5
MAX_CONTEXT_CHARS = 10_000              # Characters sent to the LLM
MAX_LLM_TOKENS   = 1536                 # Extended output limit for rich answers

# ─────────────────────────────────────────────────────────────────────────────
# Page config  (must be the very first Streamlit call)
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title=APP_TITLE,
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# Lightweight CSS – clean, bright, professional
# ─────────────────────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    /* ── Global ── */
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    /* ── App header ── */
    .app-header {
        background: linear-gradient(120deg, #1e3a5f 0%, #2563eb 100%);
        border-radius: 14px;
        padding: 1.6rem 2rem;
        margin-bottom: 1.2rem;
    }
    .app-header h1 { color: #ffffff; font-size: 2rem; font-weight: 700; margin: 0; }
    .app-header p  { color: #bfdbfe; font-size: 0.95rem; margin: 0.35rem 0 0; }
    .app-header .badge {
        display: inline-block;
        background: rgba(255,255,255,0.2);
        color: #e0f2fe;
        border-radius: 20px;
        font-size: 0.72rem;
        font-weight: 600;
        padding: 0.15rem 0.7rem;
        margin-top: 0.5rem;
        letter-spacing: 0.05em;
    }

    /* ── Pipeline step bar ── */
    .pipeline-bar {
        display: flex;
        align-items: center;
        gap: 0;
        background: #f0f7ff;
        border: 1px solid #bfdbfe;
        border-radius: 10px;
        padding: 0.6rem 1rem;
        margin: 0.8rem 0 1.2rem;
        flex-wrap: wrap;
        gap: 0.3rem;
    }
    .pipe-step {
        background: #2563eb;
        color: white;
        border-radius: 6px;
        padding: 0.25rem 0.7rem;
        font-size: 0.75rem;
        font-weight: 600;
        white-space: nowrap;
    }
    .pipe-arrow { color: #93c5fd; font-weight: 700; font-size: 1rem; margin: 0 0.2rem; }

    /* ── Info cards ── */
    .info-card {
        background: #f8faff;
        border: 1px solid #dbeafe;
        border-left: 4px solid #2563eb;
        border-radius: 8px;
        padding: 0.8rem 1rem;
        font-size: 0.85rem;
        color: #1e3a5f;
        margin: 0.5rem 0;
    }
    .info-card strong { color: #1d4ed8; }

    /* ── Chunk / passage card ── */
    .chunk-card {
        background: #f0f9ff;
        border: 1px solid #bae6fd;
        border-left: 4px solid #0ea5e9;
        border-radius: 8px;
        padding: 0.9rem 1.1rem;
        margin-bottom: 0.8rem;
        font-size: 0.87rem;
        color: #0c4a6e;
    }
    .chunk-meta {
        font-size: 0.78rem;
        font-weight: 600;
        color: #0369a1;
        margin-bottom: 0.4rem;
    }

    /* ── Answer box ── */
    .answer-box {
        background: #f0fdf4;
        border: 1px solid #bbf7d0;
        border-left: 4px solid #16a34a;
        border-radius: 8px;
        padding: 1.1rem 1.3rem;
        color: #14532d;
        line-height: 1.7;
        font-size: 0.92rem;
        margin-top: 0.6rem;
    }

    /* ── Retrieval-only answer ── */
    .retrieval-box {
        background: #fffbeb;
        border: 1px solid #fde68a;
        border-left: 4px solid #f59e0b;
        border-radius: 8px;
        padding: 1rem 1.2rem;
        color: #78350f;
        font-size: 0.88rem;
        margin-top: 0.6rem;
    }

    /* ── Warning / note ── */
    .note-box {
        background: #fff7ed;
        border: 1px solid #fed7aa;
        border-radius: 8px;
        padding: 0.7rem 1rem;
        font-size: 0.82rem;
        color: #7c2d12;
        margin: 0.5rem 0;
    }

    /* ── Section label ── */
    .section-label {
        font-size: 0.75rem;
        font-weight: 700;
        letter-spacing: 0.08em;
        color: #6b7280;
        text-transform: uppercase;
        margin-bottom: 0.3rem;
    }

    /* ── Sidebar tweaks ── */
    section[data-testid="stSidebar"] { background: #f8faff !important; }
    section[data-testid="stSidebar"] .stMarkdown p { font-size: 0.82rem; color: #374151; }
    </style>
    """,
    unsafe_allow_html=True,
)

def read_gemini_key() -> str | None:
    """Read the Gemini API key from session state override, environment variable, or st.secrets."""
    # 1. Custom UI override
    ui_key = str(st.session_state.get("custom_gemini_key", "")).strip() if "custom_gemini_key" in st.session_state else ""
    if ui_key:
        return ui_key
    # 2. Environment variable
    env_key = os.getenv("GEMINI_API_KEY", "").strip()
    if env_key:
        return env_key
    # 3. Streamlit secrets (.streamlit/secrets.toml)
    try:
        sec_key = str(st.secrets.get("GEMINI_API_KEY", "")).strip()
        if sec_key:
            return sec_key
    except Exception:
        pass
    return None


def _init_session() -> None:
    has_key = bool(read_gemini_key())
    defaults: dict[str, Any] = {
        "pdf_bytes":    None,
        "pdf_name":     None,
        "pdf_hash":     None,
        "pages":        [],       # list[{"page": int, "text": str}]
        "chunks":       [],       # list[{"page": int, "text": str}]
        "vectorizer":   None,
        "chunk_vectors": None,
        "kb_ready":     False,
        "chat_history": [],       # list[{"question": str, "answer": str, "retrieved": list, "mode": str}]
        "chunk_size":   DEFAULT_CHUNK_CHARS,
        "chunk_overlap": DEFAULT_OVERLAP_CHARS,
        "top_k":        DEFAULT_TOP_K,
        "answer_mode":  "Gemini LLM (free tier, key required)" if has_key else "Retrieval only (no API key needed)",
        "custom_gemini_key": "",
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val

_init_session()

# ─────────────────────────────────────────────────────────────────────────────
# Pure-Python utilities
# ─────────────────────────────────────────────────────────────────────────────

def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def clean_text(raw: str) -> str:
    """Collapse excessive whitespace while preserving paragraph breaks."""
    # Normalise line endings
    text = raw.replace("\r\n", "\n").replace("\r", "\n")
    # Collapse 3+ consecutive newlines to two (paragraph break)
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Collapse spaces / tabs within a line
    text = re.sub(r"[^\S\n]+", " ", text)
    return text.strip()


# ─────────────────────────────────────────────────────────────────────────────
# Stage A + B: PDF upload & text extraction
# ─────────────────────────────────────────────────────────────────────────────

def extract_pdf_pages(pdf_bytes: bytes) -> list[dict[str, Any]]:
    """
    Extract text from each page of a PDF.

    Returns a list of dicts: {"page": <1-based int>, "text": <cleaned str>}.
    Raises ValueError with a user-friendly message on failure.
    """
    try:
        reader = PdfReader(io.BytesIO(pdf_bytes))
    except Exception as exc:
        raise ValueError(
            f"Could not open the PDF. It may be corrupted or in an unsupported format. "
            f"Technical detail: {exc}"
        ) from exc

    if reader.is_encrypted:
        raise ValueError(
            "This PDF is password-protected. Remove the password before uploading."
        )

    pages: list[dict[str, Any]] = []
    for page_num, page in enumerate(reader.pages, start=1):
        try:
            raw = page.extract_text() or ""
        except Exception:
            raw = ""
        text = clean_text(raw)
        if text:
            pages.append({"page": page_num, "text": text})

    if not pages:
        raise ValueError(
            "No extractable text was found. This PDF is likely scanned (image-based). "
            "Run OCR software on it first, then upload the searchable version."
        )

    return pages


# ─────────────────────────────────────────────────────────────────────────────
# Stage C: Chunking
# ─────────────────────────────────────────────────────────────────────────────

def chunk_pages(
    pages: list[dict[str, Any]],
    chunk_chars: int = DEFAULT_CHUNK_CHARS,
    overlap_chars: int = DEFAULT_OVERLAP_CHARS,
) -> list[dict[str, Any]]:
    """
    Split each page's text into overlapping character-based chunks.

    Each chunk carries the originating page number so we can cite sources.
    Overlapping chunks prevent important sentences from being split across
    two non-adjacent chunks.
    """
    if chunk_chars < 50:
        chunk_chars = 50
    overlap_chars = min(overlap_chars, chunk_chars // 2)
    step = max(1, chunk_chars - overlap_chars)

    chunks: list[dict[str, Any]] = []
    for page in pages:
        text = page["text"]
        if not text:
            continue
        start = 0
        while start < len(text):
            end = start + chunk_chars
            piece = text[start:end].strip()
            if piece:
                chunks.append({"page": page["page"], "text": piece})
            if end >= len(text):
                break
            start += step

    return chunks


# ─────────────────────────────────────────────────────────────────────────────
# Stage D: Build knowledge base (TF-IDF index)
# ─────────────────────────────────────────────────────────────────────────────

def build_tfidf_index(
    chunks: list[dict[str, Any]],
) -> tuple[TfidfVectorizer, Any]:
    """
    Fit a TF-IDF vectorizer on the chunk texts and return the
    (vectorizer, sparse_matrix) pair.

    This is NOT training an LLM. It is fitting a classical information-
    retrieval index. Students can later swap this for semantic embeddings.
    """
    texts = [c["text"] for c in chunks]
    vectorizer = TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        ngram_range=(1, 2),
        max_features=25_000,
        sublinear_tf=True,
    )
    try:
        chunk_vectors = vectorizer.fit_transform(texts)
    except ValueError as exc:
        raise ValueError(
            "Could not build the index — the document may contain too little text. "
            f"Detail: {exc}"
        ) from exc
    return vectorizer, chunk_vectors


# ─────────────────────────────────────────────────────────────────────────────
# Stage E: Retrieval
# ─────────────────────────────────────────────────────────────────────────────

def retrieve(
    question: str,
    vectorizer: TfidfVectorizer,
    chunk_vectors: Any,
    chunks: list[dict[str, Any]],
    top_k: int,
) -> list[dict[str, Any]]:
    """
    Return the top_k most relevant chunks for the question.

    Uses cosine similarity between the TF-IDF query vector and each
    chunk vector. TF-IDF measures lexical (word) overlap — it is not
    the same as semantic similarity from neural embeddings.
    """
    q_vec = vectorizer.transform([question])
    scores = cosine_similarity(q_vec, chunk_vectors).ravel()
    ranked_idx = scores.argsort()[::-1][:top_k]

    results: list[dict[str, Any]] = []
    for idx in ranked_idx:
        score = float(scores[idx])
        if score <= 0.0:
            continue
        item = chunks[int(idx)].copy()
        item["score"] = score
        results.append(item)
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Stage F – Mode 2: Free-tier LLM answer (Google Gemini)
# ─────────────────────────────────────────────────────────────────────────────


def generate_llm_answer(
    question: str,
    retrieved: list[dict[str, Any]],
    api_key: str,
) -> tuple[str, str]:
    """
    Send the retrieved context + question to Google Gemini (free tier).
    Tries candidate models in order of performance and availability.
    Returns (answer_text, model_name_used).
    """
    if not _GEMINI_AVAILABLE:
        raise RuntimeError(
            "The 'google-generativeai' package is not installed. "
            "Add it to requirements.txt and redeploy."
        )

    # Build an enriched context string from top retrieved chunks
    context_parts: list[str] = []
    total_chars = 0
    for item in retrieved:
        score_info = f" | Relevance: {item.get('score', 0):.3f}" if 'score' in item else ""
        snippet = f"--- [Document Excerpt · Page {item['page']}{score_info}] ---\n{item['text']}"
        if total_chars + len(snippet) > MAX_CONTEXT_CHARS:
            break
        context_parts.append(snippet)
        total_chars += len(snippet)
    context = "\n\n".join(context_parts)

    system_prompt = (
        "You are an expert AI teaching assistant and tutor for a Retrieval-Augmented Generation (RAG) educational lab.\n"
        "Your task is to provide clear, high-quality, comprehensive, and well-structured answers to the student's question.\n\n"
        "STRICT GROUNDING & QUALITY RULES:\n"
        "1. GROUNDING: Answer using ONLY the information provided in the retrieved PDF excerpts below. "
        "Do not invent facts or assume information not present in the text.\n"
        "2. CITATIONS: Attribute key facts to their source page using clear brackets, e.g., [Page 2] or [Pages 2, 4].\n"
        "3. FORMATTING: Structure your response cleanly using Markdown formatting with headers (##, ###), bullet points, bold key terms, and summaries.\n"
        "4. COMPLETENESS: Provide thorough, pedagogical explanations so students learn both the core answer and the relevant background from the text.\n"
        "5. GAPS: If the excerpts only partially answer the question or lack relevant details, explain what the document mentions and clearly indicate what information is not covered."
    )
    user_message = (
        f"Student Question: {question}\n\n"
        f"Retrieved Document Excerpts:\n{context}\n\n"
        f"Instructions: Synthesize a structured, accurate, well-explained response citing the relevant page numbers."
    )

    genai.configure(api_key=api_key)

    last_err: Exception | None = None
    for model_name in CANDIDATE_MODELS:
        try:
            model = genai.GenerativeModel(
                model_name=model_name,
                system_instruction=system_prompt,
                generation_config=genai.types.GenerationConfig(
                    max_output_tokens=MAX_LLM_TOKENS,
                    temperature=0.25,
                ),
            )
            response = model.generate_content(
                user_message,
                request_options={"timeout": 35},
            )
            text = getattr(response, "text", None) or ""
            if text.strip():
                return text.strip(), model_name
        except Exception as exc:
            last_err = exc
            continue

    if last_err:
        raise last_err
    raise RuntimeError("The model returned an empty response.")


# ─────────────────────────────────────────────────────────────────────────────
# Downloadable ZIP helper
# ─────────────────────────────────────────────────────────────────────────────

def make_project_zip() -> bytes:
    """
    Bundle app.py and requirements.txt into an in-memory ZIP.
    Called lazily so it does not re-read the file on every render.
    """
    try:
        app_code = open(__file__, "rb").read()
    except Exception:
        app_code = b"# Could not read app.py"

    req_content = (
        "pypdf>=4.0,<6.0\n"
        "scikit-learn>=1.4,<2.0\n"
        "google-generativeai>=0.7,<1.0\n"
    ).encode()

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("pdf-rag-lab/app.py", app_code)
        zf.writestr("pdf-rag-lab/requirements.txt", req_content)
        zf.writestr(
            "pdf-rag-lab/.streamlit/secrets.toml",
            b"# Paste your Gemini API key (from https://aistudio.google.com/app/apikey)\n"
            b'GEMINI_API_KEY = "AIza..."\n',
        )
        zf.writestr(
            "pdf-rag-lab/.gitignore",
            b".streamlit/secrets.toml\n__pycache__/\n*.pyc\n.env\n",
        )
    buf.seek(0)
    return buf.read()


# ─────────────────────────────────────────────────────────────────────────────
# Helper: hash the settings that affect the index
# ─────────────────────────────────────────────────────────────────────────────

def _settings_sig(chunk_size: int, chunk_overlap: int) -> str:
    return f"{chunk_size}_{chunk_overlap}"


# ─────────────────────────────────────────────────────────────────────────────
# ██████████████████████  U  I  ████████████████████████████████████████████
# ─────────────────────────────────────────────────────────────────────────────

# ── App header ────────────────────────────────────────────────────────────────
st.markdown(
    f"""
    <div class="app-header">
        <h1>📚 {APP_TITLE}</h1>
        <p>{APP_SUBTITLE}</p>
        <span class="badge">🎓 AI LEARNING LAB</span>
    </div>
    """,
    unsafe_allow_html=True,
)

# ── RAG pipeline overview bar ────────────────────────────────────────────────
st.markdown(
    """
    <div class="pipeline-bar">
        <span class="pipe-step">① Upload PDF</span>
        <span class="pipe-arrow">→</span>
        <span class="pipe-step">② Extract Text</span>
        <span class="pipe-arrow">→</span>
        <span class="pipe-step">③ Chunk Text</span>
        <span class="pipe-arrow">→</span>
        <span class="pipe-step">④ TF-IDF Index</span>
        <span class="pipe-arrow">→</span>
        <span class="pipe-step">⑤ Retrieval</span>
        <span class="pipe-arrow">→</span>
        <span class="pipe-step">⑥ Answer</span>
    </div>
    """,
    unsafe_allow_html=True,
)

# ─────────────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Settings")

    # ── PDF upload ────────────────────────────────────────────────────────────
    st.markdown("### 📄 Upload PDF")
    uploaded_file = st.file_uploader(
        "Choose a text-based PDF (max 20 MB)",
        type=["pdf"],
        label_visibility="collapsed",
    )
    st.caption("ℹ️ Only text-based PDFs work. Scanned (image) PDFs need OCR first.")

    st.markdown("---")

    # ── Chunking settings ─────────────────────────────────────────────────────
    st.markdown("### ✂️ Chunking")
    chunk_size = st.slider(
        "Chunk size (characters)",
        min_value=200,
        max_value=2000,
        value=st.session_state.chunk_size,
        step=50,
        help="How many characters each text chunk contains. Larger chunks give more context; smaller chunks allow finer retrieval.",
    )
    chunk_overlap = st.slider(
        "Overlap (characters)",
        min_value=0,
        max_value=400,
        value=st.session_state.chunk_overlap,
        step=20,
        help="How many characters two consecutive chunks share. Overlap prevents important information from being cut off at a boundary.",
    )
    with st.expander("💡 What are chunks?"):
        st.markdown(
            "Long documents are split into **chunks** so that only the most "
            "relevant pieces are passed to the language model — not the entire PDF. "
            "**Overlap** ensures a sentence is never split across two non-adjacent chunks."
        )

    st.markdown("---")

    # ── Retrieval settings ────────────────────────────────────────────────────
    st.markdown("### 🔍 Retrieval")
    top_k = st.slider(
        "Chunks to retrieve",
        min_value=1,
        max_value=10,
        value=st.session_state.top_k,
        help="How many of the most relevant passages are fetched for each question.",
    )

    st.markdown("---")

    # ── Answer mode ───────────────────────────────────────────────────────────
    st.markdown("### 🤖 Answer Mode")
    active_key = read_gemini_key()
    mode_options = [
        "Retrieval only (no API key needed)",
        "Gemini LLM (free tier, key required)",
    ]
    cur_mode = st.session_state.get("answer_mode", mode_options[1] if active_key else mode_options[0])
    selected_idx = mode_options.index(cur_mode) if cur_mode in mode_options else (1 if active_key else 0)

    answer_mode = st.radio(
        "Select mode",
        options=mode_options,
        index=selected_idx,
        help="Retrieval mode shows the most relevant passages. LLM mode uses Google Gemini to synthesise an answer.",
        label_visibility="collapsed",
    )
    st.session_state.answer_mode = answer_mode

    if answer_mode == "Gemini LLM (free tier, key required)":
        gem_key = read_gemini_key()
        if gem_key:
            st.success("✅ GEMINI_API_KEY detected & active")
            st.caption(f"Model: `{GEMINI_MODEL}` (Free tier)")
        else:
            st.warning(
                "No `GEMINI_API_KEY` found in `.streamlit/secrets.toml`.\n\n"
                "Get a **free** key at [aistudio.google.com/app/apikey]"
                "(https://aistudio.google.com/app/apikey) or paste it below."
            )

        # Allow quick in-browser API key override/input
        user_override = st.text_input(
            "Gemini API Key (direct input / override)",
            value=st.session_state.get("custom_gemini_key", ""),
            type="password",
            help="Optional: paste your key here if you didn't edit secrets.toml",
        )
        if user_override != st.session_state.get("custom_gemini_key", ""):
            st.session_state["custom_gemini_key"] = user_override
            st.rerun()

        if not _GEMINI_AVAILABLE:
            st.error(
                "`google-generativeai` is not installed. "
                "Make sure it appears in `requirements.txt`."
            )
        st.caption(
            "ℹ️ Free tier: ~15 req/min, ~1 500 req/day (no credit card). "
            "Check [ai.google.dev/pricing](https://ai.google.dev/pricing) "
            "for current limits."
        )

    st.markdown("---")

    # ── Reset ─────────────────────────────────────────────────────────────────
    if st.button("🗑️ Reset / Clear Everything", use_container_width=True):
        keep = {"chunk_size", "chunk_overlap", "top_k", "answer_mode"}
        for k in list(st.session_state.keys()):
            if k not in keep:
                del st.session_state[k]
        _init_session()
        st.rerun()

    st.markdown("---")

    # ── Download project ZIP ──────────────────────────────────────────────────
    st.markdown("### 📥 Download Project")
    st.download_button(
        label="⬇️ Download app.py + requirements.txt (ZIP)",
        data=make_project_zip(),
        file_name="pdf_rag_lab.zip",
        mime="application/zip",
        use_container_width=True,
    )
    st.caption("Contains both project files ready for GitHub and Streamlit Cloud.")

    # Sync slider values to session state
    st.session_state.chunk_size    = chunk_size
    st.session_state.chunk_overlap = chunk_overlap
    st.session_state.top_k        = top_k
    st.session_state.answer_mode  = answer_mode

# ─────────────────────────────────────────────────────────────────────────────
# PDF Processing – triggered whenever a new file is uploaded
# ─────────────────────────────────────────────────────────────────────────────
if uploaded_file is not None:
    raw_bytes = uploaded_file.getvalue()

    # ── File-size guard ───────────────────────────────────────────────────────
    if len(raw_bytes) > MAX_PDF_MB * 1024 * 1024:
        st.error(
            f"❌ File is too large ({len(raw_bytes)/1024/1024:.1f} MB). "
            f"Please upload a PDF smaller than {MAX_PDF_MB} MB."
        )
        uploaded_file = None
    else:
        current_hash = sha256_hex(raw_bytes)
        current_sig  = _settings_sig(chunk_size, chunk_overlap)

        # Detect file or settings change → invalidate index
        if (
            st.session_state.pdf_hash != current_hash
            or st.session_state.get("_settings_sig") != current_sig
        ):
            st.session_state.pdf_bytes    = raw_bytes
            st.session_state.pdf_name     = uploaded_file.name
            st.session_state.pdf_hash     = current_hash
            st.session_state["_settings_sig"] = current_sig
            st.session_state.pages        = []
            st.session_state.chunks       = []
            st.session_state.vectorizer   = None
            st.session_state.chunk_vectors = None
            st.session_state.kb_ready     = False
            st.session_state.chat_history = []

            # Extract text immediately on upload
            with st.spinner("Extracting text from PDF…"):
                try:
                    pages = extract_pdf_pages(raw_bytes)
                    st.session_state.pages = pages
                except ValueError as err:
                    st.error(f"❌ {err}")
                    st.stop()

# ─────────────────────────────────────────────────────────────────────────────
# Main tabs
# ─────────────────────────────────────────────────────────────────────────────
tab_doc, tab_kb, tab_qa, tab_inspect, tab_pipeline = st.tabs([
    "📄 Document",
    "🗂️ Knowledge Base",
    "💬 Ask Questions",
    "🔎 Retrieval Inspector",
    "🔬 RAG Pipeline",
])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — Document
# ══════════════════════════════════════════════════════════════════════════════
with tab_doc:
    if not st.session_state.pages:
        st.info(
            "👈 Upload a PDF using the sidebar to get started. "
            "Only text-based PDFs are supported — scanned documents need OCR preprocessing."
        )
        st.markdown(
            """
            <div class="info-card">
            <strong>What is RAG?</strong><br>
            Retrieval-Augmented Generation (RAG) is a technique that enhances an AI model's
            answers by first <em>retrieving</em> relevant passages from a document and then
            using those passages as context for <em>generating</em> a grounded response.
            RAG does <strong>not</strong> train or fine-tune a language model on your document.
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        pages = st.session_state.pages
        total_chars = sum(len(p["text"]) for p in pages)

        col1, col2, col3 = st.columns(3)
        col1.metric("📄 File", st.session_state.pdf_name or "—")
        col2.metric("📃 Pages with text", len(pages))
        col3.metric("🔤 Characters extracted", f"{total_chars:,}")

        st.markdown("---")
        st.markdown("#### 📖 Extracted Text Preview")
        with st.expander("Show first 3 pages of extracted text", expanded=False):
            for page in pages[:3]:
                st.markdown(f"**Page {page['page']}**")
                st.text(textwrap.shorten(page["text"], width=800, placeholder=" …"))
                st.divider()

        st.markdown(
            """
            <div class="info-card">
            <strong>Stage A → B complete.</strong>
            The PDF has been parsed page-by-page. Each page's text is cleaned and stored
            with its page number so answers can later cite their source pages.
            </div>
            """,
            unsafe_allow_html=True,
        )

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — Knowledge Base
# ══════════════════════════════════════════════════════════════════════════════
with tab_kb:
    if not st.session_state.pages:
        st.info("📄 Upload a PDF first (Stage B must complete before Stage D).")
    else:
        st.markdown("#### 🗂️ Build the Knowledge Base")
        st.markdown(
            """
            <div class="info-card">
            <strong>What happens here?</strong><br>
            The extracted text is split into overlapping chunks, then each chunk is
            converted into a numerical TF-IDF vector. This creates a fast, searchable
            index. <em>This is not LLM training</em> — it is classical information retrieval.
            </div>
            """,
            unsafe_allow_html=True,
        )

        col_btn, col_status = st.columns([1, 2])
        with col_btn:
            build_btn = st.button(
                "🔨 Build Knowledge Base",
                type="primary",
                use_container_width=True,
                disabled=st.session_state.kb_ready,
            )

        if build_btn:
            with st.spinner("Chunking text and building TF-IDF index…"):
                try:
                    chunks = chunk_pages(
                        st.session_state.pages,
                        chunk_chars=chunk_size,
                        overlap_chars=chunk_overlap,
                    )
                    if not chunks:
                        st.error("❌ No chunks were created. The document may be too short.")
                        st.stop()
                    vectorizer, chunk_vectors = build_tfidf_index(chunks)
                    st.session_state.chunks       = chunks
                    st.session_state.vectorizer   = vectorizer
                    st.session_state.chunk_vectors = chunk_vectors
                    st.session_state.kb_ready     = True
                except ValueError as err:
                    st.error(f"❌ {err}")

        if st.session_state.kb_ready:
            chunks = st.session_state.chunks
            c1, c2, c3 = st.columns(3)
            c1.metric("✂️ Chunks created", len(chunks))
            c2.metric("📏 Chunk size", f"{chunk_size} chars")
            c3.metric("🔁 Overlap", f"{chunk_overlap} chars")

            st.success(
                f"✅ Knowledge base is ready — {len(chunks)} chunks indexed. "
                "Go to **Ask Questions** to start querying."
            )

            with st.expander("🔬 Inspect a sample chunk", expanded=False):
                sample = chunks[len(chunks) // 2]
                st.markdown(
                    f'<div class="chunk-card">'
                    f'<div class="chunk-meta">Page {sample["page"]} · '
                    f'{len(sample["text"])} characters</div>'
                    f'{sample["text"]}'
                    f"</div>",
                    unsafe_allow_html=True,
                )

# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — Ask Questions
# ══════════════════════════════════════════════════════════════════════════════
with tab_qa:
    if not st.session_state.kb_ready:
        st.info("🗂️ Build the knowledge base first (see the **Knowledge Base** tab).")
    else:
        st.markdown("#### 💬 Ask a Question About Your Document")

        # ── Chat history ──────────────────────────────────────────────────────
        for entry in st.session_state.chat_history:
            with st.chat_message("user"):
                st.write(entry["question"])
            with st.chat_message("assistant"):
                mode_label = (
                    "📚 Retrieval mode"
                    if entry["mode"] == "retrieval"
                    else f"🤖 {entry['mode'].title()}"
                )
                st.caption(mode_label)
                if entry["mode"] == "retrieval":
                    st.markdown(
                        f'<div class="retrieval-box">{entry["answer"]}</div>',
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown(
                        f'<div class="answer-box">{entry["answer"]}</div>',
                        unsafe_allow_html=True,
                    )
                if entry["retrieved"]:
                    st.caption(
                        f"Based on {len(entry['retrieved'])} retrieved chunk(s) "
                        f"from pages: "
                        + ", ".join(str(r["page"]) for r in entry["retrieved"])
                    )

        # ── Question input ────────────────────────────────────────────────────
        question = st.chat_input(
            "Type your question here and press Enter…",
        )

        if question:
            q = question.strip()
            if not q:
                st.warning("Please enter a non-empty question.")
            else:
                with st.chat_message("user"):
                    st.write(q)

                with st.spinner("Retrieving relevant passages…"):
                    retrieved = retrieve(
                        q,
                        st.session_state.vectorizer,
                        st.session_state.chunk_vectors,
                        st.session_state.chunks,
                        top_k,
                    )

                if not retrieved:
                    answer_text = (
                        "No sufficiently relevant passages were found for this question. "
                        "Try rephrasing using terms that appear in the document."
                    )
                    mode_used = "retrieval"
                elif answer_mode == "Retrieval only (no API key needed)":
                    # Mode 1: Present the retrieved passages directly
                    lines = ["**Most relevant passages from the document:**\n"]
                    for i, item in enumerate(retrieved, 1):
                        short = textwrap.shorten(item["text"], width=300, placeholder="…")
                        lines.append(
                            f"**{i}. Page {item['page']}** (score: {item['score']:.3f})  \n"
                            f"{short}"
                        )
                    answer_text = "\n\n".join(lines)
                    mode_used = "retrieval"
                else:
                    # Mode 2: LLM generation
                    gem_key = read_gemini_key()
                    if not gem_key:
                        answer_text = (
                            "⚠️ No Gemini API key found. "
                            "Add `GEMINI_API_KEY` to `.streamlit/secrets.toml` or Streamlit Cloud Secrets."
                        )
                        mode_used = "retrieval"
                    elif not _GEMINI_AVAILABLE:
                        answer_text = (
                            "⚠️ `google-generativeai` is not installed. "
                            "Add it to `requirements.txt` and restart."
                        )
                        mode_used = "retrieval"
                    else:
                        try:
                            with st.spinner(
                                f"Generating comprehensive answer with Google Gemini (free tier)…"
                            ):
                                answer_text, model_used = generate_llm_answer(q, retrieved, gem_key)
                            mode_used = f"gemini ({model_used})"
                        except Exception as exc:
                            err_str = str(exc)
                            if "quota" in err_str.lower() or "429" in err_str:
                                answer_text = (
                                    "⚠️ Gemini free-tier rate limit reached. "
                                    "Wait a minute and try again, or switch to "
                                    "**Retrieval only** mode to continue without an API key. "
                                    f"Detail: {err_str}"
                                )
                            elif "api_key" in err_str.lower() or "401" in err_str or "403" in err_str:
                                answer_text = (
                                    "⚠️ Gemini API key is invalid or lacks permission. "
                                    "Check your key at [aistudio.google.com](https://aistudio.google.com/app/apikey). "
                                    f"Detail: {err_str}"
                                )
                            else:
                                answer_text = (
                                    f"⚠️ LLM call failed: {err_str}. "
                                    "Retrieval passages are still available in the **Retrieval Inspector** tab."
                                )
                            mode_used = "retrieval"

                # Store in history
                st.session_state.chat_history.append({
                    "question": q,
                    "answer": answer_text,
                    "retrieved": retrieved,
                    "mode": mode_used,
                })

                with st.chat_message("assistant"):
                    mode_label = (
                        "📚 Retrieval mode"
                        if mode_used == "retrieval"
                        else f"🤖 {mode_used.title()}"
                    )
                    st.caption(mode_label)
                    if mode_used == "retrieval":
                        st.markdown(
                            f'<div class="retrieval-box">{answer_text}</div>',
                            unsafe_allow_html=True,
                        )
                    else:
                        st.markdown(
                            f'<div class="answer-box">{answer_text}</div>',
                            unsafe_allow_html=True,
                        )
                    if retrieved:
                        st.caption(
                            f"Based on {len(retrieved)} retrieved chunk(s) "
                            f"from pages: "
                            + ", ".join(str(r["page"]) for r in retrieved)
                        )

                if answer_mode == "Gemini LLM (free tier, key required)":
                    st.markdown(
                        '<div class="note-box">⚠️ <strong>Note:</strong> '
                        "Selected passages from your PDF are sent to Google's servers. "
                        "This demo uses Gemini's free tier. Full documents are never sent — "
                        "only the top retrieved chunks. "
                        "Check <a href='https://ai.google.dev/pricing'>ai.google.dev/pricing</a> "
                        "for current quota limits.</div>",
                        unsafe_allow_html=True,
                    )

# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — Retrieval Inspector
# ══════════════════════════════════════════════════════════════════════════════
with tab_inspect:
    if not st.session_state.chat_history:
        st.info("💬 Ask at least one question first to see retrieval results here.")
    else:
        st.markdown("#### 🔎 Retrieval Inspector")
        st.markdown(
            """
            <div class="info-card">
            <strong>What you see here:</strong>
            For the most recent question, these are the document chunks that the TF-IDF
            retrieval system ranked as most relevant. The <em>similarity score</em> is
            the cosine similarity between the question and each chunk — higher means more
            lexically similar. This is <strong>not</strong> the same as semantic similarity
            from neural embeddings.
            </div>
            """,
            unsafe_allow_html=True,
        )

        last = st.session_state.chat_history[-1]
        st.markdown(f"**Last question:** {last['question']}")
        retrieved = last.get("retrieved") or []

        if not retrieved:
            st.warning("No relevant chunks were retrieved for the last question.")
        else:
            for pos, item in enumerate(retrieved, start=1):
                st.markdown(
                    f'<div class="chunk-card">'
                    f'<div class="chunk-meta">'
                    f"Rank {pos} · Page {item['page']} · "
                    f"Similarity score: {item['score']:.4f}"
                    f"</div>"
                    f"{item['text']}"
                    f"</div>",
                    unsafe_allow_html=True,
                )

            st.markdown(
                """
                <div class="note-box">
                <strong>Cosine similarity</strong> measures how much the question and the chunk
                share the same vocabulary (after TF-IDF weighting). A score of 1.0 means
                identical vocabulary; 0.0 means no shared terms. Modern RAG systems use
                neural embedding models for <em>semantic</em> similarity instead — but the
                pipeline structure is the same.
                </div>
                """,
                unsafe_allow_html=True,
            )

            # Show all past questions with a selector
            if len(st.session_state.chat_history) > 1:
                with st.expander("Inspect an earlier question"):
                    history_labels = [
                        f"{i+1}. {entry['question'][:60]}"
                        for i, entry in enumerate(st.session_state.chat_history)
                    ]
                    selected_idx = st.selectbox("Select question", range(len(history_labels)), format_func=lambda i: history_labels[i])
                    sel_entry = st.session_state.chat_history[selected_idx]
                    for pos, item in enumerate(sel_entry.get("retrieved") or [], start=1):
                        st.markdown(
                            f'<div class="chunk-card">'
                            f'<div class="chunk-meta">Rank {pos} · Page {item["page"]} · Score: {item["score"]:.4f}</div>'
                            f'{item["text"]}'
                            f"</div>",
                            unsafe_allow_html=True,
                        )

# ══════════════════════════════════════════════════════════════════════════════
# TAB 5 — RAG Pipeline
# ══════════════════════════════════════════════════════════════════════════════
with tab_pipeline:
    st.markdown("#### 🔬 How RAG Works — Step by Step")

    stages = [
        (
            "① Upload PDF",
            "The student uploads a PDF file. The application reads the raw bytes "
            "and passes them to a PDF parsing library (pypdf). Only text-based PDFs "
            "are supported; scanned PDFs need OCR preprocessing.",
            "A",
        ),
        (
            "② Extract Text",
            "The text is extracted page by page. Each page's text is cleaned "
            "(whitespace normalised) and stored alongside its page number. "
            "Page numbers let us cite sources in the final answer.",
            "B",
        ),
        (
            "③ Chunk Text",
            "Long pages are split into overlapping character-level chunks. "
            "Chunks are short enough to fit into a retrieval index (and later a model "
            "context window), while overlaps prevent important sentences from being cut. "
            "Each chunk remembers which page it came from.",
            "C",
        ),
        (
            "④ TF-IDF Indexing",
            "Each chunk is converted into a sparse numerical vector using TF-IDF "
            "(Term Frequency–Inverse Document Frequency). This is classical information "
            "retrieval — it captures how distinctive each word is across the document. "
            "This step does NOT train an LLM; it is an indexing step.",
            "D",
        ),
        (
            "⑤ Retrieval",
            "When a question is asked, it is also converted to a TF-IDF vector. "
            "Cosine similarity is computed between the question vector and every chunk "
            "vector. The top-k most similar chunks are selected as the retrieved context. "
            "TF-IDF measures lexical overlap — modern systems use neural embeddings for "
            "semantic similarity.",
            "E",
        ),
        (
            "⑥ Answer Generation",
            "In retrieval-only mode, the retrieved passages are displayed directly — "
            "no LLM is needed. In LLM mode, the retrieved chunks and the question are "
            "sent to Google Gemini (free tier). The model is instructed to answer ONLY "
            "from the provided context, preventing hallucinated facts.",
            "F",
        ),
    ]

    for title, explanation, stage in stages:
        with st.expander(f"**Stage {stage} — {title}**", expanded=False):
            st.markdown(explanation)

    st.markdown("---")
    st.markdown("#### 📌 Key Concepts for Students")

    concepts = {
        "RAG vs Fine-tuning": (
            "Fine-tuning changes the LLM's weights using training examples. "
            "RAG does not change any model weights — it retrieves relevant documents "
            "at query time and passes them as context."
        ),
        "Why RAG can still be wrong": (
            "If the PDF does not contain the answer, retrieval returns the closest "
            "match it can find. The LLM may then produce an answer that sounds "
            "plausible but is not supported by the document."
        ),
        "Why cite sources": (
            "Source citations (e.g., [Page 3]) let students verify the answer against "
            "the original document and detect hallucinations."
        ),
        "Upgrading retrieval": (
            "Replacing TF-IDF with a neural embedding model (e.g. sentence-transformers) "
            "and a vector database (e.g. ChromaDB, FAISS) allows semantic retrieval — "
            "the system can match questions to passages even when they use different words."
        ),
    }

    for concept, text in concepts.items():
        st.markdown(
            f'<div class="info-card"><strong>{concept}:</strong> {text}</div>',
            unsafe_allow_html=True,
        )

# ─────────────────────────────────────────────────────────────────────────────
# Footer
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("---")
st.caption(
    "🎓 **PDF RAG Lab** · Built with Streamlit + pypdf + scikit-learn · "
    "Optional LLM: Google Gemini free tier (`gemini-1.5-flash`, no credit card). "
    "This app indexes your PDF; it does **not** train any model. "
    "All free-tier quotas and rate limits of the chosen provider apply."
)
