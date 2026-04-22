"""
streamlit_app.py — PolicyQA Streamlit Frontend
Connects to FastAPI backend (localhost:8000 or cloud URL)
Run: streamlit run streamlit_app.py
"""

import time
import os
import requests
from pathlib import Path
import streamlit as st

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
API_BASE = os.getenv("API_URL", "http://localhost:8000")  # Cloud or local API
SUPPORTED_EXTS = ["pdf", "png", "jpg", "jpeg", "tiff", "txt", "docx"]

LANGUAGE_MAP = {
    "🇬🇧  English (Default)": "en",
    "🇫🇷  French":            "fr",
    "🇩🇪  German":            "de",
    "🇪🇸  Spanish":           "es",
    "🇮🇳  Hindi":             "hi",
    "🇨🇳  Chinese":           "zh",
    "🇸🇦  Arabic":            "ar",
    "🇵🇹  Portuguese":        "pt",
    "🇯🇵  Japanese":          "ja",
    "🇷🇺  Russian":           "ru",
    "🇰🇷  Korean":            "ko",
}

# ─────────────────────────────────────────────
# PAGE SETUP
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="PolicyQA — BERT Document Intelligence",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@300;400;500;700&family=Outfit:wght@300;400;600;700;800&family=Playfair+Display:ital,wght@0,600;1,400&display=swap');

/* ── Base ── */
html, body, [class*="css"] { font-family: 'Outfit', sans-serif; }
.stApp { background: #080b10; color: #dde3ee; }

/* Hide/style Streamlit header bar */
[data-testid="stHeader"] {
    background: #080b10 !important;
}
header { background: #080b10 !important; }
.appViewContainer { background: #080b10 !important; }


/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: #0c0f16 !important;
    border-right: 1px solid #161d2e !important;
}
[data-testid="stSidebar"] .stMarkdown p,
[data-testid="stSidebar"] .stMarkdown div { color: #8a9ab8 !important; }

/* ── Top banner ── */
.top-banner {
    background: linear-gradient(100deg, #0c1424 0%, #0e1928 60%, #091420 100%);
    border: 1px solid #162440;
    border-radius: 18px;
    padding: 2.8rem 3.5rem 2.4rem;
    margin-bottom: 1.8rem;
    position: relative;
    overflow: hidden;
}
.top-banner::after {
    content: '';
    position: absolute;
    inset: 0;
    background:
        radial-gradient(ellipse 60% 50% at 85% 50%, rgba(56,189,248,0.06) 0%, transparent 70%),
        radial-gradient(ellipse 40% 60% at 10% 80%, rgba(99,102,241,0.05) 0%, transparent 60%);
    pointer-events: none;
}
.top-banner h1 {
    font-family: 'Outfit', sans-serif !important;
    font-size: 2.8rem;
    font-weight: 800;
    color: #f0f6ff;
    letter-spacing: -0.04em;
    line-height: 1.1;
    margin: 0 0 0.4rem;
}
.top-banner .model-tag {
    display: inline-block;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.7rem;
    background: rgba(56,189,248,0.1);
    color: #38bdf8;
    border: 1px solid rgba(56,189,248,0.2);
    border-radius: 99px;
    padding: 0.25rem 0.8rem;
    letter-spacing: 0.05em;
    margin-bottom: 1rem;
}
.top-banner .subtitle {
    font-family: 'Outfit', sans-serif;
    font-size: 1rem;
    color: #5a7aa8;
    font-weight: 300;
}

/* ── Metric cards ── */
.metric-row { display: flex; gap: 1rem; margin-bottom: 2rem; }
.mc {
    flex: 1;
    background: #0c1020;
    border: 1px solid #162030;
    border-radius: 14px;
    padding: 1.2rem 1.4rem;
    text-align: center;
}
.mc-val {
    font-family: 'Outfit', sans-serif;
    font-size: 2.2rem;
    font-weight: 800;
    background: linear-gradient(135deg, #38bdf8, #818cf8);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}
.mc-lbl {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.65rem;
    color: #2a3a5a;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    margin-top: 0.2rem;
}

/* ── Pipeline bar ── */
.pipe-bar {
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    gap: 0.3rem;
    margin-bottom: 2rem;
    padding: 1rem 1.4rem;
    background: #0c1020;
    border: 1px solid #162030;
    border-radius: 12px;
}
.pipe-node {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.72rem;
    color: #4a6a9a;
    background: #111828;
    border: 1px solid #1a2a40;
    border-radius: 8px;
    padding: 0.35rem 0.8rem;
    white-space: nowrap;
}
.pipe-node.active { color: #38bdf8; border-color: rgba(56,189,248,0.3); }
.pipe-arrow { color: #1a3050; font-size: 0.9rem; }

/* ── Section headers ── */
.sec-hdr {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.68rem;
    font-weight: 700;
    color: #2a3a5a;
    text-transform: uppercase;
    letter-spacing: 0.15em;
    margin-bottom: 0.8rem;
    padding-bottom: 0.5rem;
    border-bottom: 1px solid #0e1828;
}

/* ── Status indicators ── */
.dot-green  { color: #4ade80; font-size: 0.8rem; font-family: 'JetBrains Mono', monospace; }
.dot-red    { color: #f87171; font-size: 0.8rem; font-family: 'JetBrains Mono', monospace; }
.dot-yellow { color: #fbbf24; font-size: 0.8rem; font-family: 'JetBrains Mono', monospace; }
.dot-gray   { color: #374151; font-size: 0.8rem; font-family: 'JetBrains Mono', monospace; }

/* ── Answer card ── */
.ans-card {
    background: linear-gradient(135deg, #09131f, #0c1828);
    border: 1px solid #163050;
    border-left: 4px solid #38bdf8;
    border-radius: 12px;
    padding: 1.8rem 2.2rem;
    margin: 1.2rem 0;
}
.ans-label {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.65rem;
    color: #38bdf8;
    text-transform: uppercase;
    letter-spacing: 0.15em;
    margin-bottom: 0.8rem;
}
.ans-text {
    font-family: 'Playfair Display', serif;
    font-size: 1.15rem;
    color: #c8ddf5;
    line-height: 1.8;
}
.ans-meta {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.68rem;
    color: #1e3050;
    margin-top: 1rem;
    padding-top: 0.8rem;
    border-top: 1px solid #0e1828;
}

/* ── Summary card ── */
.sum-card {
    background: linear-gradient(135deg, #091508, #0c1a10);
    border: 1px solid #163020;
    border-left: 4px solid #4ade80;
    border-radius: 12px;
    padding: 1.8rem 2.2rem;
    margin: 1.2rem 0;
}
.sum-label {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.65rem;
    color: #4ade80;
    text-transform: uppercase;
    letter-spacing: 0.15em;
    margin-bottom: 0.8rem;
}
.sum-text {
    font-family: 'Outfit', sans-serif;
    font-size: 1rem;
    color: #b8d4c0;
    line-height: 1.85;
    font-weight: 300;
}
.sum-meta {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.68rem;
    color: #163020;
    margin-top: 1rem;
    padding-top: 0.8rem;
    border-top: 1px solid #0e1a0c;
}

/* ── Error / warning ── */
.err-card {
    background: #120808;
    border: 1px solid #3a1010;
    border-left: 4px solid #f87171;
    border-radius: 10px;
    padding: 1rem 1.4rem;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.78rem;
    color: #f87171;
}
.ok-card {
    background: #091208;
    border: 1px solid #163020;
    border-left: 4px solid #4ade80;
    border-radius: 10px;
    padding: 1rem 1.4rem;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.78rem;
    color: #4ade80;
}

/* ── Confidence bar ── */
.conf-wrap {
    margin: 1rem 0 0.3rem;
}
.conf-top {
    display: flex;
    justify-content: space-between;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.68rem;
    color: #2a3a5a;
    margin-bottom: 0.4rem;
}
.conf-track {
    background: #0e1828;
    border-radius: 99px;
    height: 5px;
    overflow: hidden;
}
.conf-fill {
    height: 100%;
    border-radius: 99px;
    background: linear-gradient(90deg, #1d4ed8, #38bdf8);
}

/* ── Context viewer ── */
.ctx-box {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.72rem;
    color: #2a4a7a;
    line-height: 1.75;
    background: #060a10;
    border: 1px solid #0e1828;
    border-radius: 10px;
    padding: 1.2rem 1.5rem;
    white-space: pre-wrap;
    word-break: break-word;
}

/* ── History items ── */
.hist-item {
    background: #0c1020;
    border: 1px solid #141e30;
    border-radius: 10px;
    padding: 0.8rem 1rem;
    margin-bottom: 0.6rem;
}
.hist-q { font-family: 'JetBrains Mono', monospace; font-size: 0.7rem; color: #2a4a7a; }
.hist-a { font-family: 'Outfit', sans-serif; font-size: 0.82rem; color: #7a9ab8; margin-top: 0.3rem; }

/* ── Streamlit overrides ── */
.stTextInput > div > div > input,
.stTextArea > div > div > textarea {
    background: #0c1020 !important;
    border: 1px solid #162030 !important;
    color: #dde3ee !important;
    border-radius: 10px !important;
    font-family: 'Outfit', sans-serif !important;
    font-size: 0.95rem !important;
}
.stTextInput > div > div > input:focus,
.stTextArea > div > div > textarea:focus {
    border-color: #38bdf8 !important;
    box-shadow: 0 0 0 2px rgba(56,189,248,0.1) !important;
}
div[data-baseweb="select"] > div {
    background: #0c1020 !important;
    border: 1px solid #162030 !important;
    color: #dde3ee !important;
    border-radius: 10px !important;
}
.stFileUploader > div {
    background: #080b10 !important;
    border: 2px dashed #162030 !important;
    border-radius: 14px !important;
}
.stFileUploader > div > div {
    background: #080b10 !important;
}
.stFileUploader > div > div > div {
    background: #080b10 !important;
}
.stFileUploader > div > div > div > div {
    background: #080b10 !important;
}
/* Target all text in file uploader */
.stFileUploader p,
.stFileUploader span,
.stFileUploader div {
    color: #8a9ab8 !important;
}
.stFileUploader label { color: #4a6a9a !important; }

/* Primary buttons */
.stButton > button[kind="primary"],
.stButton > button:not([kind="secondary"]) {
    background: linear-gradient(135deg, #1d4ed8 0%, #2563eb 100%) !important;
    color: #fff !important;
    border: none !important;
    border-radius: 10px !important;
    font-family: 'Outfit', sans-serif !important;
    font-weight: 700 !important;
    font-size: 0.9rem !important;
    letter-spacing: 0.02em !important;
    padding: 0.55rem 1.6rem !important;
    transition: all 0.2s !important;
}
.stButton > button:not([kind="secondary"]):hover {
    background: linear-gradient(135deg, #2563eb, #38bdf8) !important;
    box-shadow: 0 4px 20px rgba(56,189,248,0.25) !important;
    transform: translateY(-1px) !important;
}
.stButton > button[kind="secondary"] {
    background: #0e1828 !important;
    border: 1px solid #162030 !important;
    color: #4a6a9a !important;
    border-radius: 10px !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.75rem !important;
}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    background: transparent !important;
    border-bottom: 1px solid #0e1828 !important;
    gap: 0.3rem;
}
.stTabs [data-baseweb="tab"] {
    font-family: 'Outfit', sans-serif !important;
    font-size: 0.88rem !important;
    font-weight: 600 !important;
    color: #2a3a5a !important;
    border-radius: 8px 8px 0 0 !important;
    padding: 0.55rem 1.3rem !important;
}
.stTabs [aria-selected="true"] {
    color: #dde3ee !important;
    background: #0c1020 !important;
    border-bottom: 2px solid #38bdf8 !important;
}

/* Expander */
.streamlit-expanderHeader {
    background: #0c1020 !important;
    border: 1px solid #162030 !important;
    border-radius: 10px !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.78rem !important;
    color: #4a6a9a !important;
}
div[data-testid="stExpander"] { border: none !important; }

/* Progress / spinner */
.stSpinner > div { border-top-color: #38bdf8 !important; }
hr { border-color: #0e1828 !important; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────────
defaults = {
    "qa_history":   [],
    "indexed_docs": [],
    "chunk_count":  0,
    "api_online":   False,
    "last_answer":  None,
    "last_summary": None,
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v


# ─────────────────────────────────────────────
# API HELPERS
# ─────────────────────────────────────────────
def check_health() -> dict | None:
    try:
        r = requests.get(f"{API_BASE}/health", timeout=4)
        d = r.json()
        st.session_state.api_online  = True
        st.session_state.chunk_count = d.get("num_chunks", 0)
        return d
    except Exception:
        st.session_state.api_online = False
        return None


def call_upload(data: bytes, filename: str) -> tuple[dict, int]:
    try:
        r = requests.post(
            f"{API_BASE}/upload",
            files={"file": (filename, data, "application/octet-stream")},
            timeout=300,
        )
        return r.json(), r.status_code
    except requests.exceptions.ConnectionError:
        return {"detail": "API server not reachable. Run: python api.py"}, 503
    except Exception as e:
        return {"detail": str(e)}, 500


def call_ask(question: str, lang: str) -> tuple[dict, int]:
    try:
        r = requests.post(
            f"{API_BASE}/ask",
            json={"question": question, "language": lang},
            timeout=90,
        )
        return r.json(), r.status_code
    except requests.exceptions.ConnectionError:
        return {"detail": "API server not reachable."}, 503
    except Exception as e:
        return {"detail": str(e)}, 500


def call_summarize(lang: str) -> tuple[dict, int]:
    try:
        r = requests.post(
            f"{API_BASE}/summarize",
            json={"language": lang},
            timeout=300,
        )
        return r.json(), r.status_code
    except requests.exceptions.ConnectionError:
        return {"detail": "API server not reachable."}, 503
    except Exception as e:
        return {"detail": str(e)}, 500


# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="margin-bottom:1.5rem;">
      <div style="font-family:'Outfit',sans-serif;font-size:1.3rem;
                  font-weight:800;color:#f0f6ff;letter-spacing:-0.03em;">
        🧠 PolicyQA
      </div>
      <div style="font-family:'JetBrains Mono',monospace;font-size:0.62rem;
                  color:#1e3050;letter-spacing:0.12em;text-transform:uppercase;
                  margin-top:0.2rem;">
        BERT-Large · Fine-tuned
      </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Live status ──
    health = check_health()
    st.markdown('<div class="sec-hdr">System Status</div>', unsafe_allow_html=True)

    if st.session_state.api_online:
        indexed = health.get("index_loaded", False)
        n       = health.get("num_chunks", 0)
        mp      = health.get("model_path", "./bert-large")
        st.markdown(f'<div class="dot-green">● API  Online</div>', unsafe_allow_html=True)
        st.markdown(
            f'<div class="dot-{"green" if indexed else "yellow"}">'
            f'{"●" if indexed else "○"} Index  {"Loaded" if indexed else "Empty"}'
            f'{"  (" + str(n) + " chunks)" if indexed else ""}</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            f'<div style="font-family:JetBrains Mono,monospace;font-size:0.62rem;'
            f'color:#1e3050;margin-top:0.4rem;word-break:break-all;">'
            f'Model: {Path(mp).name}</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown('<div class="dot-red">● API  Offline</div>', unsafe_allow_html=True)
        st.markdown("""
        <div style="font-family:'JetBrains Mono',monospace;font-size:0.72rem;
                    color:#3a1010;margin-top:0.5rem;line-height:1.8;">
          Start backend:<br>
          <span style="color:#fbbf24;">python api.py</span>
        </div>""", unsafe_allow_html=True)

    if st.button("↻  Refresh", use_container_width=True, key="refresh"):
        st.rerun()

    st.markdown("---")

    # ── Indexed docs ──
    st.markdown('<div class="sec-hdr">Indexed Documents</div>', unsafe_allow_html=True)
    if st.session_state.indexed_docs:
        for d in st.session_state.indexed_docs:
            st.markdown(
                f'<div style="font-family:JetBrains Mono,monospace;font-size:0.7rem;'
                f'color:#1e3a6a;padding:0.3rem 0;border-bottom:1px solid #0e1828;">'
                f'📄 {d}</div>',
                unsafe_allow_html=True,
            )
    else:
        st.markdown(
            '<div style="font-family:JetBrains Mono,monospace;font-size:0.7rem;color:#1a2a40;">No docs yet</div>',
            unsafe_allow_html=True,
        )

    st.markdown("---")

    # ── Q&A history ──
    st.markdown('<div class="sec-hdr">Recent Q&A</div>', unsafe_allow_html=True)
    if st.session_state.qa_history:
        for item in reversed(st.session_state.qa_history[-5:]):
            q_short = item["q"][:55] + ("…" if len(item["q"]) > 55 else "")
            a_short = item["a"][:75] + ("…" if len(item["a"]) > 75 else "")
            st.markdown(
                f'<div class="hist-item">'
                f'<div class="hist-q">Q: {q_short}</div>'
                f'<div class="hist-a">{a_short}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )
        if st.button("Clear History", use_container_width=True, key="clr"):
            st.session_state.qa_history = []
            st.rerun()
    else:
        st.markdown(
            '<div style="font-family:JetBrains Mono,monospace;font-size:0.7rem;color:#1a2a40;">No questions yet</div>',
            unsafe_allow_html=True,
        )

    st.markdown("---")
    st.markdown("""
    <div style="font-family:'JetBrains Mono',monospace;font-size:0.62rem;
                color:#0e1828;line-height:1.9;">
      bert-large-uncased-wwm<br>
      all-mpnet-base-v2<br>
      facebook/bart-large-cnn<br>
      Helsinki-NLP/opus-mt<br>
      FAISS · FastAPI · Streamlit
    </div>""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# MAIN AREA — HEADER
# ─────────────────────────────────────────────
st.markdown("""
<div class="top-banner">
  <div class="model-tag">bert-large-uncased-whole-word-masking · SQuAD fine-tuned · FAISS RAG</div>
  <h1>🧠 PolicyQA</h1>
  <div class="subtitle">
    Upload a policy document · Ask questions · Get instant answers · Summarize · Translate
  </div>
</div>
""", unsafe_allow_html=True)

# ── Pipeline bar ──
health_now = check_health()
indexed_now = health_now.get("index_loaded", False) if health_now else False

st.markdown(f"""
<div class="pipe-bar">
  <span class="pipe-node active">📄 Upload</span>
  <span class="pipe-arrow">→</span>
  <span class="pipe-node">🔍 OCR/Extract</span>
  <span class="pipe-arrow">→</span>
  <span class="pipe-node">🧩 Chunk</span>
  <span class="pipe-arrow">→</span>
  <span class="pipe-node">📐 Embed (mpnet)</span>
  <span class="pipe-arrow">→</span>
  <span class="pipe-node {'active' if indexed_now else ''}">⚡ FAISS Index</span>
  <span class="pipe-arrow">→</span>
  <span class="pipe-node">🤖 BERT-Large QA</span>
  <span class="pipe-arrow">→</span>
  <span class="pipe-node">🌐 Multilingual</span>
</div>
""", unsafe_allow_html=True)

# ── Metric strip ──
c1, c2, c3, c4 = st.columns(4)
with c1:
    st.markdown(f'<div class="mc"><div class="mc-val">{len(st.session_state.indexed_docs)}</div><div class="mc-lbl">Documents</div></div>', unsafe_allow_html=True)
with c2:
    st.markdown(f'<div class="mc"><div class="mc-val">{st.session_state.chunk_count}</div><div class="mc-lbl">Chunks Indexed</div></div>', unsafe_allow_html=True)
with c3:
    st.markdown(f'<div class="mc"><div class="mc-val">{len(st.session_state.qa_history)}</div><div class="mc-lbl">Questions Asked</div></div>', unsafe_allow_html=True)
with c4:
    st.markdown(f'<div class="mc"><div class="mc-val">{len(LANGUAGE_MAP)}</div><div class="mc-lbl">Output Languages</div></div>', unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────
tab_upload, tab_qa, tab_sum = st.tabs([
    "📤  Upload Document",
    "❓  Ask a Question",
    "📝  Summarize",
])


# ══════════════════════════════════════════════
# TAB 1 — UPLOAD
# ══════════════════════════════════════════════
with tab_upload:
    st.markdown('<div class="sec-hdr">Row 1 — Document Input & Processing</div>', unsafe_allow_html=True)

    col_left, col_right = st.columns([3, 1])

    with col_left:
        uploaded = st.file_uploader(
            "Drop your policy document here",
            type=SUPPORTED_EXTS,
            help="Scanned PDFs are auto-OCR'd via Tesseract. Text PDFs use pdfplumber.",
        )

    with col_right:
        st.markdown("""
        <div style="padding:1rem;font-family:'JetBrains Mono',monospace;
                    font-size:0.72rem;color:#1e3050;line-height:2;
                    background:#0c1020;border:1px solid #162030;border-radius:12px;">
          <b style="color:#2a4a7a;">Accepted:</b><br>
          PDF · PNG · JPG<br>TIFF · TXT · DOCX<br><br>
          <b style="color:#2a4a7a;">Pipeline:</b><br>
          → Text extract / OCR<br>
          → Sliding window chunks<br>
          → mpnet embeddings<br>
          → FAISS index
        </div>
        """, unsafe_allow_html=True)

    if uploaded:
        size_kb = uploaded.size / 1024
        st.markdown(
            f'<div style="font-family:JetBrains Mono,monospace;font-size:0.72rem;'
            f'color:#2a4a7a;margin:0.5rem 0;">'
            f'Selected: <b style="color:#4a7aaa;">{uploaded.name}</b>'
            f'&nbsp;·&nbsp;{size_kb:.1f} KB</div>',
            unsafe_allow_html=True,
        )

        if st.button("🚀  Process & Index Document", key="upload_btn"):
            if not st.session_state.api_online:
                st.markdown('<div class="err-card">❌  API server offline — run: python api.py</div>', unsafe_allow_html=True)
            else:
                with st.spinner(f"Processing {uploaded.name} — OCR + chunking + embedding + indexing…"):
                    t0 = time.time()
                    resp, code = call_upload(uploaded.getvalue(), uploaded.name)
                    elapsed = time.time() - t0

                if code == 200:
                    n = resp.get("num_chunks", 0)
                    st.markdown(
                        f'<div class="ok-card">✅  {resp.get("message","Success")}<br>'
                        f'⏱  {elapsed:.1f}s &nbsp;·&nbsp; {n} total chunks in index</div>',
                        unsafe_allow_html=True,
                    )
                    if uploaded.name not in st.session_state.indexed_docs:
                        st.session_state.indexed_docs.append(uploaded.name)
                    st.session_state.chunk_count = n
                    time.sleep(0.4)
                    st.rerun()
                else:
                    st.markdown(
                        f'<div class="err-card">❌  {resp.get("detail","Unknown error")}</div>',
                        unsafe_allow_html=True,
                    )

    st.markdown("<br>", unsafe_allow_html=True)
    with st.expander("⚙️  Processing pipeline — step by step"):
        st.markdown("""
**1. File type detection** — extension → processing branch  
**2. PDF** → `pdfplumber` for text layer → Tesseract OCR fallback for scanned pages  
**3. Image** → Tesseract OCR directly  
**4. Text/DOCX** → direct read  
**5. Cleaning** → whitespace normalisation, non-ASCII strip  
**6. Chunking** → 400-word sliding window, 50-word overlap (preserves cross-boundary context)  
**7. Embedding** → `sentence-transformers/all-mpnet-base-v2` → 768-dim vectors  
**8. FAISS index** → `IndexFlatIP` (cosine similarity via normalised inner product)
        """)


# ══════════════════════════════════════════════
# TAB 2 — Q&A
# ══════════════════════════════════════════════
with tab_qa:
    st.markdown('<div class="sec-hdr">Row 2 — Q&A Path · BERT-Large RAG</div>', unsafe_allow_html=True)

    col_q, col_l = st.columns([3, 1])
    with col_q:
        question = st.text_area(
            "Your question",
            placeholder=(
                "e.g.  What is the maximum coverage amount?\n"
                "      What are the policy exclusions?\n"
                "      When does the policy expire?"
            ),
            height=110,
            label_visibility="collapsed",
            key="question_box",
        )
    with col_l:
        qa_lang_display = st.selectbox(
            "Answer language",
            options=list(LANGUAGE_MAP.keys()),
            key="qa_lang_sel",
            label_visibility="collapsed",
        )

    btn_col, _ = st.columns([1, 4])
    with btn_col:
        ask_clicked = st.button("🔍  Get Answer", key="ask_btn", use_container_width=True)

    if ask_clicked:
        if not question.strip():
            st.warning("Please type a question first.")
        elif not st.session_state.api_online:
            st.markdown('<div class="err-card">❌  API offline — run: python api.py</div>', unsafe_allow_html=True)
        else:
            lang_code = LANGUAGE_MAP[qa_lang_display]
            with st.spinner("Retrieving context… extracting answer with BERT-Large…"):
                t0 = time.time()
                resp, code = call_ask(question.strip(), lang_code)
                elapsed = time.time() - t0

            if code == 200:
                answer  = resp["answer"]
                score   = resp["score"]
                source  = resp["source"]
                page    = resp["page"]
                context = resp.get("context", "")
                pct     = min(int(score * 100), 100)

                st.session_state.last_answer = resp
                st.session_state.qa_history.append({
                    "q": question, "a": answer, "score": score
                })

                # Answer display
                st.markdown(f"""
                <div class="ans-card">
                  <div class="ans-label">▶ Answer</div>
                  <div class="ans-text">{answer}</div>
                  <div class="ans-meta">
                    📄 {source} &nbsp;·&nbsp; Page {page}
                    &nbsp;·&nbsp; ⏱ {elapsed:.2f}s
                    &nbsp;·&nbsp; 🌐 {qa_lang_display.strip()}
                  </div>
                </div>
                """, unsafe_allow_html=True)

                # Confidence bar
                bar_color = (
                    "#4ade80" if pct >= 70
                    else "#fbbf24" if pct >= 40
                    else "#f87171"
                )
                st.markdown(f"""
                <div class="conf-wrap">
                  <div class="conf-top">
                    <span>Model Confidence</span>
                    <span style="color:{bar_color};">{pct}%</span>
                  </div>
                  <div class="conf-track">
                    <div class="conf-fill" style="width:{pct}%;background:linear-gradient(90deg,#1d4ed8,{bar_color});"></div>
                  </div>
                </div>
                """, unsafe_allow_html=True)

                # Context viewer
                if context:
                    with st.expander("🔎  View retrieved context passage"):
                        st.markdown(
                            f'<div class="ctx-box">{context[:1200]}{"…" if len(context)>1200 else ""}</div>',
                            unsafe_allow_html=True,
                        )

            elif code == 400 and "No document" in resp.get("detail", ""):
                st.markdown(
                    '<div class="err-card">⚠️  No document indexed — upload one in the Upload tab first.</div>',
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f'<div class="err-card">❌  {resp.get("detail", "API error")}</div>',
                    unsafe_allow_html=True,
                )

    st.markdown("<br>", unsafe_allow_html=True)
    with st.expander("⚙️  RAG + BERT-Large QA — how it works"):
        st.markdown("""
**1. Query embedding** — question encoded with `all-mpnet-base-v2`  
**2. FAISS search** — top-5 most similar chunks retrieved (cosine similarity)  
**3. BERT-Large QA** — `bert-large-uncased-whole-word-masking` fine-tuned on your SQuAD JSON  
   → Model loaded from `./bert-large/` (your trained checkpoint)  
   → Extracts exact answer span from each retrieved context  
**4. Score fusion** — `retrieval_score × QA_confidence` → best answer selected  
**5. Translation** — Helsinki-NLP OPUS-MT if non-English chosen  
        """)


# ══════════════════════════════════════════════
# TAB 3 — SUMMARIZE
# ══════════════════════════════════════════════
with tab_sum:
    st.markdown('<div class="sec-hdr">Row 2 — Summarization Path · BART-large-cnn</div>', unsafe_allow_html=True)

    col_s, col_info = st.columns([2, 1])

    with col_s:
        st.markdown("""
        <div style="font-family:'Outfit',sans-serif;font-size:0.95rem;
                    color:#3a5a7a;font-weight:300;line-height:1.8;margin-bottom:1.2rem;">
          Generate an abstractive summary of the entire indexed document using
          <b style="color:#5a8ab8;">facebook/bart-large-cnn</b>.<br>
          Long documents are handled with chunked map-reduce summarisation.
        </div>
        """, unsafe_allow_html=True)

        sum_lang_display = st.selectbox(
            "Summary language",
            options=list(LANGUAGE_MAP.keys()),
            key="sum_lang_sel",
        )

        sum_clicked = st.button("✨  Generate Summary", key="sum_btn")

    with col_info:
        st.markdown("""
        <div style="padding:1.2rem;font-family:'JetBrains Mono',monospace;
                    font-size:0.7rem;color:#1e3050;line-height:2;
                    background:#0c1020;border:1px solid #162030;border-radius:12px;">
          <b style="color:#2a4a7a;">Model</b><br>
          bart-large-cnn<br><br>
          <b style="color:#2a4a7a;">Strategy</b><br>
          Map-reduce for long docs<br><br>
          <b style="color:#2a4a7a;">Output</b><br>
          150–300 word abstract<br><br>
          <b style="color:#2a4a7a;">Time</b><br>
          ~15–60s depending on size
        </div>
        """, unsafe_allow_html=True)

    if sum_clicked:
        if not st.session_state.api_online:
            st.markdown('<div class="err-card">❌  API offline — run: python api.py</div>', unsafe_allow_html=True)
        else:
            lang_code = LANGUAGE_MAP[sum_lang_display]
            with st.spinner("Summarizing document — BART-large-cnn running…"):
                t0 = time.time()
                resp, code = call_summarize(lang_code)
                elapsed = time.time() - t0

            if code == 200:
                summary   = resp["summary"]
                lang_name = resp["language_name"]
                st.session_state.last_summary = resp

                st.markdown(f"""
                <div class="sum-card">
                  <div class="sum-label">▶ Document Summary</div>
                  <div class="sum-text">{summary}</div>
                  <div class="sum-meta">
                    🌐 {lang_name}
                    &nbsp;·&nbsp; ⏱ {elapsed:.1f}s
                    &nbsp;·&nbsp; {len(summary.split())} words
                  </div>
                </div>
                """, unsafe_allow_html=True)

                st.text_area(
                    "Copy text:",
                    value=summary,
                    height=110,
                    key="sum_copy_box",
                )

            elif code == 400 and "No document" in resp.get("detail", ""):
                st.markdown(
                    '<div class="err-card">⚠️  No document indexed — upload one in the Upload tab first.</div>',
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f'<div class="err-card">❌  {resp.get("detail", "API error")}</div>',
                    unsafe_allow_html=True,
                )

    st.markdown("<br>", unsafe_allow_html=True)
    with st.expander("⚙️  Summarization pipeline — how it works"):
        st.markdown("""
**1. Text assembly** — first 60 indexed chunks joined into full document text  
**2. Token check** — fits in 1024 tokens → direct BART summarization  
**3. Map-reduce** (long docs) → chunk → summarize each → combine → final summarize  
**4. BART-large-cnn** — Facebook's abstractive model (not extractive)  
**5. Translation** → Helsinki-NLP OPUS-MT if non-English language selected  
        """)


# ─────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div style="font-family:'JetBrains Mono',monospace;font-size:0.62rem;
            color:#0e1828;text-align:center;padding:0.8rem 0;">
  PolicyQA &nbsp;·&nbsp; bert-large-uncased-wwm (fine-tuned) &nbsp;·&nbsp;
  all-mpnet-base-v2 &nbsp;·&nbsp; FAISS &nbsp;·&nbsp;
  bart-large-cnn &nbsp;·&nbsp; Helsinki-NLP OPUS-MT &nbsp;·&nbsp;
  FastAPI + Streamlit
</div>
""", unsafe_allow_html=True)
