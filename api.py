"""
api.py — FastAPI Backend
Connects: document_processor → rag_pipeline (bert-large) → summarizer_translator
Run:  python api.py
Docs: http://localhost:8000/docs
"""

import shutil
import re
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from loguru import logger

from document_processor import DocumentProcessor
from rag_pipeline import RAGEngine
from summarizer_translator import OutputHandler, LANGUAGE_NAMES

# ──────────────────────────────────────────────────────
# APP INIT
# ──────────────────────────────────────────────────────
app = FastAPI(
    title="PolicyQA — BERT-Large Document Intelligence",
    description="Fine-tuned BERT-Large QA + BART Summarization + Multilingual",
    version="2.0.0",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)

doc_processor  = DocumentProcessor(chunk_size=400, chunk_overlap=50)
rag_engine: Optional[RAGEngine] = None
output_handler = None  # Lazy load

UPLOAD_DIR = Path("./uploaded_docs")
UPLOAD_DIR.mkdir(exist_ok=True)


# ──────────────────────────────────────────────────────
# LAZY LOAD — Models loaded on first request (not startup)
# ──────────────────────────────────────────────────────
def get_rag_engine():
    """Lazy-load RAG engine on first use"""
    global rag_engine
    if rag_engine is None:
        logger.info("Initializing RAG engine (lazy load)...")
        rag_engine = RAGEngine(qa_model_path="./bert-large", top_k=5)

        # Try loading existing index
        index_file = Path("./faiss_index/index.faiss")
        if index_file.exists():
            try:
                rag_engine.load_index()
                logger.success("✓ Loaded existing FAISS index")
            except Exception as e:
                logger.warning(f"Could not load index: {e}")
    return rag_engine


def get_output_handler():
    """Lazy-load output handler on first use"""
    global output_handler
    if output_handler is None:
        logger.info("Initializing output handler (lazy load)...")
        output_handler = OutputHandler(summarizer_model="bart")
    return output_handler


# ──────────────────────────────────────────────────────
# STARTUP — Simple health check only
# ──────────────────────────────────────────────────────
@app.on_event("startup")
async def startup_event():
    logger.info("🚀 PolicyQA API starting...")
    logger.info("Models will be loaded on first request (lazy loading)")


# ──────────────────────────────────────────────────────
# MODELS
# ──────────────────────────────────────────────────────
class QuestionRequest(BaseModel):
    question: str
    language: str = "en"

class SummarizeRequest(BaseModel):
    language: str = "en"

class StatusResponse(BaseModel):
    status: str
    message: str
    num_chunks: int = 0

class QAResponse(BaseModel):
    question: str
    answer: str
    score: float
    context: str
    source: str
    page: int
    language: str

class SummaryResponse(BaseModel):
    summary: str
    language: str
    language_name: str

class LanguagesResponse(BaseModel):
    languages: dict


# ──────────────────────────────────────────────────────
# ROUTES
# ──────────────────────────────────────────────────────

# ── Health ──
@app.get("/health")
async def health():
    engine = rag_engine
    indexed = engine._index is not None if engine else False
    return {
        "status": "ok",
        "index_loaded": indexed,
        "num_chunks": len(engine._chunks) if engine and engine._chunks else 0,
        "model_path": str(Path("./bert-large").resolve()),
    }


# ── Upload + Index (Row 1) ──
@app.post("/upload", response_model=StatusResponse)
async def upload_document(file: UploadFile = File(...)):
    allowed = {".pdf", ".png", ".jpg", ".jpeg", ".tiff", ".txt", ".docx"}
    suffix  = Path(file.filename).suffix.lower()
    if suffix not in allowed:
        raise HTTPException(400, detail=f"Unsupported file type: {suffix}")

    save_path = UPLOAD_DIR / file.filename
    with open(save_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    try:
        chunks = doc_processor.process(str(save_path))
    except Exception as e:
        raise HTTPException(500, detail=f"Document processing failed: {e}")

    if not chunks:
        raise HTTPException(422, detail="No text could be extracted from file.")

    engine = get_rag_engine()  # Lazy load
    all_chunks = (engine._chunks or []) + chunks
    engine.build_index(all_chunks)

    return StatusResponse(
        status="success",
        message=f"Processed '{file.filename}' — {len(chunks)} new chunks indexed.",
        num_chunks=len(all_chunks),
    )


# ── Ask Question (Row 2 — Q&A path) ──
@app.post("/ask", response_model=QAResponse)
async def ask_question(req: QuestionRequest):
    engine = get_rag_engine()  # Lazy load
    if engine._index is None:
        raise HTTPException(400, detail="No document indexed yet. Please upload a document first.")
    if not req.question.strip():
        raise HTTPException(400, detail="Question cannot be empty.")

    result = engine.answer(req.question)
    # Remove CID metadata from answer
    clean_answer = re.sub(r'\s*\(\s*cid\s*:\s*\d+\s*\)\s*', ' ', result.answer, flags=re.IGNORECASE).strip()
    clean_context = re.sub(r'\s*\(\s*cid\s*:\s*\d+\s*\)\s*', ' ', result.context, flags=re.IGNORECASE).strip()

    handler = get_output_handler()  # Lazy load
    output = handler.get_answer(clean_answer, language=req.language)

    return QAResponse(
        question=req.question,
        answer=output.content,
        score=round(result.score, 4),
        context=clean_context,
        source=result.source,
        page=result.page,
        language=req.language,
    )


# ── Summarize (Row 2 — Summary path) ──
@app.post("/summarize", response_model=SummaryResponse)
async def summarize_document(req: SummarizeRequest):
    engine = get_rag_engine()  # Lazy load
    if not engine._chunks:
        raise HTTPException(400, detail="No document indexed yet.")

    full_text = " ".join(c.text for c in engine._chunks[:60])
    # Remove CID metadata before summarization
    full_text = re.sub(r'\s*\(\s*cid\s*:\s*\d+\s*\)\s*', ' ', full_text, flags=re.IGNORECASE).strip()

    handler = get_output_handler()  # Lazy load
    output = handler.get_summary(full_text, language=req.language)

    return SummaryResponse(
        summary=output.content,
        language=output.language,
        language_name=output.language_name,
    )


# ── Languages ──
@app.get("/languages", response_model=LanguagesResponse)
async def get_languages():
    handler = get_output_handler()  # Lazy load
    return LanguagesResponse(languages=handler.available_languages())


# ──────────────────────────────────────────────────────
# ENTRY POINT
# ──────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=False, workers=1)
