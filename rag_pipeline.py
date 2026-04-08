"""
rag_pipeline.py — RAG Engine
Loads your fine-tuned BERT-Large from ./bert-large/
Uses FAISS for retrieval + BERT QA for answer extraction.
"""

import os
import pickle
from pathlib import Path
from typing import List, Optional
from dataclasses import dataclass

import numpy as np
import torch
from loguru import logger
from transformers import AutoTokenizer, AutoModelForQuestionAnswering

from document_processor import DocumentChunk


# ──────────────────────────────────────────────
# DATA CLASSES
# ──────────────────────────────────────────────
@dataclass
class RetrievedContext:
    chunk: DocumentChunk
    score: float


@dataclass
class QAResult:
    question: str
    answer: str
    score: float
    context: str
    source: str
    page: int


# ──────────────────────────────────────────────
# RAG ENGINE
# ──────────────────────────────────────────────
class RAGEngine:
    """
    Full RAG pipeline:
      Embed → FAISS Index → Retrieve → BERT-Large QA
    Points to ./bert-large/ by default (your saved model).
    """

    def __init__(
        self,
        qa_model_path: str = "./bert-large",
        embed_model: str = "sentence-transformers/all-mpnet-base-v2",
        top_k: int = 5,
        max_answer_length: int = 150,
        index_path: str = "./faiss_index",
    ):
        self.top_k       = top_k
        self.index_path  = Path(index_path)
        self.index_path.mkdir(parents=True, exist_ok=True)
        self._chunks: List[DocumentChunk] = []
        self._index = None

        logger.info("Loading sentence-transformer embedding model...")
        from sentence_transformers import SentenceTransformer
        self.embedder = SentenceTransformer(embed_model)

        model_path = Path(qa_model_path)
        if not model_path.exists():
            logger.warning(
                f"Local model not found at {qa_model_path}. "
                "Falling back to deepset/bert-large-uncased-whole-word-masking-squad2"
            )
            model_id = "deepset/bert-large-uncased-whole-word-masking-squad2"
        else:
            logger.info(f"Loading fine-tuned BERT-Large from {model_path}...")
            model_id = str(model_path)

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.qa_tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.qa_model = AutoModelForQuestionAnswering.from_pretrained(model_id)
        self.qa_model = self.qa_model.to(self.device)
        self.qa_model.eval()
        self.max_answer_length = max_answer_length
        logger.success("RAG engine ready.")

    def build_index(self, chunks: List[DocumentChunk]):
        import faiss
        self._chunks = chunks
        logger.info(f"Embedding {len(chunks)} chunks...")
        embeddings = self.embedder.encode(
            [c.text for c in chunks],
            batch_size=64,
            show_progress_bar=True,
            normalize_embeddings=True,
        ).astype("float32")

        dim = embeddings.shape[1]
        if len(chunks) > 10_000:
            nlist     = min(256, len(chunks) // 10)
            quantizer = faiss.IndexFlatIP(dim)
            self._index = faiss.IndexIVFFlat(quantizer, dim, nlist, faiss.METRIC_INNER_PRODUCT)
            self._index.train(embeddings)
        else:
            self._index = faiss.IndexFlatIP(dim)

        self._index.add(embeddings)
        self._save_index()
        logger.success(f"FAISS index built: {self._index.ntotal} vectors.")

    def load_index(self):
        import faiss
        idx_file  = self.index_path / "index.faiss"
        meta_file = self.index_path / "chunks.pkl"
        if not idx_file.exists():
            raise FileNotFoundError("No saved index. Call build_index() first.")
        self._index = faiss.read_index(str(idx_file))
        with open(meta_file, "rb") as f:
            self._chunks = pickle.load(f)
        logger.success(f"Loaded index: {self._index.ntotal} vectors, {len(self._chunks)} chunks.")

    def _save_index(self):
        import faiss
        faiss.write_index(self._index, str(self.index_path / "index.faiss"))
        with open(self.index_path / "chunks.pkl", "wb") as f:
            pickle.dump(self._chunks, f)

    def retrieve(self, query: str, top_k: Optional[int] = None) -> List[RetrievedContext]:
        k     = top_k or self.top_k
        q_emb = self.embedder.encode([query], normalize_embeddings=True).astype("float32")
        scores, idxs = self._index.search(q_emb, k)
        return [
            RetrievedContext(chunk=self._chunks[i], score=float(s))
            for s, i in zip(scores[0], idxs[0]) if i != -1
        ]

    def _extract_answer(self, question: str, context: str) -> dict:
        """Extract answer from context using BERT QA model."""
        inputs = self.qa_tokenizer(
            question,
            context,
            add_special_tokens=True,
            return_tensors="pt",
            max_length=512,
            truncation=True,
        )

        input_ids = inputs["input_ids"].to(self.device)

        with torch.no_grad():
            outputs = self.qa_model(input_ids=input_ids)

        start_scores = outputs.start_logits
        end_scores = outputs.end_logits

        start_idx = torch.argmax(start_scores)
        end_idx = torch.argmax(end_scores) + 1

        answer_tokens = input_ids[0][start_idx:end_idx]
        answer = self.qa_tokenizer.decode(answer_tokens, skip_special_tokens=True)

        score = float((start_scores[0][start_idx] + end_scores[0][end_idx - 1]) / 2)
        score = torch.softmax(torch.tensor([score]), dim=0)[0].item()

        return {"answer": answer, "score": score}

    def _answer_directly(self, question: str) -> dict:
        """Answer using BERT directly without retrieval (zero-shot mode)."""
        try:
            # Use a default context if no document is available
            default_context = "This is a policy document containing general information."
            return self._extract_answer(question, default_context)
        except Exception as e:
            logger.warning(f"Direct BERT answer failed: {e}")
            return {"answer": "", "score": 0.0}

    def answer(self, question: str) -> QAResult:
        contexts = self.retrieve(question)
        if not contexts:
            return QAResult(question=question, answer="No relevant content found.",
                            score=0.0, context="", source="", page=0)

        candidates = []
        for rc in contexts:
            try:
                out = self._extract_answer(question=question, context=rc.chunk.text)
                candidates.append({
                    "answer":  out["answer"],
                    "score":   out["score"] * rc.score,
                    "context": rc.chunk.text,
                    "source":  rc.chunk.source,
                    "page":    rc.chunk.page,
                })
            except Exception as e:
                logger.warning(f"QA error on chunk: {e}")

        if not candidates:
            return QAResult(question=question, answer="Could not extract an answer.",
                            score=0.0, context="", source="", page=0)

        best = max(candidates, key=lambda x: x["score"])
        return QAResult(
            question=question, answer=best["answer"], score=best["score"],
            context=best["context"], source=best["source"], page=best["page"],
        )
