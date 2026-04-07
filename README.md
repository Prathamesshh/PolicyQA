# Policy Document QA System
> BERT-Large fine-tuned on SQuAD → RAG pipeline → BART Summarization → Multilingual output

---

## Architecture 

```
ROW 1 — Document Input & Processing
┌──────┐   ┌──────────────┐   ┌─────────────────────┐   ┌─────────────┐   ┌───────────────┐
│Start │→  │User Uploads  │→  │Choose Input Type    │→  │Process Input│→  │AI Model:      │
│      │   │Policy Doc    │   │PDF / Image / Text   │   │OCR or Direct│   │BERT-Large QA  │
└──────┘   └──────────────┘   └─────────────────────┘   └─────────────┘   └───────────────┘

ROW 2 — User Interaction & Output
┌────────────────┐   ┌──────────────┐   ┌──────────────┐   ┌──────────────────┐   ┌─────┐
│User Selects    │→  │BART/T5 (sum) │→  │Display Output│→  │Multilingual?     │→  │ End │
│Action          │   │RAG Model (QA)│   │              │   │Translate/Default │   │     │
└────────────────┘   └──────────────┘   └──────────────┘   └──────────────────┘   └─────┘
```

---

## Files Overview

| File | Purpose |
|---|---|
| `training_bert.ipynb` | Fine-tune BERT-Large on SQuAD (Step 1) |
| `document_processor.py` | PDF/Image/Text ingestion + OCR (Row 1) |
| `rag_pipeline.py` | FAISS indexing + BERT-Large QA retrieval (Row 2 Q&A) |
| `summarizer_translator.py` | BART summarization + multilingual translation |
| `api.py` | FastAPI backend connecting everything |
| `app_ui.py` | Gradio frontend UI |

---

## Step-by-Step Setup

### 1. Environment

```bash
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt

# For OCR support (required for scanned PDFs / images):
# Ubuntu/Debian:
sudo apt install tesseract-ocr tesseract-ocr-eng

# macOS:
brew install tesseract


### 2. Train BERT-Large on SQuAD

```bash
# Option A: Use HuggingFace SQuAD 

# Option B: Use YOUR OWN SQuAD-style dataset
# Edit cfg.custom_data_path in train_bert.ipynb:
python train_bert.ipynb
```

**Your SQuAD-format JSON structure:**
```json
{
  "data": [{
    "title": "Policy Document",
    "paragraphs": [{
      "context": "The insurance covers...",
      "qas": [{
        "id": "q1",
        "question": "What does it cover?",
        "answers": [{"text": "insurance covers", "answer_start": 4}],
        "is_impossible": false
      }]
    }]
  }]
}
```

**Training on GPU (recommended):**
- `bert-large` needs ~16GB VRAM for batch_size=8
- Reduce `batch_size=4` + increase `grad_accumulation=8` for 8GB GPU
- Training time: ~2-3 hrs on A100, ~6-8 hrs on V100

**Expected metrics after training:**
- SQuAD v1.1: EM ~86%, F1 ~92%
- SQuAD v2.0: EM ~80%, F1 ~83%

### 3. Test Document Processing

```bash
python document_processor.py 
# Output: chunks with page numbers and text previews
```

### 4. Test RAG Pipeline

```bash
python rag_pipeline.py 
# Builds FAISS index, then interactive Q&A loop
```

### 5. Start the API Server

```bash
python api.py
# → http://localhost:8000
# → http://localhost:8000/docs  
```

### 6. Start the UI

```bash
# In a second terminal:
python app_ui.py
# → http://localhost:7860
```

---

## API Reference

### POST `/upload`
Upload PDF/Image/Text document. Builds/updates FAISS index.
```bash
curl -X POST http://localhost:8000/upload \
  -F "file=@policy.pdf"
```

### POST `/ask`
Ask a question about the indexed document.
```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the deductible?", "language": "en"}'
```

### POST `/summarize`
Summarize the document in any language.
```bash
curl -X POST http://localhost:8000/summarize \
  -H "Content-Type: application/json" \
  -d '{"language": "fr"}'
```

### GET `/languages`
Returns all supported output languages.

### GET `/health`
System status + index info.

---

## Model Configuration

### QA Model (BERT-Large)
Edit `QATrainingConfig` in `train_bert.ipynb`:
```python
model_name = "bert-large-uncased-whole-word-masking"  
# Alternatives:
# "deepset/roberta-large-squad2"     — slightly better F1
# "deepset/deberta-v3-large-squad2"  — best performance, slower
```

### Summarization Model
Edit `OutputHandler` in `api.py`:
```python
output_handler = OutputHandler(summarizer_model="bart")
# Options: "bart" | "t5" | "pegasus"
```

### Embedding Model (RAG)
Edit `RAGEngine` in `rag_pipeline.py`:
```python
embed_model = "sentence-transformers/all-mpnet-base-v2"  # default (best quality)
# Faster alternative: "sentence-transformers/all-MiniLM-L6-v2"
```

---


## Production Notes

1. **Model serving**: Wrap `api.py` with gunicorn + nginx for production
2. **GPU inference**: Set `device=0` in all pipeline calls
3. **Large document corpora**: Switch FAISS flat index to IVF (auto-triggered at >10k chunks)
4. **Security**: Add API key auth middleware to `api.py` before exposing publicly
5. **Caching**: Add Redis cache for repeated question/summary requests
