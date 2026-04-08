#!/bin/bash
# Cloud startup script - builds index on first deployment

echo "🚀 Starting PolicyQA API..."

# Download models (cached by HF hub)
echo "📥 Pre-loading models..."
python -c "
from transformers import AutoModel, AutoTokenizer
# BERT-QA
AutoTokenizer.from_pretrained('deepset/bert-large-uncased-whole-word-masking-squad2')
AutoModel.from_pretrained('deepset/bert-large-uncased-whole-word-masking-squad2')

# Embeddings
from sentence_transformers import SentenceTransformer
SentenceTransformer('sentence-transformers/all-mpnet-base-v2')

# Summarizer
from transformers import AutoModelForSeq2SeqLM
AutoModelForSeq2SeqLM.from_pretrained('facebook/bart-large-cnn')

print('✓ All models loaded')
"

# Start API
echo "🔧 Starting Uvicorn server..."
uvicorn api:app --host 0.0.0.0 --port $PORT
