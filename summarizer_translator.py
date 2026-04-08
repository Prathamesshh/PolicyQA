"""
STEP 4: Summarization (BART / T5) + Multilingual Translation
Row 2 of your diagram — the "Wants Summary" path + multilingual output.
"""

from typing import Optional, List
from dataclasses import dataclass

import torch
from loguru import logger
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


# ──────────────────────────────────────────────
# SUMMARIZER
# ──────────────────────────────────────────────
class PolicySummarizer:
    """
    Abstractive summarisation using BART-large-cnn (default) or T5.
    Handles long documents via chunked summarisation (map-reduce style).
    """

    SUPPORTED_MODELS = {
        "bart": "facebook/bart-large-cnn",
        "t5":   "google/flan-t5-large",
        "pegasus": "google/pegasus-xsum",
    }

    def __init__(self, model_key: str = "bart", device: Optional[int] = None):
        model_name = self.SUPPORTED_MODELS.get(model_key, model_key)
        dev = device if device is not None else (0 if torch.cuda.is_available() else -1)
        logger.info(f"Loading summarization model: {model_name}")

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

        if dev >= 0:
            self.model = self.model.to(f"cuda:{dev}")
        else:
            self.model = self.model.to("cpu")

        self.device = "cuda" if dev >= 0 else "cpu"
        self.model_key = model_key
        self.max_input_tokens = 1024   # BART limit; T5 can go higher

    def summarize(
        self,
        text: str,
        max_length: int = 300,
        min_length: int = 80,
        num_beams: int = 4,
    ) -> str:
        """
        Summarize text of any length.
        For long texts: chunk → summarize each → summarize summaries.
        """
        tokens = self.tokenizer.encode(text, truncation=False)
        if len(tokens) <= self.max_input_tokens:
            return self._summarize_chunk(text, max_length, min_length, num_beams)

        # Map-reduce: split into chunks, summarize each, then combine
        logger.info(f"Long text ({len(tokens)} tokens) — using chunked summarization.")
        chunk_summaries = []
        step = self.max_input_tokens - 100   # slight overlap
        for start in range(0, len(tokens), step):
            chunk_tokens = tokens[start: start + self.max_input_tokens]
            chunk_text = self.tokenizer.decode(chunk_tokens, skip_special_tokens=True)
            s = self._summarize_chunk(chunk_text, max_length // 2, min_length // 2, num_beams)
            chunk_summaries.append(s)

        combined = " ".join(chunk_summaries)
        return self._summarize_chunk(combined, max_length, min_length, num_beams)

    def _summarize_chunk(self, text, max_length, min_length, num_beams) -> str:
        inputs = self.tokenizer.encode(text, return_tensors="pt", truncation=True, max_length=1024)
        inputs = inputs.to(self.device)

        summary_ids = self.model.generate(
            inputs,
            max_length=max_length,
            min_length=min_length,
            num_beams=num_beams,
            early_stopping=True,
        )

        summary_text = self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        return summary_text

    def bullet_summary(self, text: str, n_points: int = 5) -> List[str]:
        """Return a bullet-point style summary as a list of sentences."""
        summary = self.summarize(text)
        import re
        sentences = re.split(r"(?<=[.!?])\s+", summary.strip())
        return sentences[:n_points]


# ──────────────────────────────────────────────
# MULTILINGUAL TRANSLATOR
# ──────────────────────────────────────────────
LANGUAGE_MODELS = {
    # language_code: Helsinki-NLP model
    "fr": "Helsinki-NLP/opus-mt-en-fr",
    "de": "Helsinki-NLP/opus-mt-en-de",
    "es": "Helsinki-NLP/opus-mt-en-es",
    "hi": "Helsinki-NLP/opus-mt-en-hi",
    "zh": "Helsinki-NLP/opus-mt-en-zh",
    "ar": "Helsinki-NLP/opus-mt-en-ar",
    "pt": "Helsinki-NLP/opus-mt-en-ROMANCE",  # covers pt/es/fr/it/ro
    "ja": "Helsinki-NLP/opus-mt-en-jap",
    "ru": "Helsinki-NLP/opus-mt-en-ru",
    "ko": "Helsinki-NLP/opus-mt-tc-big-en-ko",
}

LANGUAGE_NAMES = {
    "en": "English (Default)",
    "fr": "French",
    "de": "German",
    "es": "Spanish",
    "hi": "Hindi",
    "zh": "Chinese",
    "ar": "Arabic",
    "pt": "Portuguese",
    "ja": "Japanese",
    "ru": "Russian",
    "ko": "Korean",
}


class MultilingualTranslator:
    """
    Lazy-loads translation models for requested languages.
    Uses Helsinki-NLP OPUS-MT models (free, offline).
    """

    def __init__(self):
        self._models = {}
        self._tokenizers = {}
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def translate(self, text: str, target_lang: str) -> str:
        if target_lang == "en":
            return text                         # default language — no translation
        if target_lang not in LANGUAGE_MODELS:
            raise ValueError(f"Unsupported language: {target_lang}. "
                             f"Supported: {list(LANGUAGE_MODELS.keys())}")

        tokenizer, model = self._get_model(target_lang)

        # Handle long texts by splitting into sentences
        import re
        sentences = re.split(r"(?<=[.!?])\s+", text.strip())
        translated = []
        for sent in sentences:
            if not sent.strip():
                continue

            inputs = tokenizer(sent, return_tensors="pt", truncation=True, max_length=512)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                translated_ids = model.generate(**inputs, max_length=512)

            translated_text = tokenizer.decode(translated_ids[0], skip_special_tokens=True)
            translated.append(translated_text)

        return " ".join(translated)

    def _get_model(self, lang: str):
        if lang not in self._models:
            model_name = LANGUAGE_MODELS[lang]
            logger.info(f"Loading translation model: {model_name}")

            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
            model = model.to(self.device)
            model.eval()

            self._models[lang] = model
            self._tokenizers[lang] = tokenizer

        return self._tokenizers[lang], self._models[lang]

    @staticmethod
    def supported_languages() -> dict:
        return LANGUAGE_NAMES


# ──────────────────────────────────────────────
# COMBINED OUTPUT HANDLER
# ──────────────────────────────────────────────
@dataclass
class OutputResult:
    content: str
    language: str
    language_name: str
    content_type: str    # "summary" | "answer"


class OutputHandler:
    """
    Combines summarizer + translator.
    Implements both branches of Row 2 in your diagram.
    """

    def __init__(self, summarizer_model: str = "bart"):
        self.summarizer = PolicySummarizer(model_key=summarizer_model)
        self.translator = MultilingualTranslator()

    def get_summary(self, text: str, language: str = "en") -> OutputResult:
        summary = self.summarizer.summarize(text)
        if language != "en":
            summary = self.translator.translate(summary, language)
        return OutputResult(
            content=summary,
            language=language,
            language_name=LANGUAGE_NAMES.get(language, language),
            content_type="summary",
        )

    def get_answer(self, answer: str, language: str = "en") -> OutputResult:
        if language != "en":
            answer = self.translator.translate(answer, language)
        return OutputResult(
            content=answer,
            language=language,
            language_name=LANGUAGE_NAMES.get(language, language),
            content_type="answer",
        )

    @staticmethod
    def available_languages() -> dict:
        return LANGUAGE_NAMES


# ── Quick test ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    handler = OutputHandler()
    sample = """
    The insurance policy covers all accidental damage to the insured vehicle.
    Third-party liability is included up to $500,000. Personal injury protection
    pays up to $10,000 per person per accident. The deductible is $500 for collision
    claims and $250 for comprehensive claims. Policy expires on December 31, 2024.
    Renewal requires a new medical exam for drivers over 70 years of age.
    """
    print("=== Summary (English) ===")
    result = handler.get_summary(sample, language="en")
    print(result.content)

    print("\n=== Summary (French) ===")
    result = handler.get_summary(sample, language="fr")
    print(result.content)
