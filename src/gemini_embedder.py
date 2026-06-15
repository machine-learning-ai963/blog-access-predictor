from __future__ import annotations

import os
import time
from typing import Iterable

from google import genai

from src.settings import DEFAULT_EMBEDDING_MODEL, MAX_EMBEDDING_BYTES
from src.text_utils import truncate_by_bytes


class GeminiEmbedder:
    def __init__(self, api_key: str, model: str | None = None, sleep_seconds: float = 0.5):
        if not api_key:
            raise ValueError("GEMINI_API_KEY が設定されていません。Streamlit Cloud の Secrets に設定してください。")
        self.client = genai.Client(api_key=api_key)
        self.model = model or os.getenv("EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL)
        self.sleep_seconds = sleep_seconds

    def embed_text(self, text: str) -> list[float]:
        safe_text = truncate_by_bytes(text or "", MAX_EMBEDDING_BYTES)
        if not safe_text.strip():
            safe_text = "空の本文"

        result = self.client.models.embed_content(
            model=self.model,
            contents=safe_text,
        )
        values = self._extract_values(result)

        if self.sleep_seconds > 0:
            time.sleep(self.sleep_seconds)
        return values

    def embed_many(self, texts: Iterable[str], progress_callback=None) -> list[list[float]]:
        embeddings: list[list[float]] = []
        texts = list(texts)
        total = len(texts)
        for i, text in enumerate(texts, start=1):
            embeddings.append(self.embed_text(text))
            if progress_callback:
                progress_callback(i, total)
        return embeddings

    @staticmethod
    def _extract_values(result) -> list[float]:
        embeddings = getattr(result, "embeddings", None)
        if embeddings:
            first = embeddings[0]
            values = getattr(first, "values", None)
            if values is not None:
                return list(values)
            if isinstance(first, dict):
                if "values" in first:
                    return list(first["values"])
                if "embedding" in first and "values" in first["embedding"]:
                    return list(first["embedding"]["values"])

        embedding = getattr(result, "embedding", None)
        if embedding is not None:
            values = getattr(embedding, "values", None)
            if values is not None:
                return list(values)
            if isinstance(embedding, dict) and "values" in embedding:
                return list(embedding["values"])

        raise RuntimeError(f"embedding の戻り値を解釈できませんでした: {result}")
