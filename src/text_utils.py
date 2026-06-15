from __future__ import annotations

import re


def clean_text(text: str | None) -> str:
    if not text:
        return ""
    return re.sub(r"\s+", " ", str(text)).strip()


def normalize_multiline_text(text: str | None) -> str:
    if not text:
        return ""
    text = str(text).replace("\r\n", "\n").replace("\r", "\n")
    lines = [clean_text(line) for line in text.split("\n")]
    lines = [line for line in lines if line]
    return "\n".join(lines)


def truncate_by_bytes(text: str, max_bytes: int) -> str:
    data = text.encode("utf-8")
    if len(data) <= max_bytes:
        return text
    clipped = data[:max_bytes]
    return clipped.decode("utf-8", errors="ignore")


def build_embedding_text(*, title: str, category: str, excerpt: str, body_text: str) -> str:
    return normalize_multiline_text(
        f"""
タイトル: {title}
カテゴリ: {category}
概要: {excerpt}
本文:
{body_text}
"""
    )
