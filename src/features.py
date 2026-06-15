from __future__ import annotations

import ast
import json
from typing import Any

import numpy as np
import pandas as pd


def parse_embedding(value: Any) -> list[float]:
    if isinstance(value, list):
        return [float(x) for x in value]
    if isinstance(value, np.ndarray):
        return [float(x) for x in value.tolist()]
    if isinstance(value, str):
        value = value.strip()
        if not value:
            return []
        try:
            return [float(x) for x in json.loads(value)]
        except Exception:
            return [float(x) for x in ast.literal_eval(value)]
    return []


def expand_embedding_columns(df: pd.DataFrame, embedding_col: str = "embedding") -> tuple[pd.DataFrame, list[str]]:
    if embedding_col not in df.columns:
        raise ValueError("embedding列がありません。先にGemini embedding化を実行してください。")

    embeddings = df[embedding_col].apply(parse_embedding).tolist()
    if not embeddings or not embeddings[0]:
        raise ValueError("embeddingが空です。embedding作成処理を確認してください。")

    dim = len(embeddings[0])
    for i, emb in enumerate(embeddings):
        if len(emb) != dim:
            raise ValueError(f"embeddingの次元数が揃っていません。row={i}, dim={len(emb)}, expected={dim}")

    emb_cols = [f"emb_{i}" for i in range(dim)]
    emb_df = pd.DataFrame(embeddings, columns=emb_cols, index=df.index)
    out = pd.concat([df.drop(columns=[embedding_col]), emb_df], axis=1)
    return out, emb_cols


def make_single_feature_row(
    *,
    title: str,
    body_text: str,
    category: str,
    days_since_publish: int,
    embedding: list[float],
    embedding_cols: list[str],
) -> pd.DataFrame:
    if len(embedding) != len(embedding_cols):
        raise ValueError(
            f"入力記事のembedding次元数が学習時と違います。input={len(embedding)}, trained={len(embedding_cols)}"
        )

    row = {
        "title": title,
        "body_text": body_text,
        "category": category or "未分類",
        "days_since_publish": max(int(days_since_publish), 1),
        "body_length": len(body_text or ""),
    }
    row.update({col: float(val) for col, val in zip(embedding_cols, embedding)})
    return pd.DataFrame([row])


def cosine_similarity_matrix(query: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    query_norm = np.linalg.norm(query)
    matrix_norm = np.linalg.norm(matrix, axis=1)
    denom = np.maximum(query_norm * matrix_norm, 1e-12)
    return matrix @ query / denom
