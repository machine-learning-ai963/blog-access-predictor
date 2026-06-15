from __future__ import annotations

import json
from datetime import datetime, timezone
from io import BytesIO

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.features import expand_embedding_columns
from src.settings import RANDOM_STATE


def train_model_from_embedded_df(df_raw: pd.DataFrame) -> tuple[dict, dict]:
    df, embedding_cols = expand_embedding_columns(df_raw, embedding_col="embedding")

    df = df.copy()
    df["category"] = df["category"].fillna("未分類").astype(str)
    df["days_since_publish"] = pd.to_numeric(df["days_since_publish"], errors="coerce").fillna(1).clip(lower=1)
    df["body_length"] = pd.to_numeric(df["body_length"], errors="coerce").fillna(0).clip(lower=0)
    df["view_count"] = pd.to_numeric(df["view_count"], errors="coerce")
    df = df.dropna(subset=["view_count"])

    if len(df) < 5:
        raise RuntimeError("学習に使える記事が少なすぎます。最低5件以上必要です。")

    feature_cols = ["days_since_publish", "body_length", "category"] + embedding_cols
    X = df[feature_cols]
    y_log = np.log1p(df["view_count"].astype(float))

    numeric_cols = ["days_since_publish", "body_length"]
    categorical_cols = ["category"]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
            ("emb", StandardScaler(), embedding_cols),
        ],
        remainder="drop",
    )

    pipeline = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("model", Ridge(alpha=10.0)),
        ]
    )

    metrics: dict[str, float | int | None] = {"n_rows": int(len(df))}

    if len(df) >= 10:
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y_log,
            test_size=0.2,
            random_state=RANDOM_STATE,
        )
        pipeline.fit(X_train, y_train)
        pred_log = pipeline.predict(X_test)
        pred = np.expm1(pred_log)
        actual = np.expm1(y_test)
        metrics["test_mae_views"] = float(mean_absolute_error(actual, pred))
        metrics["test_r2_log"] = float(r2_score(y_test, pred_log))
    else:
        metrics["test_mae_views"] = None
        metrics["test_r2_log"] = None

    pipeline.fit(X, y_log)

    reference_cols = [
        "url",
        "title",
        "published_date",
        "category",
        "view_count",
        "days_since_publish",
        "body_length",
    ]
    for col in reference_cols:
        if col not in df.columns:
            df[col] = ""

    reference_df = df[reference_cols + embedding_cols].copy()

    bundle = {
        "pipeline": pipeline,
        "feature_cols": feature_cols,
        "embedding_cols": embedding_cols,
        "numeric_cols": numeric_cols,
        "categorical_cols": categorical_cols,
        "reference_df": reference_df,
        "target": "log1p(view_count)",
        "prediction_is": "cumulative_view_count",
    }

    metadata = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "model": "Ridge regression on log1p(view_count)",
        "embedding_dim": len(embedding_cols),
        "feature_note": "Gemini embedding + category + body_length + days_since_publish",
        "prediction_note": "累計アクセス数を予測します。公開からの日数も特徴量に入れています。",
        "metrics": metrics,
    }

    return bundle, metadata


def bundle_to_joblib_bytes(bundle: dict) -> bytes:
    buffer = BytesIO()
    joblib.dump(bundle, buffer)
    buffer.seek(0)
    return buffer.read()


def metadata_to_json_bytes(metadata: dict) -> bytes:
    return json.dumps(metadata, ensure_ascii=False, indent=2).encode("utf-8")


def format_views(value: float) -> str:
    value = max(float(value), 0.0)
    return f"{value:,.0f} PV"
