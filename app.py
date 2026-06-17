from __future__ import annotations

import io
import json
import math
import re
import time
from datetime import date
from typing import Any
from urllib.parse import urljoin, urlparse

import joblib
import numpy as np
import pandas as pd
import requests
import streamlit as st
from bs4 import BeautifulSoup
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


SITE_ROOT = "https://ai-fukushi.net"
DEFAULT_ARCHIVE_URL = "https://ai-fukushi.net/archive/"
DEFAULT_EMBED_MODEL = "gemini-embedding-2"

st.set_page_config(
    page_title="ブログPV予測 Gemini Embedding",
    page_icon="📈",
    layout="wide",
)


# -----------------------------
# 共通ユーティリティ
# -----------------------------

def clean_text(text: str | None) -> str:
    if not text:
        return ""
    return re.sub(r"\s+", " ", text).strip()


def get_secret_api_key() -> str:
    try:
        return st.secrets.get("GEMINI_API_KEY", "")
    except Exception:
        return ""


def get_api_key_from_ui() -> str:
    secret_key = get_secret_api_key()
    with st.sidebar:
        st.subheader("Gemini APIキー")
        if secret_key:
            st.success("Streamlit SecretsのGEMINI_API_KEYを使用します。")
            return secret_key
        api_key = st.text_input(
            "GEMINI_API_KEY",
            type="password",
            help="Streamlit Secretsに入れていない場合だけ、ここに一時入力してください。",
        )
        return api_key.strip()


def fetch_html(url: str) -> str:
    headers = {
        "User-Agent": "Mozilla/5.0 (compatible; blog-pv-research/1.0; +https://streamlit.io)"
    }
    res = requests.get(url, headers=headers, timeout=30)
    res.raise_for_status()
    res.encoding = res.apparent_encoding
    return res.text


def archive_page_url(base_url: str, page: int) -> str:
    base_url = base_url.rstrip("/") + "/"
    if page == 1:
        return base_url
    return urljoin(base_url, f"page/{page}/")


def is_article_url(url: str) -> bool:
    parsed = urlparse(url)
    if parsed.netloc and parsed.netloc != "ai-fukushi.net":
        return False
    path = parsed.path.strip("/")
    if not path:
        return False
    skip_prefixes = (
        "archive",
        "category",
        "tag",
        "author",
        "page",
        "privacy",
        "terms",
        "wp-content",
        "wp-admin",
        "contact",
    )
    return not any(path.startswith(prefix) for prefix in skip_prefixes)


def parse_archive_item(item: Any) -> dict[str, Any] | None:
    link = item.select_one("a[href]")
    if not link:
        return None

    url = urljoin(SITE_ROOT, link.get("href", ""))
    if not is_article_url(url):
        return None

    title_el = item.select_one(".p-postList__title, .entry-title, h2, h3")
    excerpt_el = item.select_one(".p-postList__excerpt, .entry-summary, .excerpt")
    category_el = item.select_one(".p-postList__cat, .cat, .category")

    title = clean_text(title_el.get_text(" ", strip=True)) if title_el else clean_text(link.get_text(" ", strip=True))
    excerpt = clean_text(excerpt_el.get_text(" ", strip=True)) if excerpt_el else ""
    category = clean_text(category_el.get_text(" ", strip=True)) if category_el else ""

    all_text = clean_text(item.get_text(" ", strip=True))
    dates = re.findall(r"\d{4}-\d{2}-\d{2}", all_text)
    published_date = dates[-1] if dates else ""

    view_count = None
    if published_date and published_date in all_text:
        after_date = all_text.split(published_date)[-1].strip()
        view_match = re.search(r"([0-9,]+)\s*$", after_date)
        if view_match:
            view_count = int(view_match.group(1).replace(",", ""))
            if not category:
                category = clean_text(after_date[: view_match.start()])

    if view_count is None:
        nums = re.findall(r"(?<!\d)([0-9][0-9,]{1,})(?!\d)", all_text)
        if nums:
            try:
                view_count = int(nums[-1].replace(",", ""))
            except Exception:
                view_count = None

    if not title or view_count is None:
        return None

    return {
        "url": url,
        "title": title,
        "excerpt": excerpt,
        "published_date": published_date,
        "category": category,
        "view_count": view_count,
    }


def parse_archive_page(html: str) -> list[dict[str, Any]]:
    soup = BeautifulSoup(html, "html.parser")
    main = soup.select_one("main") or soup

    items = main.select("li.p-postList__item, article")
    if not items:
        items = main.select("li, div")

    records: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in items:
        data = parse_archive_item(item)
        if not data:
            continue
        if data["url"] in seen:
            continue
        seen.add(data["url"])
        records.append(data)
    return records


def parse_article_body(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup.select(
        "script, style, noscript, iframe, nav, aside, form, "
        ".p-toc, .c-shareBtns, .p-shareBtns, .p-adBox, .p-authorBox, "
        "header, footer"
    ):
        tag.decompose()

    body_area = (
        soup.select_one(".post_content")
        or soup.select_one(".c-entry__content")
        or soup.select_one(".entry-content")
        or soup.select_one("article")
        or soup.select_one("main")
        or soup
    )
    text = body_area.get_text("\n", strip=True)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def calc_days_since_publish(published_date: str) -> int:
    try:
        d = pd.to_datetime(published_date).date()
        return max((date.today() - d).days, 1)
    except Exception:
        return 1


def scrape_articles(base_url: str, max_pages: int, sleep_sec: float) -> pd.DataFrame:
    all_records: list[dict[str, Any]] = []
    seen_urls: set[str] = set()

    progress = st.progress(0)
    status = st.empty()

    total_steps = max_pages
    for page in range(1, max_pages + 1):
        list_url = archive_page_url(base_url, page)
        status.info(f"一覧ページ取得中: {page}/{max_pages}  {list_url}")
        html = fetch_html(list_url)
        records = parse_archive_page(html)

        for i, record in enumerate(records, start=1):
            if record["url"] in seen_urls:
                continue
            seen_urls.add(record["url"])
            status.info(f"本文取得中: ページ{page} / {i}/{len(records)}  {record['title'][:40]}")
            try:
                article_html = fetch_html(record["url"])
                body_text = parse_article_body(article_html)
                record["body_text"] = body_text
                record["body_length"] = len(body_text)
                record["days_since_publish"] = calc_days_since_publish(record.get("published_date", ""))
                all_records.append(record)
            except Exception as e:
                record["body_text"] = ""
                record["body_length"] = 0
                record["days_since_publish"] = calc_days_since_publish(record.get("published_date", ""))
                record["error"] = str(e)
                all_records.append(record)
            time.sleep(sleep_sec)

        progress.progress(page / total_steps)
        time.sleep(sleep_sec)

    status.success("記事抽出が完了しました。")
    df = pd.DataFrame(all_records)
    if df.empty:
        return df

    columns = [
        "url",
        "title",
        "excerpt",
        "published_date",
        "category",
        "view_count",
        "days_since_publish",
        "body_length",
        "body_text",
    ]
    for col in columns:
        if col not in df.columns:
            df[col] = "" if col not in ["view_count", "days_since_publish", "body_length"] else 0
    return df[columns]


def csv_download_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig")


def read_uploaded_csv(uploaded_file: Any) -> pd.DataFrame:
    return pd.read_csv(uploaded_file)


# -----------------------------
# Gemini Embedding
# -----------------------------

def make_embedding_text(row: pd.Series) -> str:
    title = str(row.get("title", ""))
    category = str(row.get("category", ""))
    excerpt = str(row.get("excerpt", ""))
    body = str(row.get("body_text", ""))
    # Gemini embeddingの入力上限に当たりにくくするため短めに切る
    body = body[:12000]
    return f"タイトル: {title}\nカテゴリ: {category}\n概要: {excerpt}\n本文:\n{body}"


def extract_embedding_values(response: Any) -> list[float]:
    if hasattr(response, "embeddings") and response.embeddings:
        first = response.embeddings[0]
        values = getattr(first, "values", None)
        if values is not None:
            return list(values)
        if isinstance(first, dict) and "values" in first:
            return list(first["values"])

    if hasattr(response, "embedding"):
        emb = response.embedding
        values = getattr(emb, "values", None)
        if values is not None:
            return list(values)
        if isinstance(emb, dict) and "values" in emb:
            return list(emb["values"])

    raise ValueError("Gemini embeddingのレスポンス形式を解析できませんでした。")


def embed_one_text(api_key: str, text: str, model_name: str) -> list[float]:
    from google import genai

    client = genai.Client(api_key=api_key)
    response = client.models.embed_content(model=model_name, contents=text)
    return extract_embedding_values(response)


def create_embeddings(df: pd.DataFrame, api_key: str, model_name: str, sleep_sec: float) -> pd.DataFrame:
    if not api_key:
        raise ValueError("Gemini APIキーが未設定です。Streamlit Secretsまたはサイドバーに入力してください。")

    df = df.copy()
    embeddings: list[str] = []
    progress = st.progress(0)
    status = st.empty()

    total = len(df)
    for idx, row in df.iterrows():
        title = str(row.get("title", ""))[:40]
        status.info(f"Embedding作成中: {len(embeddings)+1}/{total}  {title}")
        text = make_embedding_text(row)
        values = embed_one_text(api_key, text, model_name)
        embeddings.append(json.dumps(values, ensure_ascii=False))
        progress.progress((len(embeddings)) / max(total, 1))
        time.sleep(sleep_sec)

    df["embedding_json"] = embeddings
    status.success("Embedding作成が完了しました。")
    return df


# -----------------------------
# 機械学習
# -----------------------------

def parse_embedding_json(value: Any) -> list[float]:
    if isinstance(value, list):
        return [float(x) for x in value]
    if pd.isna(value):
        return []
    if isinstance(value, str):
        return [float(x) for x in json.loads(value)]
    return []


def build_feature_frame(df: pd.DataFrame, expected_columns: list[str] | None = None) -> pd.DataFrame:
    embeddings = df["embedding_json"].apply(parse_embedding_json).tolist()
    if not embeddings or not embeddings[0]:
        raise ValueError("embedding_json列が空です。先にGemini embeddingを作成してください。")

    max_len = max(len(x) for x in embeddings)
    emb_matrix = np.array([x + [0.0] * (max_len - len(x)) for x in embeddings], dtype=float)
    emb_cols = [f"emb_{i}" for i in range(emb_matrix.shape[1])]
    emb_df = pd.DataFrame(emb_matrix, columns=emb_cols)

    numeric = pd.DataFrame(
        {
            "days_since_publish": pd.to_numeric(df.get("days_since_publish", 1), errors="coerce").fillna(1),
            "body_length": pd.to_numeric(df.get("body_length", 0), errors="coerce").fillna(0),
            "title_length": df.get("title", "").astype(str).str.len(),
        }
    )

    category = df.get("category", pd.Series([""] * len(df))).fillna("").astype(str)
    cat_df = pd.get_dummies(category, prefix="category")

    X = pd.concat([emb_df, numeric, cat_df], axis=1)

    if expected_columns is not None:
        for col in expected_columns:
            if col not in X.columns:
                X[col] = 0
        X = X[expected_columns]

    return X


def train_model(df: pd.DataFrame) -> tuple[dict[str, Any], dict[str, Any]]:
    required = {"view_count", "embedding_json"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"必要な列がありません: {', '.join(sorted(missing))}")

    work = df.copy()
    work["view_count"] = pd.to_numeric(work["view_count"], errors="coerce")
    work = work.dropna(subset=["view_count", "embedding_json"])
    work = work[work["view_count"] >= 0]

    if len(work) < 5:
        raise ValueError("学習データが少なすぎます。最低5件以上の記事が必要です。")

    X = build_feature_frame(work)
    y = np.log1p(work["view_count"].astype(float).values)

    scaler = StandardScaler(with_mean=False)
    X_scaled = scaler.fit_transform(X)

    model = Ridge(alpha=10.0, random_state=42)

    metrics: dict[str, Any] = {"n_samples": int(len(work)), "n_features": int(X.shape[1])}
    if len(work) >= 10:
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.25, random_state=42
        )
        model.fit(X_train, y_train)
        pred_log = model.predict(X_test)
        pred = np.expm1(pred_log)
        actual = np.expm1(y_test)
        metrics["mae"] = float(mean_absolute_error(actual, pred))
        try:
            metrics["r2_log"] = float(r2_score(y_test, pred_log))
        except Exception:
            metrics["r2_log"] = None
        model.fit(X_scaled, y)
    else:
        model.fit(X_scaled, y)
        metrics["mae"] = None
        metrics["r2_log"] = None

    bundle = {
        "model": model,
        "scaler": scaler,
        "feature_columns": list(X.columns),
        "training_columns": list(work.columns),
        "metrics": metrics,
        "created_at": pd.Timestamp.now().isoformat(),
        "target": "log1p(view_count)",
    }
    return bundle, metrics


def model_to_bytes(bundle: dict[str, Any]) -> bytes:
    buf = io.BytesIO()
    joblib.dump(bundle, buf)
    return buf.getvalue()


def bytes_to_model(uploaded_file: Any) -> dict[str, Any]:
    return joblib.load(uploaded_file)


def predict_view_count(
    bundle: dict[str, Any],
    api_key: str,
    model_name: str,
    title: str,
    category: str,
    body_text: str,
    days_since_publish: int,
) -> float:
    row = pd.DataFrame(
        [
            {
                "title": title,
                "category": category,
                "excerpt": "",
                "body_text": body_text,
                "body_length": len(body_text),
                "days_since_publish": days_since_publish,
            }
        ]
    )
    text = make_embedding_text(row.iloc[0])
    emb = embed_one_text(api_key, text, model_name)
    row["embedding_json"] = [json.dumps(emb, ensure_ascii=False)]

    X = build_feature_frame(row, expected_columns=bundle["feature_columns"])
    X_scaled = bundle["scaler"].transform(X)
    pred_log = bundle["model"].predict(X_scaled)[0]
    pred = float(np.expm1(pred_log))
    if math.isnan(pred) or pred < 0:
        return 0.0
    return pred


# -----------------------------
# UI
# -----------------------------

st.title("ブログ記事アクセス数予測アプリ")
st.caption("CSV軽量版：Streamlit上で記事抽出、CSV作成、Gemini embedding、機械学習、予測まで行います。")

api_key = get_api_key_from_ui()

with st.sidebar:
    st.divider()
    st.caption("最初は取得ページ数1でテストしてください。問題なければ13に増やします。")

for key in ["articles_df", "embedded_df", "model_bundle"]:
    if key not in st.session_state:
        st.session_state[key] = None


tab1, tab2, tab3, tab4 = st.tabs([
    "1 CSV作成",
    "2 Gemini embedding",
    "3 機械学習",
    "4 予測",
])

with tab1:
    st.header("1 記事抽出・CSV作成")
    base_url = st.text_input("アーカイブURL", value=DEFAULT_ARCHIVE_URL)
    col1, col2 = st.columns(2)
    with col1:
        max_pages = st.number_input("取得ページ数", min_value=1, max_value=30, value=1, step=1)
    with col2:
        sleep_sec = st.number_input("取得間隔 秒", min_value=0.0, max_value=5.0, value=1.0, step=0.5)

    if st.button("記事を抽出してCSVを作る", type="primary"):
        try:
            df = scrape_articles(base_url, int(max_pages), float(sleep_sec))
            st.session_state.articles_df = df
            if df.empty:
                st.warning("記事を取得できませんでした。アーカイブURLやサイト構造を確認してください。")
            else:
                st.success(f"{len(df)}件の記事を取得しました。")
        except Exception as e:
            st.error(f"記事抽出中にエラーが発生しました: {e}")

    df = st.session_state.articles_df
    if isinstance(df, pd.DataFrame) and not df.empty:
        st.dataframe(df[["title", "published_date", "category", "view_count", "body_length"]], use_container_width=True)
        st.download_button(
            "CSVをダウンロード",
            data=csv_download_bytes(df),
            file_name="articles.csv",
            mime="text/csv",
        )

with tab2:
    st.header("2 Gemini embedding作成")
    st.write("CSV作成タブで作ったデータ、またはアップロードしたCSVを使います。")

    uploaded_csv_for_embed = st.file_uploader("既存CSVを使う場合はこちらにアップロード", type=["csv"], key="embed_csv")
    source_df = None
    if uploaded_csv_for_embed is not None:
        try:
            source_df = read_uploaded_csv(uploaded_csv_for_embed)
            st.info(f"アップロードCSV: {len(source_df)}件")
        except Exception as e:
            st.error(f"CSVを読めませんでした: {e}")
    elif isinstance(st.session_state.articles_df, pd.DataFrame):
        source_df = st.session_state.articles_df

    col1, col2 = st.columns(2)
    with col1:
        embed_model = st.text_input("Embeddingモデル", value=DEFAULT_EMBED_MODEL)
    with col2:
        embed_sleep = st.number_input("API呼び出し間隔 秒", min_value=0.0, max_value=5.0, value=0.5, step=0.5)

    if source_df is not None:
        st.dataframe(source_df.head(5), use_container_width=True)

    if st.button("Gemini embeddingを作成", type="primary"):
        try:
            if source_df is None:
                st.warning("先にCSVを作成するか、CSVをアップロードしてください。")
            else:
                embedded_df = create_embeddings(source_df, api_key, embed_model, float(embed_sleep))
                st.session_state.embedded_df = embedded_df
                st.success(f"Embedding済みデータを作成しました: {len(embedded_df)}件")
        except Exception as e:
            st.error(f"Embedding作成中にエラーが発生しました: {e}")

    embedded_df = st.session_state.embedded_df
    if isinstance(embedded_df, pd.DataFrame) and not embedded_df.empty:
        show_cols = [c for c in ["title", "category", "view_count", "embedding_json"] if c in embedded_df.columns]
        st.dataframe(embedded_df[show_cols].head(5), use_container_width=True)
        st.download_button(
            "Embedding済みCSVをダウンロード",
            data=csv_download_bytes(embedded_df),
            file_name="articles_with_embeddings.csv",
            mime="text/csv",
        )

with tab3:
    st.header("3 機械学習")
    st.write("Embedding済みCSVを使って、累計アクセス数を予測するモデルを作ります。")

    uploaded_csv_for_train = st.file_uploader("Embedding済みCSVをアップロード", type=["csv"], key="train_csv")
    train_df = None
    if uploaded_csv_for_train is not None:
        try:
            train_df = read_uploaded_csv(uploaded_csv_for_train)
            st.info(f"アップロードCSV: {len(train_df)}件")
        except Exception as e:
            st.error(f"CSVを読めませんでした: {e}")
    elif isinstance(st.session_state.embedded_df, pd.DataFrame):
        train_df = st.session_state.embedded_df

    if train_df is not None:
        st.dataframe(train_df.head(5), use_container_width=True)

    if st.button("モデルを学習する", type="primary"):
        try:
            if train_df is None:
                st.warning("先にEmbedding済みCSVを作成するか、アップロードしてください。")
            else:
                bundle, metrics = train_model(train_df)
                st.session_state.model_bundle = bundle
                st.success("モデル学習が完了しました。")
                st.json(metrics)
        except Exception as e:
            st.error(f"学習中にエラーが発生しました: {e}")

    bundle = st.session_state.model_bundle
    if isinstance(bundle, dict):
        st.subheader("モデル情報")
        st.json(bundle.get("metrics", {}))
        st.download_button(
            "学習済みモデルをダウンロード",
            data=model_to_bytes(bundle),
            file_name="model_bundle.joblib",
            mime="application/octet-stream",
        )

with tab4:
    st.header("4 新しい記事のアクセス数を予測")

    uploaded_model = st.file_uploader("学習済みモデル joblib をアップロード", type=["joblib"], key="model_upload")
    predict_bundle = st.session_state.model_bundle
    if uploaded_model is not None:
        try:
            predict_bundle = bytes_to_model(uploaded_model)
            st.success("アップロードしたモデルを読み込みました。")
        except Exception as e:
            st.error(f"モデルを読めませんでした: {e}")

    pred_model_name = st.text_input("予測時のEmbeddingモデル", value=DEFAULT_EMBED_MODEL, key="pred_model")
    title = st.text_input("記事タイトル")
    category = st.text_input("カテゴリ", value="")
    days_since_publish = st.number_input("公開からの日数", min_value=1, max_value=3650, value=30, step=1)
    body_text = st.text_area("記事本文", height=300)

    if st.button("アクセス数を予測する", type="primary"):
        try:
            if not isinstance(predict_bundle, dict):
                st.warning("先にモデルを学習するか、学習済みモデルをアップロードしてください。")
            elif not api_key:
                st.warning("Gemini APIキーが必要です。")
            elif not title.strip() or not body_text.strip():
                st.warning("タイトルと本文を入力してください。")
            else:
                pred = predict_view_count(
                    predict_bundle,
                    api_key,
                    pred_model_name,
                    title,
                    category,
                    body_text,
                    int(days_since_publish),
                )
                st.metric("予測アクセス数", f"{pred:,.0f} PV")
                st.caption("小規模データでの推定なので、目安として見てください。")
        except Exception as e:
            st.error(f"予測中にエラーが発生しました: {e}")
