from __future__ import annotations

import json
from io import BytesIO

import joblib
import numpy as np
import pandas as pd
import streamlit as st

from src.features import cosine_similarity_matrix, make_single_feature_row
from src.gemini_embedder import GeminiEmbedder
from src.modeling import (
    bundle_to_joblib_bytes,
    format_views,
    metadata_to_json_bytes,
    train_model_from_embedded_df,
)
from src.scraper import dataframe_to_csv_bytes, scrape_articles
from src.secrets import get_gemini_api_key
from src.settings import DEFAULT_ARCHIVE_URL, DEFAULT_MODEL_PATH, DEFAULT_SITE_ROOT
from src.text_utils import build_embedding_text

st.set_page_config(
    page_title="ブログ記事アクセス数予測アプリ",
    page_icon="📈",
    layout="wide",
)


for key, default in {
    "articles_df": None,
    "embedded_df": None,
    "model_bundle": None,
    "model_metadata": None,
}.items():
    if key not in st.session_state:
        st.session_state[key] = default


@st.cache_resource(show_spinner=False)
def load_committed_model_if_exists():
    if DEFAULT_MODEL_PATH.exists():
        return joblib.load(DEFAULT_MODEL_PATH)
    return None


def get_active_bundle():
    if st.session_state.get("model_bundle") is not None:
        return st.session_state["model_bundle"]
    return load_committed_model_if_exists()


def get_parquet_bytes(df: pd.DataFrame) -> bytes:
    buffer = BytesIO()
    df.to_parquet(buffer, index=False)
    buffer.seek(0)
    return buffer.read()


def get_embedded_csv_bytes(df: pd.DataFrame) -> bytes:
    out = df.copy()
    out["embedding"] = out["embedding"].apply(lambda x: json.dumps(x, ensure_ascii=False) if isinstance(x, list) else x)
    return out.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig")


st.title("ブログ記事アクセス数予測アプリ")
st.caption("Streamlit上で記事データ抽出 → CSV作成 → Gemini embedding化 → 機械学習 → 予測まで行います。")

with st.sidebar:
    st.header("設定")
    api_key_input = st.text_input(
        "Gemini APIキー（Secrets未設定時のみ）",
        type="password",
        help="Streamlit CloudのSecretsに GEMINI_API_KEY を設定している場合は空欄でOKです。",
    )
    api_key = get_gemini_api_key(api_key_input)
    if api_key:
        st.success("Gemini APIキーを読み込めています。")
    else:
        st.warning("Gemini APIキーが未設定です。embedding化と予測には必要です。")

    st.divider()
    st.caption("注意：Streamlit Cloud上で作ったCSVやモデルは、アプリ再起動時に消える場合があります。必要なファイルはダウンロードしてください。")


tab1, tab2, tab3, tab4 = st.tabs([
    "1. 記事抽出・CSV作成",
    "2. embedding化",
    "3. 学習",
    "4. 予測",
])

with tab1:
    st.subheader("1. ブログ記事データを抽出してCSVを作る")
    st.markdown(
        "一覧ページから記事URL・タイトル・概要・公開日・カテゴリ・表示アクセス数を取り、個別記事ページから本文を取得します。"
    )

    col_a, col_b = st.columns([2, 1])
    with col_a:
        archive_url = st.text_input("アーカイブページURL", value=DEFAULT_ARCHIVE_URL)
        site_root = st.text_input("サイトURL", value=DEFAULT_SITE_ROOT)
    with col_b:
        max_pages = st.number_input("取得ページ数", min_value=1, max_value=50, value=1, step=1)
        sleep_seconds = st.number_input("取得間隔 秒", min_value=0.0, max_value=10.0, value=1.0, step=0.5)

    st.info("最初は取得ページ数を1にして試し、問題なければ13に増やしてください。")

    run_scrape = st.button("記事データを抽出してCSVを作る", type="primary")

    if run_scrape:
        status = st.empty()
        progress = st.progress(0)

        def progress_callback(kind, current, total, message):
            status.write(message)
            if kind == "list" and total:
                progress.progress(min(current / total, 1.0))

        try:
            with st.spinner("記事データを取得しています。ページ数が多いと数分かかります..."):
                df = scrape_articles(
                    archive_url=archive_url,
                    site_root=site_root,
                    max_pages=int(max_pages),
                    sleep_seconds=float(sleep_seconds),
                    progress_callback=progress_callback,
                )
            st.session_state["articles_df"] = df
            st.success(f"CSV作成用データを取得しました。取得件数: {len(df)}件")
            progress.progress(1.0)
        except Exception as exc:
            st.error(f"記事データの取得に失敗しました: {exc}")

    articles_df = st.session_state.get("articles_df")
    if articles_df is not None:
        st.metric("取得済み記事数", f"{len(articles_df)}件")
        st.dataframe(
            articles_df[["title", "published_date", "category", "view_count", "days_since_publish", "url"]],
            use_container_width=True,
            hide_index=True,
        )
        st.download_button(
            "CSVをダウンロード",
            data=dataframe_to_csv_bytes(articles_df),
            file_name="articles.csv",
            mime="text/csv",
        )

        with st.expander("CSVの列を確認する"):
            st.write(list(articles_df.columns))
            st.dataframe(articles_df.head(3), use_container_width=True)

with tab2:
    st.subheader("2. Gemini embedding化")
    st.markdown("手順1で作った記事データを、Gemini embeddingでベクトル化します。")

    articles_df = st.session_state.get("articles_df")
    if articles_df is None:
        st.warning("先に『1. 記事抽出・CSV作成』でCSVデータを作ってください。")
    elif not api_key:
        st.warning("Gemini APIキーが必要です。Streamlit Secretsに設定するか、サイドバーに入力してください。")
    else:
        col_e1, col_e2 = st.columns([1, 1])
        with col_e1:
            embed_sleep = st.number_input("embedding API呼び出し間隔 秒", min_value=0.0, max_value=5.0, value=0.5, step=0.1)
        with col_e2:
            max_embed_rows = st.number_input(
                "embedding化する最大件数",
                min_value=1,
                max_value=max(len(articles_df), 1),
                value=len(articles_df),
                step=1,
            )

        if st.button("embedding化を実行", type="primary"):
            work_df = articles_df.head(int(max_embed_rows)).copy()
            progress = st.progress(0)
            message = st.empty()

            try:
                embedder = GeminiEmbedder(api_key=api_key, sleep_seconds=float(embed_sleep))
                texts = []
                for _, row in work_df.iterrows():
                    texts.append(
                        build_embedding_text(
                            title=str(row.get("title", "")),
                            category=str(row.get("category", "")),
                            excerpt=str(row.get("excerpt", "")),
                            body_text=str(row.get("body_text", "")),
                        )
                    )

                def progress_callback(i, total):
                    progress.progress(i / total)
                    title = str(work_df.iloc[i - 1].get("title", ""))[:60]
                    message.write(f"embedding作成中: {i}/{total}件 - {title}")

                with st.spinner("Gemini embeddingを作成しています..."):
                    embeddings = embedder.embed_many(texts, progress_callback=progress_callback)

                work_df["embedding"] = embeddings
                st.session_state["embedded_df"] = work_df
                st.success(f"embedding化が完了しました。件数: {len(work_df)}件")
            except Exception as exc:
                st.error(f"embedding化に失敗しました: {exc}")

    embedded_df = st.session_state.get("embedded_df")
    if embedded_df is not None:
        st.metric("embedding済み記事数", f"{len(embedded_df)}件")
        first_emb = embedded_df.iloc[0]["embedding"]
        st.metric("embedding次元数", len(first_emb) if isinstance(first_emb, list) else "不明")

        col_d1, col_d2 = st.columns(2)
        with col_d1:
            st.download_button(
                "embedding済みCSVをダウンロード",
                data=get_embedded_csv_bytes(embedded_df),
                file_name="articles_with_embeddings.csv",
                mime="text/csv",
            )
        with col_d2:
            st.download_button(
                "embedding済みParquetをダウンロード",
                data=get_parquet_bytes(embedded_df),
                file_name="articles_with_embeddings.parquet",
                mime="application/octet-stream",
            )

        st.dataframe(
            embedded_df[["title", "category", "view_count", "days_since_publish", "body_length"]],
            use_container_width=True,
            hide_index=True,
        )

with tab3:
    st.subheader("3. 機械学習モデルを作る")
    st.markdown("embedding済みデータを使い、累計アクセス数を予測するモデルを学習します。")

    embedded_df = st.session_state.get("embedded_df")
    if embedded_df is None:
        st.warning("先に『2. embedding化』を実行してください。")
    else:
        if st.button("学習を実行", type="primary"):
            try:
                with st.spinner("モデルを学習しています..."):
                    bundle, metadata = train_model_from_embedded_df(embedded_df)
                st.session_state["model_bundle"] = bundle
                st.session_state["model_metadata"] = metadata
                st.success("学習が完了しました。")
            except Exception as exc:
                st.error(f"学習に失敗しました: {exc}")

    bundle = st.session_state.get("model_bundle")
    metadata = st.session_state.get("model_metadata")
    if bundle is not None:
        st.metric("学習記事数", f"{len(bundle['reference_df'])}件")
        st.metric("embedding次元数", f"{len(bundle['embedding_cols'])}")
        if metadata:
            st.json(metadata)

        col_m1, col_m2 = st.columns(2)
        with col_m1:
            st.download_button(
                "学習済みモデルをダウンロード",
                data=bundle_to_joblib_bytes(bundle),
                file_name="model_bundle.joblib",
                mime="application/octet-stream",
            )
        with col_m2:
            st.download_button(
                "モデル情報JSONをダウンロード",
                data=metadata_to_json_bytes(metadata or {}),
                file_name="metadata.json",
                mime="application/json",
            )

with tab4:
    st.subheader("4. 新しい記事のアクセス数を予測する")
    bundle = get_active_bundle()

    if bundle is None:
        st.warning("まだモデルがありません。先に『3. 学習』を実行してください。")
    elif not api_key:
        st.warning("予測する記事本文もembedding化するため、Gemini APIキーが必要です。")
    else:
        pipeline = bundle["pipeline"]
        embedding_cols = bundle["embedding_cols"]
        reference_df = bundle["reference_df"]

        categories = sorted([str(c) for c in reference_df["category"].dropna().unique().tolist()])
        if "未分類" not in categories:
            categories.insert(0, "未分類")

        col_p1, col_p2 = st.columns([2, 1])
        with col_p1:
            pred_title = st.text_input("記事タイトル", placeholder="例：障害者雇用の現状と企業に求められる準備")
            pred_body = st.text_area("記事本文", height=360, placeholder="ここに記事本文を貼り付けてください。")
        with col_p2:
            pred_category = st.selectbox("カテゴリ", categories)
            pred_days = st.number_input(
                "公開からの日数",
                min_value=1,
                value=30,
                step=1,
                help="案1なので、累計アクセス数予測の補正として使います。公開30日後の見込みなら30を入れます。",
            )
            st.metric("学習記事数", f"{len(reference_df)}件")
            st.metric("embedding次元数", f"{len(embedding_cols)}")

        if st.button("アクセス数を予測する", type="primary"):
            if not pred_title.strip() and not pred_body.strip():
                st.error("タイトルか本文を入力してください。")
                st.stop()

            try:
                with st.spinner("Gemini embeddingを作成して予測しています..."):
                    embedder = GeminiEmbedder(api_key=api_key, sleep_seconds=0.0)
                    embedding_text = build_embedding_text(
                        title=pred_title,
                        category=pred_category,
                        excerpt="",
                        body_text=pred_body,
                    )
                    embedding = embedder.embed_text(embedding_text)

                    X_one = make_single_feature_row(
                        title=pred_title,
                        body_text=pred_body,
                        category=pred_category,
                        days_since_publish=int(pred_days),
                        embedding=embedding,
                        embedding_cols=embedding_cols,
                    )

                    pred_log = float(pipeline.predict(X_one)[0])
                    pred_views = max(float(np.expm1(pred_log)), 0.0)

                st.success("予測が完了しました。")
                st.metric("予測累計アクセス数", format_views(pred_views))
                st.caption(f"目安レンジ：{format_views(pred_views * 0.7)} 〜 {format_views(pred_views * 1.3)}")

                ref_embeddings = reference_df[embedding_cols].to_numpy(dtype=float)
                query = np.array(embedding, dtype=float)
                similarities = cosine_similarity_matrix(query, ref_embeddings)

                similar_df = reference_df.copy()
                similar_df["similarity"] = similarities
                similar_df = similar_df.sort_values("similarity", ascending=False).head(5)

                st.subheader("内容が近い過去記事")
                show_cols = ["title", "category", "published_date", "view_count", "days_since_publish", "similarity", "url"]
                st.dataframe(
                    similar_df[show_cols],
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        "title": "タイトル",
                        "category": "カテゴリ",
                        "published_date": "公開日",
                        "view_count": st.column_config.NumberColumn("累計アクセス数", format="%d"),
                        "days_since_publish": "公開からの日数",
                        "similarity": st.column_config.NumberColumn("類似度", format="%.3f"),
                        "url": st.column_config.LinkColumn("URL"),
                    },
                )
            except Exception as exc:
                st.error(f"予測に失敗しました: {exc}")

st.divider()
st.caption("注意：この予測は過去記事データに基づく目安です。検索順位、SNS拡散、公開時期、ニュース性などで実際のアクセス数は大きく変動します。")
