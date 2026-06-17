# ブログPV予測アプリ CSV軽量版

Streamlit Cloud上で、ブログ記事を抽出してCSVを作成し、Gemini embeddingと機械学習でアクセス数を予測するアプリです。

## 特徴

- pyarrow / parquet 不使用
- GitHub Actions 不使用
- 隠しフォルダ不要
- Streamlit画面上でCSV作成、embedding、学習、予測まで実行

## GitHubにアップロードするもの

以下をリポジトリ直下にアップロードしてください。

- app.py
- requirements.txt
- runtime.txt
- README.md

## Streamlit Cloud設定

Main file path は以下にします。

```text
app.py
```

Secretsには以下を登録してください。

```toml
GEMINI_API_KEY = "あなたのGemini APIキー"
```

## 使い方

1. 「1 CSV作成」で取得ページ数1から実行
2. CSVを確認してダウンロード
3. 「2 Gemini embedding」でembedding作成
4. 「3 機械学習」でモデル作成
5. 「4 予測」で新しい記事のアクセス数を予測

Streamlit Cloud上の一時データは再起動で消えることがあります。必要なCSVやモデルは必ずダウンロードしてください。
