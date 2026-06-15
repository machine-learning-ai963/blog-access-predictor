# ブログ記事アクセス数予測アプリ

Streamlit上で、ブログ記事データの抽出、CSV作成、Gemini embedding化、機械学習、アクセス数予測まで行うアプリです。

## できること

1. `https://ai-fukushi.net/archive/` から記事データを抽出
2. Streamlit画面上で `articles.csv` を作成してダウンロード
3. Gemini embeddingで記事本文をベクトル化
4. 累計アクセス数を目的変数として機械学習
5. 新しい記事本文を入力して、予測累計アクセス数を表示

## GitHubにアップロードするファイル

このフォルダの中身を、そのままGitHubリポジトリにアップロードしてください。

```text
app.py
requirements.txt
runtime.txt
README.md
src/
data/
artifacts/
```

`.github` や `.streamlit` などの隠しフォルダは必須ではありません。  
GitHub Actionsは使わず、Streamlit画面上で処理します。

## Streamlit Cloudで公開する手順

1. GitHubに新しいリポジトリを作る
2. このフォルダの中身をアップロードする
3. Streamlit Community Cloudで `New app` を押す
4. Repositoryに今回のGitHubリポジトリを指定する
5. Main file pathに `app.py` を指定する
6. Advanced settings または App settings の Secrets に以下を入れる

```toml
GEMINI_API_KEY = "あなたのGemini APIキー"
```

7. Deployを押す

## 使い方

### 1. 記事抽出・CSV作成

最初は `取得ページ数` を `1` にして実行してください。  
成功したら `13` に増やします。

抽出後、画面に一覧が表示され、`CSVをダウンロード` ボタンから `articles.csv` を保存できます。

### 2. embedding化

Gemini APIキーが読み込めている状態で、`embedding化を実行` を押します。  
完了すると、embedding済みCSVとParquetをダウンロードできます。

### 3. 学習

`学習を実行` を押すと、累計アクセス数を予測するモデルを作ります。  
目的変数は `view_count` です。  
公開からの日数 `days_since_publish` も特徴量に入れるため、古い記事ほどアクセス数が増えやすい問題をある程度補正します。

### 4. 予測

新しい記事タイトルと本文を入力すると、Gemini embeddingを作成し、予測累計アクセス数を表示します。

## 注意点

Streamlit Cloud上で作ったCSVやモデルは、アプリの再起動やスリープで消えることがあります。  
必要なCSV、embedding済みデータ、学習済みモデルは、必ず画面上のボタンからダウンロードしてください。

また、スクレイピング対象サイトに負荷をかけないよう、取得間隔は `1.0秒` 以上がおすすめです。
