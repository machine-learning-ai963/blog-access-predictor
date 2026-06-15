from __future__ import annotations

import re
import time
from datetime import date
from urllib.parse import urljoin, urlparse

import pandas as pd
import requests
from bs4 import BeautifulSoup

from src.settings import DEFAULT_ARCHIVE_URL, DEFAULT_SITE_ROOT
from src.text_utils import clean_text, normalize_multiline_text

HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; blog-pv-predictor-streamlit/1.0)"
}


def fetch_html(url: str) -> str:
    response = requests.get(url, headers=HEADERS, timeout=30)
    response.raise_for_status()
    response.encoding = response.apparent_encoding
    return response.text


def archive_page_url(archive_url: str, page: int) -> str:
    archive_url = archive_url.rstrip("/") + "/"
    if page == 1:
        return archive_url
    return f"{archive_url}page/{page}/"


def is_article_url(url: str, site_root: str = DEFAULT_SITE_ROOT) -> bool:
    parsed = urlparse(url)
    root_host = urlparse(site_root).netloc
    if parsed.netloc != root_host:
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
        "contact",
        "wp-",
    )
    return not any(path.startswith(prefix) for prefix in skip_prefixes)


def parse_archive_item(item, site_root: str = DEFAULT_SITE_ROOT) -> dict | None:
    link = item.select_one("a[href]")
    if not link:
        return None

    url = urljoin(site_root, link.get("href"))
    if not is_article_url(url, site_root=site_root):
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
                category = clean_text(after_date[: view_match.start()]) or "未分類"

    if view_count is None:
        return None

    return {
        "url": url,
        "title": title,
        "excerpt": excerpt,
        "published_date": published_date,
        "category": category or "未分類",
        "view_count": int(view_count),
    }


def parse_archive_page(html: str, site_root: str = DEFAULT_SITE_ROOT) -> list[dict]:
    soup = BeautifulSoup(html, "html.parser")
    main = soup.select_one("main") or soup

    items = main.select("li.p-postList__item")
    if not items:
        items = main.select("article, li")

    records = []
    for item in items:
        record = parse_archive_item(item, site_root=site_root)
        if record:
            records.append(record)

    unique = {}
    for record in records:
        unique[record["url"]] = record
    return list(unique.values())


def parse_article_body(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")

    for tag in soup.select(
        "script, style, noscript, iframe, nav, aside, form, "
        ".p-toc, .c-shareBtns, .p-shareBtns, .p-adBox, .p-authorBox, "
        ".l-header, .l-footer, .p-breadcrumb, .widget"
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

    body_text = body_area.get_text("\n", strip=True)
    return normalize_multiline_text(body_text)


def calc_days_since_publish(published_date: str) -> int:
    try:
        published = pd.to_datetime(published_date).date()
        return max((date.today() - published).days, 1)
    except Exception:
        return 1


def scrape_articles(
    *,
    archive_url: str = DEFAULT_ARCHIVE_URL,
    site_root: str = DEFAULT_SITE_ROOT,
    max_pages: int = 1,
    sleep_seconds: float = 1.0,
    progress_callback=None,
) -> pd.DataFrame:
    all_records: list[dict] = []
    seen_urls: set[str] = set()

    max_pages = max(int(max_pages), 1)
    sleep_seconds = max(float(sleep_seconds), 0.0)

    for page in range(1, max_pages + 1):
        list_url = archive_page_url(archive_url, page)
        if progress_callback:
            progress_callback("list", page, max_pages, f"一覧ページ取得中: {list_url}")

        html = fetch_html(list_url)
        records = parse_archive_page(html, site_root=site_root)

        for idx, record in enumerate(records, start=1):
            if record["url"] in seen_urls:
                continue
            seen_urls.add(record["url"])

            if progress_callback:
                progress_callback(
                    "article",
                    len(all_records) + 1,
                    None,
                    f"本文取得中: {record['title'][:60]}",
                )

            try:
                article_html = fetch_html(record["url"])
                body_text = parse_article_body(article_html)
                record["body_text"] = body_text
                record["body_length"] = len(body_text)
                record["days_since_publish"] = calc_days_since_publish(record["published_date"])
                all_records.append(record)
                time.sleep(sleep_seconds)
            except Exception as exc:
                record["error"] = str(exc)
                continue

        time.sleep(sleep_seconds)

    df = pd.DataFrame(all_records)
    if df.empty:
        raise RuntimeError("記事データを取得できませんでした。URLやサイト構造を確認してください。")

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
            df[col] = "" if col != "view_count" else 0

    return df[columns].drop_duplicates(subset=["url"]).reset_index(drop=True)


def dataframe_to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig")
