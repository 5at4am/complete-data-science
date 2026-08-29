from __future__ import annotations

import re
from html import unescape
from pathlib import Path
from urllib import error, request

from bs4 import BeautifulSoup

from .sitemap import CrawledPage


def _fetch(url: str) -> str | None:
    req = request.Request(url, headers={"User-Agent": "taxkraft-bot/1.0"})
    try:
        with request.urlopen(req, timeout=30) as resp:
            if "text/html" not in resp.headers.get("Content-Type", ""):
                return None
            return resp.read().decode("utf-8", errors="replace")
    except (error.HTTPError, error.URLError, TimeoutError):
        return None


def extract_text_from_spa(html: str, url: str) -> tuple[str, str, str]:
    """Extract title/description and any statically-rendered body text."""
    soup = BeautifulSoup(html, "html.parser")
    title = (soup.title.get_text(strip=True) if soup.title else "") or ""
    meta = soup.find("meta", attrs={"name": "description"})
    description = unescape(meta.get("content", "")) if meta else ""
    body = soup.find("body")
    text = " ".join(body.get_text(" ", strip=True).split()) if body else ""
    text = re.sub(r"^\s*(home|menu|close|skip\s+to\s+content)\b.*?$", "", text, flags=re.I | re.M)
    return title, description, text[:20000]


def crawl_sitemap(urls: list[str]) -> list[CrawledPage]:
    pages: list[CrawledPage] = []
    for url in urls:
        html = _fetch(url)
        if html is None:
            continue
        title, description, text = extract_text_from_spa(html, url)
        pages.append(CrawledPage(url=url, title=title, description=description, text=text))
    return pages


def write_corpus(pages: list[CrawledPage], kb_dir: Path | None = None) -> int:
    """Write each page to knowledge_base/crawled/<slug>.md with provenance.

    SPA pages carry a marker so consumers know the text is only metadata + shell
    until the Playwright renderer is run (see README).
    """
    kb_dir = kb_dir or (Path(__file__).resolve().parents[1] / "knowledge_base" / "crawled")
    kb_dir.mkdir(parents=True, exist_ok=True)
    n = 0
    for page in pages:
        slug = page.url.rstrip("/").split("/")[-1] or "home"
        content = (
            f"<!--\n"
            f"source-title: {page.title or slug}\n"
            f"source-url: {page.url}\n"
            f"verified-at: {page.fetched_at}\n"
            f"provenance: crawled from public website (sitemap-driven)\n"
            f"-->\n\n"
            f"# {page.title or slug}\n\n"
            f"{page.description}\n\n"
            f"{page.text}\n"
        )
        (kb_dir / f"{slug}.md").write_text(content, encoding="utf-8")
        n += 1
    return n