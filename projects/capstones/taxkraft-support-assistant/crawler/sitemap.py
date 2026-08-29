from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from urllib import request
from xml.etree import ElementTree


@dataclass
class CrawledPage:
    url: str
    title: str = ""
    description: str = ""
    text: str = ""
    fetched_at: str = field(default_factory=lambda: date.today().isoformat())


def fetch_sitemap(base_url: str) -> list[str]:
    """Return the list of page URLs from sitemap.xml (deepest page first is useful)."""
    req = request.Request(f"{base_url}/sitemap.xml", headers={"User-Agent": "taxkraft-bot/1.0"})
    with request.urlopen(req, timeout=30) as resp:
        xml_data = resp.read()
    root = ElementTree.fromstring(xml_data)
    ns = {"s": "http://www.sitemaps.org/schemas/sitemap/0.9"}
    urls = [loc.text.strip() for loc in root.findall(".//s:loc", ns) if loc.text]
    return urls