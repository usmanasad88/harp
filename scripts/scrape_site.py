#!/usr/bin/env python3
"""Scrape a website's content pages into clean Markdown files for the RAG corpus.

This is a lightweight context-builder for HARP. It crawls the same-domain pages
linked from a base URL, extracts the readable text of each page, and writes one
Markdown file per page into the output directory (``data/`` by default). Those
files are what Phase 2 indexes into the vector store.

Only the standard ``requests`` + ``beautifulsoup4`` libraries are used, so it
runs on a plain laptop with no headless browser.

Usage:
    python scripts/scrape_site.py https://indusrasexpo.gov.pk/
    python scripts/scrape_site.py https://example.com/ --out data --max-pages 50

The scraper is intentionally polite: it stays on one domain, skips obvious
non-content paths (asset files, API routes, registration forms), and pauses
briefly between requests.
"""

from __future__ import annotations

import argparse
import re
import sys
import time
from datetime import date
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup

# Paths we never treat as "knowledge" pages: forms, API endpoints, raw assets.
SKIP_PREFIXES = ("/api/", "/registration/", "/_next/", "/assets/")
SKIP_SUFFIXES = (
    ".css", ".js", ".png", ".jpg", ".jpeg", ".gif", ".svg", ".ico",
    ".webp", ".mp4", ".pdf", ".zip", ".woff", ".woff2",
)

HEADERS = {
    "User-Agent": "HARP-context-builder/0.1 (+research; bilingual voice assistant)",
}


def slugify(path: str) -> str:
    """Turn a URL path into a safe filename stem, e.g. '/about' -> 'about'."""
    s = path.strip("/").replace("/", "_")
    s = re.sub(r"[^a-zA-Z0-9_-]+", "-", s)
    return s or "index"


def is_content_path(path: str) -> bool:
    if any(path.startswith(p) for p in SKIP_PREFIXES):
        return False
    if any(path.lower().endswith(s) for s in SKIP_SUFFIXES):
        return False
    return True


def discover_links(base_url: str, html: str) -> list[str]:
    """Return same-domain content URLs linked from the given HTML."""
    base_host = urlparse(base_url).netloc
    soup = BeautifulSoup(html, "html.parser")
    found: dict[str, None] = {}  # dict preserves insertion order, dedupes
    for a in soup.find_all("a", href=True):
        url = urljoin(base_url, a["href"]).split("#")[0]
        parts = urlparse(url)
        if parts.netloc != base_host:
            continue
        if not is_content_path(parts.path):
            continue
        found[url] = None
    return list(found)


def extract_markdown(html: str, url: str) -> tuple[str, str]:
    """Extract (title, markdown_body) of the main readable content of a page."""
    soup = BeautifulSoup(html, "html.parser")

    # Drop non-content elements entirely.
    for tag in soup(["script", "style", "noscript", "svg", "nav", "footer", "header"]):
        tag.decompose()

    title = soup.title.get_text(strip=True) if soup.title else url

    root = soup.find("main") or soup.body or soup
    lines: list[str] = []
    for el in root.find_all(["h1", "h2", "h3", "h4", "h5", "h6", "p", "li", "tr"]):
        if el.name == "tr":
            cells = [
                " ".join(c.get_text(separator=" ", strip=True).split())
                for c in el.find_all(["th", "td"])
            ]
            cells = [c for c in cells if c]
            if cells:
                lines.append("| " + " | ".join(cells) + " |")
            continue
        text = " ".join(el.get_text(separator=" ", strip=True).split())
        if not text:
            continue
        if el.name and el.name.startswith("h"):
            level = int(el.name[1])
            lines.append("\n" + "#" * level + " " + text)
        elif el.name == "li":
            lines.append("- " + text)
        else:
            lines.append(text)

    # Collapse runs of duplicate consecutive lines (repeated boilerplate).
    body_lines: list[str] = []
    for ln in lines:
        if body_lines and body_lines[-1] == ln:
            continue
        body_lines.append(ln)

    return title, "\n".join(body_lines).strip()


def fetch(url: str, timeout: int) -> str | None:
    try:
        resp = requests.get(url, headers=HEADERS, timeout=timeout)
        resp.raise_for_status()
        return resp.text
    except requests.RequestException as exc:
        print(f"  ! failed: {exc}", file=sys.stderr)
        return None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("base_url", help="Site to scrape, e.g. https://indusrasexpo.gov.pk/")
    ap.add_argument("--out", default="data", help="Output directory (default: data)")
    ap.add_argument("--max-pages", type=int, default=50, help="Safety cap on pages")
    ap.add_argument("--delay", type=float, default=0.5, help="Seconds between requests")
    ap.add_argument("--timeout", type=int, default=25, help="Per-request timeout (s)")
    args = ap.parse_args()

    import os
    os.makedirs(args.out, exist_ok=True)

    print(f"Fetching home page: {args.base_url}")
    home = fetch(args.base_url, args.timeout)
    if home is None:
        print("Could not fetch the base URL. Aborting.", file=sys.stderr)
        return 1

    urls = [args.base_url] + discover_links(args.base_url, home)
    # Dedupe while preserving order.
    seen: dict[str, None] = {}
    for u in urls:
        seen[u.rstrip("/") or u] = None
    urls = list(seen)[: args.max_pages]

    print(f"Discovered {len(urls)} content page(s).\n")

    written = 0
    for url in urls:
        path = urlparse(url).path or "/"
        print(f"- {url}")
        html = home if url.rstrip("/") == args.base_url.rstrip("/") else fetch(url, args.timeout)
        if html is None:
            continue
        title, body = extract_markdown(html, url)
        if not body:
            print("  (no extractable text, skipped)")
            continue
        stem = slugify(path)
        out_path = os.path.join(args.out, f"{stem}.md")
        front = (
            f"# {title}\n\n"
            f"> Source: {url}\n"
            f"> Fetched: {date.today().isoformat()}\n\n"
        )
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(front + body + "\n")
        print(f"  -> {out_path} ({len(body)} chars)")
        written += 1
        if args.delay:
            time.sleep(args.delay)

    print(f"\nDone. Wrote {written} file(s) to {args.out}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
