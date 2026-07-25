#!/usr/bin/env python3
"""Check the generated public site for broken local references and private files."""

from __future__ import annotations

import sys
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote, urlparse


FORBIDDEN_TEXT = (
    "AGENTS.md",
    "Vault Conventions",
    "工作约定与进度",
    "学习约定与进度",
    "阅读与审校记录",
    "system prompt",
    "developer prompt",
)


class ReferenceParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.references: list[str] = []

    def handle_starttag(
        self, tag: str, attrs: list[tuple[str, str | None]]
    ) -> None:
        attributes = dict(attrs)
        if tag == "a" and attributes.get("href"):
            self.references.append(str(attributes["href"]))
        if tag in {"img", "script"} and attributes.get("src"):
            self.references.append(str(attributes["src"]))
        if tag == "link" and attributes.get("href"):
            self.references.append(str(attributes["href"]))


def resolve_reference(site_root: Path, page: Path, reference: str) -> Path | None:
    parsed = urlparse(reference)
    if parsed.scheme or parsed.netloc or reference.startswith("//"):
        return None
    if not parsed.path:
        return None

    clean_path = unquote(parsed.path)
    if clean_path.startswith("/"):
        target = site_root / clean_path.lstrip("/")
    else:
        target = page.parent / clean_path

    if target.is_dir() or clean_path.endswith("/"):
        target = target / "index.html"
    return target.resolve()


def main() -> int:
    site_root = (
        Path(sys.argv[1]).resolve()
        if len(sys.argv) == 2
        else (Path(__file__).resolve().parent.parent / "_site").resolve()
    )
    if not site_root.is_dir():
        print(f"Built site not found: {site_root}", file=sys.stderr)
        return 1

    html_files = sorted(site_root.rglob("*.html"))
    failures: list[str] = []

    for page in html_files:
        content = page.read_text(encoding="utf-8")
        for forbidden in FORBIDDEN_TEXT:
            if forbidden.casefold() in content.casefold():
                failures.append(f"private marker {forbidden!r} found in {page}")

        parser = ReferenceParser()
        parser.feed(content)
        for reference in parser.references:
            target = resolve_reference(site_root, page, reference)
            if target is not None and not target.exists():
                failures.append(
                    f"broken reference in {page.relative_to(site_root)}: {reference}"
                )

    note_pages = list((site_root / "notes").glob("*/index.html"))
    notes_page = site_root / "notes" / "index.html"
    if not notes_page.is_file():
        failures.append("Notes archive page is missing")
    elif "Math Notes" in notes_page.read_text(encoding="utf-8"):
        failures.append("Legacy 'Math Notes' label remains on the Notes archive")

    if len(note_pages) == 0:
        failures.append("No note pages were generated")
    elif "mathjax@4/tex-mml-chtml.js" not in note_pages[0].read_text(encoding="utf-8"):
        failures.append("MathJax is not loaded on note pages")

    if failures:
        print("\n".join(failures), file=sys.stderr)
        return 1

    print(
        f"Checked {len(html_files)} HTML pages, {len(note_pages)} notes, "
        "all local references, and private-content exclusions."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
