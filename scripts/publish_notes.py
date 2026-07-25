#!/usr/bin/env python3
"""Publish selected notes into the Jekyll site.

This script creates public copies, converts wiki links, and copies only assets
referenced by published notes.
"""

from __future__ import annotations

import json
import re
import shutil
import sys
import unicodedata
from pathlib import Path


INTERNAL_FILENAMES = {
    "AGENTS.md",
    "CLAUDE.md",
    "GEMINI.md",
    "README.md",
    "Vault Conventions.md",
}

INTERNAL_NAME_PATTERN = re.compile(
    r"(agent|prompt|instruction|convention|工作约定|学习约定|阅读与审校记录|进度|审校)",
    re.IGNORECASE,
)

INTERNAL_TAG_PREFIXES = ("internal", "internal/", "meta", "meta/", "project/")


def parse_frontmatter(text: str) -> tuple[dict[str, object], str]:
    if not text.startswith("---\n"):
        return {}, text

    parts = text.split("---\n", 2)
    if len(parts) != 3:
        return {}, text

    raw, body = parts[1], parts[2]
    metadata: dict[str, object] = {}
    active_list: str | None = None

    for line in raw.splitlines():
        list_item = re.match(r"^\s+-\s+(.+?)\s*$", line)
        if list_item and active_list:
            value = list_item.group(1).strip().strip("\"'")
            current = metadata.setdefault(active_list, [])
            if isinstance(current, list):
                current.append(value)
            continue

        pair = re.match(r"^([A-Za-z_][A-Za-z0-9_-]*):(?:\s*(.*))?$", line)
        if not pair:
            continue

        key, value = pair.group(1), (pair.group(2) or "").strip()
        if not value:
            metadata[key] = []
            active_list = key
        else:
            metadata[key] = value.strip("\"'")
            active_list = None

    return metadata, body


def slugify(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", value)
    ascii_value = normalized.encode("ascii", "ignore").decode("ascii")
    ascii_value = ascii_value.replace(".", "-")
    slug = re.sub(r"[^A-Za-z0-9]+", "-", ascii_value).strip("-").lower()
    return slug or "note"


def yaml_scalar(value: object) -> str:
    return json.dumps(str(value), ensure_ascii=False)


def public_note(path: Path, metadata: dict[str, object]) -> bool:
    if path.name in INTERNAL_FILENAMES:
        return False
    if path.name.startswith(".") or INTERNAL_NAME_PATTERN.search(path.name):
        return False
    if str(metadata.get("publish", "")).lower() == "false":
        return False

    tags = metadata.get("tags", [])
    if not isinstance(tags, list):
        tags = [tags]
    return not any(str(tag).lower().startswith(INTERNAL_TAG_PREFIXES) for tag in tags)


def first_heading(body: str, fallback: str) -> str:
    match = re.search(r"^#\s+(.+?)\s*$", body, re.MULTILINE)
    return match.group(1) if match else fallback


def display_title(metadata: dict[str, object], body: str, fallback: str) -> str:
    aliases = metadata.get("aliases", [])
    if isinstance(aliases, list) and aliases:
        return str(aliases[0])

    heading = first_heading(body, fallback)
    return re.sub(r"^[A-Za-z]+\d+(?:\.\d+)*-\d+(?:\.\d+)*-", "", heading)


def remove_first_h1(body: str) -> str:
    return re.sub(r"^\s*#\s+.+?\n+", "", body, count=1)


def preserve_math_delimiters(body: str) -> str:
    """Keep TeX delimiters intact through Kramdown's backslash handling."""
    display_parts = body.split("$$")
    if (len(display_parts) - 1) % 2:
        raise ValueError("Unbalanced $$ display-math delimiters")

    rebuilt: list[str] = []
    for index, part in enumerate(display_parts):
        rebuilt.append(part)
        if index < len(display_parts) - 1:
            rebuilt.append(r"\\[" if index % 2 == 0 else r"\\]")

    protected = "".join(rebuilt)
    protected = protected.replace(r"\(", r"\\(").replace(r"\)", r"\\)")
    inline_parts = re.split(r"(?<!\\)\$", protected)
    if (len(inline_parts) - 1) % 2:
        raise ValueError("Unbalanced $ inline-math delimiters")

    rebuilt = []
    for index, part in enumerate(inline_parts):
        rebuilt.append(part)
        if index < len(inline_parts) - 1:
            rebuilt.append(r"\\(" if index % 2 == 0 else r"\\)")

    return "".join(rebuilt)


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: publish_notes.py SOURCE_DIRECTORY", file=sys.stderr)
        return 2

    script_path = Path(__file__).resolve()
    site_root = script_path.parent.parent
    source_root = Path(sys.argv[1]).expanduser().resolve()

    if not source_root.is_dir():
        print(f"Source directory not found: {source_root}", file=sys.stderr)
        return 1

    note_records: list[dict[str, object]] = []
    for source in sorted(source_root.glob("*.md")):
        text = source.read_text(encoding="utf-8")
        metadata, body = parse_frontmatter(text)
        if not metadata or not public_note(source, metadata):
            continue

        source_title = first_heading(body, source.stem)
        title = display_title(metadata, body, source.stem)
        slug = slugify(source.stem)
        aliases = metadata.get("aliases", [])
        if not isinstance(aliases, list):
            aliases = [aliases]

        note_records.append(
            {
                "source": source,
                "metadata": metadata,
                "body": body,
                "source_title": source_title,
                "title": title,
                "slug": slug,
                "aliases": aliases,
            }
        )

    if not note_records:
        print("No public notes found; refusing to replace the public archive.", file=sys.stderr)
        return 1

    link_map: dict[str, str] = {}
    for record in note_records:
        url = f"/notes/{record['slug']}/"
        source = record["source"]
        assert isinstance(source, Path)
        link_map[source.stem] = url
        link_map[str(record["source_title"])] = url
        for alias in record["aliases"]:
            link_map[str(alias)] = url

    notes_dir = site_root / "_notes"
    assets_dir = site_root / "assets" / "notes"

    if notes_dir.exists():
        shutil.rmtree(notes_dir)
    if assets_dir.exists():
        shutil.rmtree(assets_dir)
    notes_dir.mkdir(parents=True)
    assets_dir.mkdir(parents=True)

    copied_assets: set[Path] = set()

    def replace_embed(match: re.Match[str]) -> str:
        target = match.group(1).strip()
        label = (match.group(2) or Path(target).stem).strip()
        attachments_root = (source_root / "attachments").resolve()
        candidate = (source_root / target).resolve()

        if candidate.is_file() and candidate.is_relative_to(attachments_root):
            source_asset = candidate
        else:
            matches = [
                path.resolve()
                for path in attachments_root.rglob(Path(target).name)
                if path.is_file()
            ]
            if len(matches) != 1:
                raise FileNotFoundError(
                    f"Expected one attachment for {target!r}, found {len(matches)}"
                )
            source_asset = matches[0]

        try:
            relative_asset = source_asset.relative_to(attachments_root)
        except ValueError as exc:
            raise ValueError(f"Refusing to publish an embed outside attachments/: {target}") from exc

        if not source_asset.is_file():
            raise FileNotFoundError(f"Referenced asset not found: {source_asset}")

        destination = assets_dir / relative_asset
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_asset, destination)
        copied_assets.add(relative_asset)
        return f"![{label}]({{{{ '/assets/notes/{relative_asset.as_posix()}' | relative_url }}}})"

    def replace_wikilink(match: re.Match[str]) -> str:
        target = match.group(1).strip()
        heading = (match.group(2) or "").lstrip("#")
        label = (match.group(3) or heading or target).strip()
        url = link_map.get(target)
        if not url:
            return label
        if heading:
            url = f"{url}#{slugify(heading)}"
        return f"[{label}]({{{{ '{url}' | relative_url }}}})"

    def replace_local_anchor(match: re.Match[str]) -> str:
        heading = match.group(1).strip()
        label = (match.group(2) or heading).strip()
        return f"[{label}](#{slugify(heading)})"

    for record in note_records:
        metadata = record["metadata"]
        assert isinstance(metadata, dict)
        body = remove_first_h1(str(record["body"]))
        body = re.sub(r"!\[\[([^\]|]+)(?:\|([^\]]+))?\]\]", replace_embed, body)
        body = re.sub(
            r"\[\[#([^\]|]+)(?:\|([^\]]+))?\]\]",
            replace_local_anchor,
            body,
        )
        body = re.sub(
            r"\[\[([^\]|#]+)(#[^\]|]+)?(?:\|([^\]]+))?\]\]",
            replace_wikilink,
            body,
        )
        body = preserve_math_delimiters(body)

        tags = metadata.get("tags", [])
        if not isinstance(tags, list):
            tags = [tags]

        frontmatter = [
            "---",
            "layout: note",
            f"title: {yaml_scalar(record['title'])}",
            f"source_title: {yaml_scalar(record['source_title'])}",
            f"course: {yaml_scalar(metadata.get('course', 'Independent notes'))}",
            f"sequence: {yaml_scalar(metadata.get('sequence', ''))}",
            f"source_context: {yaml_scalar(metadata.get('source_context', ''))}",
            f"permalink: {yaml_scalar('/notes/' + str(record['slug']) + '/')}",
            "tags:",
        ]
        frontmatter.extend(f"  - {yaml_scalar(tag)}" for tag in tags)
        frontmatter.extend(["---", ""])

        output = "\n".join(frontmatter) + body.lstrip()
        destination = notes_dir / f"{record['slug']}.md"
        destination.write_text(output, encoding="utf-8")

    print(
        f"Published {len(note_records)} notes and {len(copied_assets)} referenced assets."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
