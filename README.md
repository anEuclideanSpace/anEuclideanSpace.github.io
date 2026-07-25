# an Euclidean space

Source for [aneuclideanspace.github.io](https://aneuclideanspace.github.io), a public archive
of mathematical and technical notes.

## Publishing notes

Run:

```bash
python3 scripts/sync_metis_notes.py /path/to/Metis
```

The publisher copies formal note files and only the images they reference. It excludes
agent instructions, vault conventions, progress records, review logs, hidden files, and
notes tagged as `meta`, `internal/...`, or `project/...`. A note can also opt out with
`publish: false` in its YAML frontmatter.

Generated note copies live in `_notes/`; referenced images live in `assets/notes/`.
The Metis vault remains the source of truth.

Before publishing, build and check the site:

```bash
bundle exec jekyll build
python3 scripts/check_public_site.py
```
