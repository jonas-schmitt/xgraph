"""CPU-only documentation build; never construct a model or a native engine."""

import os
import re
from pathlib import Path
from urllib.parse import unquote, urlsplit

project = "xgraph"
copyright = "xgraph contributors"
release = "0.0.1"
extensions = ["myst_parser", "sphinx.ext.autodoc", "sphinx.ext.napoleon"]
source_suffix = {".md": "markdown", ".rst": "restructuredtext"}
root_doc = "index"
exclude_patterns = ["_build"]
myst_enable_extensions = ["colon_fence", "deflist"]
myst_heading_anchors = 6
autodoc_member_order = "bysource"
autodoc_typehints = "signature"
autodoc_class_signature = "separated"
html_theme = "furo"
html_title = "xgraph"
html_show_sourcelink = True
html_theme_options = {"navigation_with_keys": True}


def repository_links(docname, text):
    """Copy only publication-safe source links into HTML downloads."""
    docs = Path(__file__).resolve().parent
    repository = docs.parent
    document = docs / docname
    title = r"""(?:"(?:\\.|[^"\\])*"|'(?:\\.|[^'\\])*'|\([^\n)]*\))"""
    inline = (
        r"(?<!!)\[(?P<inline_label>[^\]]+)\]\(\s*"
        r"(?:<(?P<inline_angle>[^>\n]+)>|(?P<inline_plain>(?:\\.|[^\s()\\])+))"
        rf"(?:\s+{title})?\s*\)"
    )
    definition = (
        r"^ {0,3}\[(?P<definition_label>[^\]\n]+)\]:[ \t]*(?:\n[ \t]*)?"
        r"(?:<(?P<definition_angle>[^>\n]+)>|(?P<definition_plain>\S+))"
        rf"(?:[ \t]*(?:\n[ \t]+)?{title})?[ \t]*(?=\n|$)"
    )
    reference = r"(?<!!)\[(?P<reference_label>[^\]\n]+)\](?:\[(?P<reference_key>[^\]\n]*)\])?(?!\()"
    inline_code = r"(?<!`)(?P<code_fence>`+)(?!`)(.*?)(?<!`)(?P=code_fence)(?!`)"
    # Earliest token wins: '[' starts a whole link before code in its label;
    # a standalone code span starts before any link syntax inside that span.
    pattern = re.compile(
        rf"(?P<code>{inline_code})|(?P<definition>{definition})|"
        rf"(?P<inline>{inline})|(?P<reference>{reference})",
        re.M | re.S,
    )

    def route(destination):
        url = urlsplit(destination)
        if url.scheme or url.netloc or not url.path:
            return "link", None
        destination_path = Path(unquote(url.path))
        if destination_path.is_absolute():
            return "denied", None
        target = Path(os.path.abspath(document.parent / destination_path))
        if not target.is_relative_to(repository):
            return "denied", None
        canonical_path = Path(os.path.relpath(target, document.parent)).as_posix()
        if destination_path.as_posix() != canonical_path:
            return "denied", None
        relative = target.relative_to(repository)
        if not relative.parts or any(component.startswith(".") for component in relative.parts):
            return "denied", None
        document_link = relative.parts[0] == "docs" and relative.suffix == ".md"
        root_sources = {
            "README.md",
            "DEVELOPMENT.md",
            "tests/__init__.py",
            "tests/conftest.py",
            "tests/test_oracle.py",
            "tests/test_engine.py",
            "tests/test_vllm_adapter.py",
            "tests/test_documentation_examples.py",
            "tests/test_label_index.py",
            "tests/test_tokenizer.py",
            "tests/test_engine_cpu.py",
            "tests/native_cpu.py",
            "tests/test_source_smoke.py",
            "LICENSE",
            "NOTICE",
            "THIRD_PARTY_NOTICES.md",
            "CONTRIBUTING.md",
            "GLOSSARY.md",
            "pyproject.toml",
            "CMakeLists.txt",
            "CITATION.cff",
            "SECURITY.md",
            "LICENSES/CC-BY-4.0.txt",
            "LICENSES/GWalk-NOTICE.txt",
            "LICENSES/ReFactX-Apache-2.0.txt",
            "LICENSES/nanobind-BSD-3-Clause.txt",
            "LICENSES/robin-map-MIT.txt",
        }
        source_suffixes = {
            "src": {".py"},
            "csrc": {".cu", ".cpp", ".h", ".hpp"},
            "examples": {".py", ".md"},
        }
        if not (
            document_link
            or relative.as_posix() in root_sources
            or (
                relative.parts[0] in source_suffixes
                and relative.suffix in source_suffixes[relative.parts[0]]
            )
        ):
            return "denied", None
        candidate = repository
        for component in relative.parts:
            candidate = candidate / component
            if candidate.is_symlink():
                return "denied", None
        if document_link:
            return "link", None
        target = target.resolve()
        if not target.is_relative_to(repository) or not target.is_file():
            return "denied", None
        return "download", canonical_path

    def denied(label, destination):
        # A live Markdown link would let MyST auto-download a denied file.
        label = label.replace("`", "").replace("\n", " ")
        fence = "`" * (max((len(run) for run in re.findall(r"`+", destination)), default=0) + 1)
        return f"{label} ({fence}{destination}{fence})"

    def render_link(label, destination, original):
        kind, canonical_path = route(destination)
        if kind == "denied":
            return denied(label, destination)
        if kind == "download":
            label = label.replace("`", "").replace("\n", " ")
            return f"{{download}}`{label} <{canonical_path}>`"
        return original

    def reference_key(label):
        return " ".join(label.split()).casefold()

    references = {}

    def remember_references(text):
        for match in pattern.finditer(text):
            if match["definition"] is not None:
                destination = match["definition_angle"] or match["definition_plain"]
                references.setdefault(reference_key(match["definition_label"]), destination)

    def replace(match):
        if match["code"] is not None:
            return match[0]
        if match["definition"] is not None:
            destination = match["definition_angle"] or match["definition_plain"]
            if route(destination)[0] == "denied":
                return denied(match["definition_label"], destination)
            return match[0]
        if match["inline"] is not None:
            destination = match["inline_angle"] or match["inline_plain"]
            return render_link(match["inline_label"], destination, match[0])
        label = match["reference_label"]
        key = reference_key(match["reference_key"] or label)
        if key in references:
            return render_link(label, references[key], match[0])
        return match[0]

    parts, paragraph, fence = [], [], None
    for line in text.splitlines(keepends=True):
        marker = re.match(r"^\s{0,3}(`{3,}|~{3,})", line)
        if marker:
            if paragraph:
                parts.append((False, "".join(paragraph)))
                paragraph = []
            if fence is None:
                fence = marker[1]
            elif marker[1][0] == fence[0] and len(marker[1]) >= len(fence):
                fence = None
            parts.append((True, line))
        else:
            if fence:
                parts.append((True, line))
            else:
                paragraph.append(line)
    if paragraph:
        parts.append((False, "".join(paragraph)))
    for fenced, text in parts:
        if not fenced:
            remember_references(text)
    return "".join(text if fenced else pattern.sub(replace, text) for fenced, text in parts)


def _source_read(app, docname, source):
    source[0] = repository_links(docname, source[0])


def setup(app):
    app.connect("source-read", _source_read)
