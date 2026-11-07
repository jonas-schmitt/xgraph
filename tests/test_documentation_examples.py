"""Run the published CPU examples as a user would, without network or a model."""

from __future__ import annotations

import ast
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

from xgraph.index import load_kg_index

ROOT = Path(__file__).resolve().parents[1]


def test_sphinx_cannot_download_blocked_local_links_or_warn_about_research_links(tmp_path):
    """Observe MyST's rendered HTML, including its automatic download fallback."""
    root = tmp_path
    docs = root / "docs"
    docs.mkdir()
    (docs / "conf.py").write_text((ROOT / "docs/conf.py").read_text() + '\nhtml_theme = "basic"\n')
    safe_source = root / "src/xgraph/kg.py"
    safe_source.parent.mkdir(parents=True)
    safe_source.write_text("# Invented safe library source\n")
    blocked = {
        "eval/private.json": "SYNTHETIC_EVAL_MUST_NOT_BE_READ",
        "target/private.txt": "SYNTHETIC_TARGET_MUST_NOT_BE_READ",
        "eval/research.md": "SYNTHETIC_RESEARCH_MUST_NOT_BE_READ",
        "scripts/internal.json": "SYNTHETIC_SCRIPT_MUST_NOT_BE_READ",
        "tests/helper.py": "SYNTHETIC_TEST_MUST_NOT_BE_READ",
    }
    for name, sentinel in blocked.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(sentinel)
    (docs / "concepts.md").write_text("# Concepts\n\nPublic documentation.\n")
    (docs / "index.md").write_text(
        "# Guide\n\n[safe source](../src/xgraph/kg.py)\n"
        "[`safe source`](../src/xgraph/kg.py)\n"
        + "".join(f"[blocked source](../{name})\n" for name in blocked)
        + "[angle](<../eval/private.json>)\n"
        '[titled](../target/private.txt "title")\n'
        "[`code label`](../eval/private.json)\n"
        "[`multiline`\nsource](../target/private.txt)\n"
        "[full reference][private]\n[private][]\n[private]\n\n"
        "[private]: ../eval/private.json\n\n"
        + "[concepts](concepts.md)\n[external](https://example.invalid/source.py)\n\n"
        "```{toctree}\nconcepts\n```\n"
    )
    output = root / "html"
    runner = """
import os, sys
from pathlib import Path
from sphinx.cmd.build import main
source, output, root = sys.argv[1:]
blocked = {Path(root) / name for name in (
    'eval/private.json', 'target/private.txt', 'eval/research.md',
    'scripts/internal.json', 'tests/helper.py',
)}
def audit(event, args):
    if event == 'open' and isinstance(args[0], (str, bytes)):
        if Path(os.path.abspath(os.fsdecode(args[0]))) in blocked:
            raise RuntimeError('MyST attempted to open a blocked synthetic source')
sys.addaudithook(audit)
raise SystemExit(main(['-W', '--keep-going', '-b', 'html', source, output]))
"""
    result = subprocess.run(
        [sys.executable, "-I", "-c", runner, str(docs), str(output), str(root)],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "WARNING" not in result.stdout + result.stderr
    assert sorted(path.name for path in (output / "_downloads").rglob("*.*")) == ["kg.py"]
    html = (output / "index.html").read_text()
    for name, sentinel in blocked.items():
        assert sentinel not in html
        assert name in html
        assert f'href="../{name}' not in html


def test_package_import_keeps_lightweight_readers_independent_of_index_dependencies(tmp_path):
    """Importing the namespace must not eagerly load the index or GPU boundary."""
    program = (
        "import sys\n"
        f"sys.path.insert(0, {str(ROOT / 'src')!r})\n"
        "import xgraph\n"
        "assert 'numpy' not in sys.modules\n"
        "assert 'xgraph.engine' not in sys.modules\n"
        "assert 'Oracle' in dir(xgraph)\n"
    )
    result = subprocess.run(
        [sys.executable, "-I", "-S", "-c", program],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_package_entry_points_build_and_resolve_application_triples(toy_tokenizer):
    """A user can discover the CPU workflow without knowing module locations."""
    import xgraph as xg

    tokenizer = xg.ServedTokenizer(toy_tokenizer)
    graph = xg.KnowledgeGraph(
        entities=(xg.Entity("item:atlas", ("Atlas",)), xg.Entity("person:ada", ("Ada",))),
        relations=(xg.Relation("rel:creator", ("creator",)),),
        triples=(("item:atlas", "rel:creator", "person:ada"),),
    )
    syntax = xg.SpanSyntax()
    index = xg.build_kg_index(graph, tokenizer, syntax)
    oracle = xg.Oracle(index, tokenizer, syntax)
    state = oracle.initial_state()
    for piece in (
        syntax.opener,
        " Atlas",
        syntax.separator,
        " creator",
        syntax.separator,
        " Ada",
        syntax.closer,
    ):
        for token in tokenizer.encode(piece):
            state = oracle.advance(state, token)
    assert state.last_span.resolved
    assert state.last_span.triples == (("item:atlas", "rel:creator", "person:ada"),)


def test_api_reference_is_generated_from_public_docstrings():
    """References resolve to real objects; Sphinx checks the rendered signatures."""
    import importlib
    import inspect

    references = []
    for path in (ROOT / "docs/api").glob("*.md"):
        for name in re.findall(r"```\{auto(?:class|function)\} (\S+)", path.read_text()):
            module, _, attr = name.rpartition(".")
            obj = getattr(importlib.import_module(module), attr)
            assert inspect.getdoc(obj), name
            references.append(name)
    assert "xgraph.oracle.Oracle" in references
    assert "xgraph.index.build_kg_index" in references


def user_pages():
    """The public library documentation track."""
    pages = {ROOT / "README.md"}
    pending = [ROOT / "docs/index.md"]
    while pending:
        page = pending.pop().resolve()
        if page in pages or page.relative_to(ROOT).as_posix() in {
            "docs/internals.md",
        }:
            continue
        pages.add(page)
        for block in re.findall(r"```\{toctree\}\n(.*?)```", page.read_text(), re.S):
            for line in block.splitlines():
                if line.strip() and not line.startswith(":"):
                    pending.append((page.parent / line.strip()).with_suffix(".md"))
    return sorted(pages)


PYTHON_BLOCKS = [
    pytest.param(block, id=f"{path.relative_to(ROOT)}:{i}")
    for path in user_pages()
    for i, block in enumerate(re.findall(r"```python\n(.*?)```", path.read_text(), re.S))
]

INCLUDED_SCRIPTS = sorted(
    {
        (page.parent / relative).resolve()
        for page in user_pages()
        for relative in re.findall(r"```\{literalinclude\} (\S+\.py)", page.read_text())
    }
)
SHELL_BLOCKS = [
    pytest.param(block, id=f"{page.relative_to(ROOT)}:{i}")
    for page in user_pages()
    for i, block in enumerate(re.findall(r"```(?:bash|sh)\n(.*?)```", page.read_text(), re.S))
]


@pytest.mark.parametrize("block", PYTHON_BLOCKS)
def test_published_python_blocks_execute_on_cpu(block, tmp_path):
    """Each inline Python example runs in a fresh, isolated working directory."""
    result = subprocess.run(
        [sys.executable, "-c", block], cwd=tmp_path, capture_output=True, text=True, timeout=60
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("script", INCLUDED_SCRIPTS, ids=lambda path: path.name)
def test_included_examples_execute_on_cpu(script, tmp_path):
    arguments = []
    if script.name == "vllm_generation.py":
        arguments = [
            "--dry-run",
            "--model",
            str(tmp_path / "model"),
            "--index",
            str(tmp_path / "index"),
            "--tokenizer",
            str(tmp_path / "tokenizer.json"),
        ]
    result = subprocess.run(
        [sys.executable, str(script), *arguments],
        cwd=tmp_path,
        env={**os.environ, "CUDA_VISIBLE_DEVICES": ""},
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("block", SHELL_BLOCKS)
def test_shell_recipes_execute_with_cpu_command_stand_ins(block, tmp_path):
    """Exercise quoting/exports/argv without installing tools or starting a worker."""
    commands = tmp_path / "bin"
    commands.mkdir()
    for name in ("uv", "python", "python3", "vllm", "nvcc"):
        stub = commands / name
        stub.write_text('#!/bin/bash\nprintf "%s\\n" "$0 $*" >> "$DOC_ARGV_LOG"\n')
        stub.chmod(0o755)
    result = subprocess.run(
        ["/bin/bash", "-euc", block],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        env={
            **os.environ,
            "PATH": str(commands),
            "DOC_ARGV_LOG": str(tmp_path / "argv"),
            "MODEL_DIR": str(tmp_path / "model"),
            "KG_INDEX_DIR": str(tmp_path / "index"),
        },
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert (tmp_path / "argv").is_file(), "recipe never invoked a checked command"


def test_online_request_examples_use_real_adapter_arguments(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from xgraph.vllm import XGraphLogitsProcessor

    versions = tmp_path / "versions"
    run_example("string_versions.py", "--out", versions)
    monkeypatch.setenv("XGRAPH_KG_VERSIONS", str(versions))
    checked = []
    for page in user_pages():
        for block in re.findall(r"```json\n(.*?)```", page.read_text(), re.S):
            request = json.loads(block)
            if "vllm_xargs" in request:
                XGraphLogitsProcessor.validate_params(
                    SimpleNamespace(
                        extra_args=request["vllm_xargs"],
                    )
                )
                checked.append(page)
    assert checked


def run_example(name, *arguments):
    script = ROOT / "examples" / name
    assert script.is_file(), f"the documented runnable example is missing: {script.name}"
    result = subprocess.run(
        [sys.executable, str(script), *map(str, arguments)],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    return json.loads(result.stdout)


def test_readme_quick_start_is_the_executed_example():
    """Copy/pasting the README runs the same standalone CPU program we test."""
    script = (ROOT / "examples/quick_start.py").read_text()
    readme = (ROOT / "README.md").read_text()
    assert re.findall(r"```python\n(.*?)```", readme, re.S) == [script]
    result = subprocess.run(
        [sys.executable, str(ROOT / "examples/quick_start.py")],
        cwd=ROOT,
        text=True,
        capture_output=True,
        timeout=30,
        check=True,
    )
    assert ast.literal_eval(result.stdout.strip()) == (("atlas", "creator", "ada"),)


def test_witness_example_records_the_actual_entity_chain():
    assert run_example("witnesses.py") == {
        "reached": ["Q3"],
        "uncapped_reached": ["Q3"],
        "cap_events": [],
        "witnesses": {"Q3": ["Q1", "Q2", "Q3"]},
    }


def test_string_id_versions_example_keeps_independent_readers(tmp_path):
    result = run_example("string_versions.py", "--out", tmp_path / "versions")
    assert result["base"] == result["base_again"] == ["place:cedar"]
    assert result["updated"] == ["place:delta"]
    assert result["deleted"] == []
    assert result["loaded_edits"] == {
        "updated": [["person:ada", "rel:born-in", "place:delta"]],
        "deleted": [["person:ada", "rel:born-in", None]],
    }


def test_sampling_loop_removes_the_highest_disallowed_logit():
    result = run_example("sampling_loop.py")
    assert result["triples"] == [["item:atlas", "rel:creator", "person:ada"]]
    assert result["blocked_steps"] > 0


def test_generation_example_executes_its_worker_call_with_a_cpu_stand_in(tmp_path, monkeypatch):
    import os
    import runpy
    from types import SimpleNamespace

    calls = {}

    class Runtime:
        def __init__(self, **kwargs):
            calls["llm"] = kwargs

        def chat(self, messages, *, sampling_params):
            calls["messages"] = messages
            calls["sampling"] = sampling_params
            return [SimpleNamespace(outputs=[SimpleNamespace(text="Answer: ⟦ Cedar⟧")])]

    monkeypatch.setitem(sys.modules, "vllm", SimpleNamespace(LLM=Runtime, SamplingParams=dict))
    monkeypatch.setattr(os, "environ", dict(os.environ))
    model = tmp_path / "model"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "vllm_generation.py",
            "--model",
            str(model),
            "--index",
            str(tmp_path / "index"),
            "--tokenizer",
            str(model / "tokenizer.json"),
            "--max-model-len",
            "16384",
            "--kv-cache-gib",
            "16",
        ],
    )
    runpy.run_path(str(ROOT / "examples/vllm_generation.py"), run_name="__main__")
    assert calls["llm"]["logits_processors"] == ["xgraph.vllm:XGraphLogitsProcessor"]
    assert calls["llm"]["kv_cache_memory_bytes"] == 16 * 1024**3
    assert calls["sampling"]["extra_args"] == {
        "xgraph": 1,
        "xgraph_anchor": ["item:atlas"],
    }
    assert calls["sampling"]["temperature"] == 0
    assert calls["sampling"]["seed"] == 0
    assert calls["messages"][-1]["content"] == "Where was the creator of Atlas born?"
    assert os.environ["VLLM_USE_V2_MODEL_RUNNER"] == "1"


def test_toy_example_resolves_spans_and_binds_the_answer(tmp_path):
    directory = tmp_path / "index"
    result = run_example("toy_kg.py", "--out", directory)
    assert result["entity"] == ["item:atlas"]
    assert result["fact"] == [["item:atlas", "rel:creator", "person:ada"]]
    assert result["path"] == ["place:cedar"]
    assert result["answer"] == ["place:cedar"]
    assert result["escape_resolved"] is False
    assert result["rejected_wrong_answer"] is True
    assert result["anchored_path"] == ["place:cedar"]
    assert result["rejected_anchors"] == [[], ["unknown:id"]]
    index = load_kg_index(directory)
    assert index.relations_of("item:atlas") == ["rel:creator"]
    assert {
        index.labels.owner_id(row) for row in index.object_rows("item:atlas", "rel:creator")
    } == {"person:ada"}


def test_versions_example_replaces_objects_and_pins_each_state(tmp_path):
    result = run_example("kg_versions.py", "--out", tmp_path / "versions")
    assert result["base"] == ["Q3"]
    assert result["edited"] == ["Q4"]
    assert result["base_after_edit"] == ["Q3"]
    assert result["rejected_old_answer"] is True
    assert result["loaded_edits"] == {"updated": [["Q2", "P2", "Q4"]]}
    assert result["unwritable_edits"] == 0


def test_rdf_example_preserves_iris_and_pins_replacement_and_pair_deletion(tmp_path):
    result = run_example("rdf_graph.py", "--out", tmp_path / "rdf")
    expected = [
        ["https://example.org/ada", "https://example.org/bornIn", "https://example.org/delta"]
    ]
    assert result["updated"] == result["exported"] == expected
    assert result["deleted_rows"] == []
    assert result["loaded_edits"]["deleted"] == [
        [
            "https://example.org/ada",
            "https://example.org/bornIn",
            None,
        ]
    ]
