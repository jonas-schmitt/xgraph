"""JSON handoff between fresh GCR planner and synthesis processes."""

from __future__ import annotations

import hashlib
import json
import os
import signal
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def binding(args):
    """Bind reuse to inputs, model settings and the actual stage implementation."""
    inputs = [args.requests, args.tools_db, args.kg_versions, getattr(args, "gcr_prompts", None)]
    if args.kg_versions:
        from xgraph.kgversions import version_files

        inputs += version_files(args.kg_versions)
    for model in (args.model, args.gcr_synthesis_model):
        if model:
            inputs += [
                Path(model) / name
                for name in (
                    "config.json",
                    "generation_config.json",
                    "tokenizer.json",
                    "tokenizer_config.json",
                    "chat_template.jinja",
                    "special_tokens_map.json",
                )
            ]
    inputs += [
        Path(__file__),
        Path(__file__).with_name("gcr.py"),
        Path(__file__).with_name("generate.py"),
    ]
    return {
        "arguments": {
            k: str(v) if isinstance(v, Path) else v
            for k, v in vars(args).items()
            if k not in ("gcr_stage", "gcr_handoff", "provenance_only")
        },
        "inputs": {str(p): digest(p) for p in inputs if p is not None and Path(p).is_file()},
    }


def save_path_states(states, config, path, contract):
    """Save model-free state and literal prompts; no pickle or executable input."""
    excluded = {"read", "trie", "arguments", "path_texts", "prompts", "path_hops"}
    rows = [
        {
            "state": {k: v for k, v in vars(s).items() if k not in excluded},
            "messages": s.messages("synthesis") if s.text is None else None,
            "prompt_metadata": s.prompts.metadata() if getattr(s, "prompts", None) else None,
        }
        for s in states
    ]
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        json.dump(
            {"version": 1, "binding": contract, "config": asdict(config), "states": rows}, stream
        )


def seal_handoff(path):
    Path(str(path) + ".sha256").write_text(digest(path) + "\n")


def load_handoff(path, contract):
    path = Path(path)
    if Path(str(path) + ".sha256").read_text().strip() != digest(path):
        raise ValueError("GCR path handoff changed")
    saved = json.loads(path.read_text())
    if saved["version"] != 1 or saved["binding"] != contract:
        raise ValueError("GCR path handoff has different inputs or settings")
    return saved


def synthesize(llm, saved, config, params_class):
    """Use the original state's fit, parsing and accounting methods after exec."""
    from xgraph.gcr import _State, chat_token_ids

    if saved["config"] != asdict(config):
        raise ValueError("GCR path handoff changed the decoder configuration")
    states, jobs = [], []
    tokenizer = llm.get_tokenizer() if llm is not None else None
    for row in saved["states"]:
        state = _State.__new__(_State)
        state.__dict__.update(row["state"])
        if row["prompt_metadata"] is not None:
            state.prompts = SimpleNamespace(metadata=lambda row=row: row["prompt_metadata"])
        else:
            state.prompts = None
        states.append(state)
        if state.text is not None:
            continue
        if config.max_prompt_tokens is not None and (
            len(chat_token_ids(tokenizer, row["messages"])) > config.max_prompt_tokens
        ):
            state.unanswered("context_limit", "length")
            state.trace[-1]["stage"] = "synthesis"
            continue
        jobs.append((state, row["messages"]))
    if jobs:
        params = [
            params_class(
                max_tokens=config.max_tokens,
                temperature=0.0,
                extra_args=None,
                repetition_penalty=config.repetition_penalty,
            )
            for _ in jobs
        ]
        outputs = llm.chat(
            [messages for _, messages in jobs],
            params,
            chat_template_kwargs={"enable_thinking": False},
            use_tqdm=False,
        )
        for (state, _), output in zip(jobs, outputs, strict=True):
            state.accept("synthesis", output)
    return [s.record(config) for s in states]


def coordinate(argv, args):
    """The parent never loads a model; a completed, verified planner survives retries."""
    handoff = args.gcr_handoff or Path(str(args.out) + ".gcr-paths.json")
    base = [sys.executable, "-m", "xgraph.generate", *argv]
    if Path(handoff).exists():
        load_handoff(handoff, binding(args))
    else:
        run_process(
            [*base, "--gcr-stage", "paths", "--gcr-handoff", str(handoff)],
            Path(str(args.out) + ".gcr-paths.log"),
        )
        load_handoff(handoff, binding(args))
    run_process(
        [*base, "--gcr-stage", "synthesis", "--gcr-handoff", str(handoff)],
        Path(str(args.out) + ".gcr-synthesis.log"),
    )


def run_process(command, log_path, *, silence_seconds=300, wall_seconds=86400):
    """One LLM per process, bounded startup silence, retained logs on failure."""
    from contextlib import suppress

    from xgraph.load_progress import wait_for_load

    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w") as log:
        log.write("[xgraph-load] begin process\n")
        log.flush()
        process = subprocess.Popen(
            command,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            env={**os.environ, "PYTHONUNBUFFERED": "1"},
        )
        try:
            code = wait_for_load(process, log_path, silence_seconds, wall_seconds)
        except BaseException:
            with suppress(ProcessLookupError):
                os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                with suppress(ProcessLookupError):
                    os.killpg(process.pid, signal.SIGKILL)
                process.wait()
            raise
        finally:
            with log_path.open() as retained:
                for line in retained:
                    print(line, end="", flush=True)
    if code:
        raise subprocess.CalledProcessError(code, command)


def finish_summary(args, summary, startup_seconds):
    """Keep the legacy workload boundary and disclose both process startup costs."""
    summary["startup_seconds"] = startup_seconds
    summary["gcr"]["process_stage"] = args.gcr_stage
    if args.gcr_stage == "synthesis":
        paths = json.loads(Path(str(args.out) + ".paths.jsonl.summary.json").read_text())
        summary["phase_workload_seconds"] = {
            "paths": paths["seconds"],
            "synthesis": summary["seconds"],
        }
        summary["phase_startup_seconds"] = {
            "paths": paths["startup_seconds"],
            "synthesis": startup_seconds,
        }
        summary["gcr"]["path_engine_args"] = paths["engine_args"]
        summary["seconds"] += paths["seconds"] + startup_seconds
