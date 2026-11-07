"""Opt-in development-only MINTQA size study and CPU reports."""

from __future__ import annotations

import argparse
import difflib
import fcntl
import hashlib
import json
import shlex
from pathlib import Path

from xgraph.mintqa_indexes import file_inventory
from xgraph.mintqa_panels import write_json
from xgraph.mintqa_queue import (
    FAMILIES,
    REPO,
    SHARED_GEMMA,
    Step,
    arm_steps,
    check_observe_support,
    cpu,
    family_index,
    gemma_check_step,
    run_step,
)
from xgraph.provenance import provenance, sha256_file

SCALE_FAMILIES = (*SHARED_GEMMA, "gemma31")
SCALE_ARMS = ("paths-allfix", "paths-allfix-stop", "agent-strong-8", "menuobs", "menuobs-audit")


def scale_score_step(args, family):
    prepared = args.source_root / "target/mintqa-rematch"
    directory = args.out / family
    output = directory / "report.json"
    data = args.source_root / "data/external/mintqa-full"
    return Step(
        cpu(
            "xgraph.mintqa_rematch",
            "score",
            "--panels",
            prepared / "panels/panels.json",
            "--family",
            family,
            "--directory",
            directory,
            "--data",
            data,
            "--panel",
            "development",
            *(arg for arm in SCALE_ARMS for arg in ("--arm", arm)),
            "--seed",
            17,
            "--resamples",
            args.resamples,
            "--out",
            output,
        ),
        output,
        inputs=[
            prepared / "panels/panels.json",
            prepared / "panels/development/questions.json",
            *(directory / f"development/{arm}.jsonl" for arm in SCALE_ARMS),
            *(data / f"questions/{split}.jsonl" for split in ("ti", "pop")),
        ],
        refresh=True,
    )


def scale_steps(args, family):
    prepared = args.source_root / "target/mintqa-rematch"
    yield gemma_check_step(args, family, family_index(args, family, prepared))
    # Set controls explicitly so inherited menu/observed switches cannot alter arms.
    settings = argparse.Namespace(
        **vars(args),
        menu=5,
        observe_select=False,
        observe_required=False,
        observe_no_loop=False,
        audit=False,
        audit_arms=["audit"],
        audit_menu_size=8,
        source=None,
        source_arm_name=None,
    )
    yield from arm_steps(settings, family, "development", [], prepared=prepared, scale=True)
    yield scale_score_step(args, family)


def qualify_gemma(family: str, models: Path, index: Path) -> dict:
    """Verify the shared index without building one; inspect local serving metadata."""
    from tokenizers import Tokenizer
    from transformers.utils.chat_template_utils import render_jinja_template

    manifest_path = index.with_suffix(".manifest.json")
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("settings") != {
        "unique_aliases": True,
        "verdicts": False,
        "syntax": "default",
    }:
        raise ValueError(f"shared index settings differ: {manifest_path}")
    model = models / FAMILIES[family][0]
    reference = models / FAMILIES["gemma31"][0]
    tokenizer_path = model / "tokenizer.json"
    inputs = [manifest_path]
    # A raw-file SHA is not the normalized ServedTokenizer fingerprint.
    recorded = [
        row["sha256"]
        for row in manifest["provenance"]["inputs"]
        if Path(row["path"]).name == "tokenizer.json"
    ]
    raw_hash = sha256_file(tokenizer_path)
    if len(recorded) != 1 or raw_hash != recorded[0]:
        raise ValueError(f"tokenizer SHA-256 differs from shared index manifest: {tokenizer_path}")
    if sha256_file(reference / "tokenizer.json") != raw_hash:
        raise ValueError(f"reference tokenizer differs from shared index manifest: {reference}")
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    fingerprint = hashlib.sha256(tokenizer.to_str().encode()).hexdigest()
    if fingerprint != manifest["tokenizer_fingerprint"]:
        raise ValueError(f"tokenizer fingerprint differs from shared index: {tokenizer_path}")
    if file_inventory(index) != manifest["files"]:
        raise ValueError(f"shared index files differ from manifest: {index}")
    if tokenizer.token_to_id("<|channel>") is None:
        raise ValueError(f"tokenizer lacks the Gemma channel token: {tokenizer_path}")
    configs, templates, prompts, architectures = {}, {}, {}, {}
    for name, directory in (("model", model), ("gemma31", reference)):
        paths = [
            directory / f
            for f in (
                "tokenizer.json",
                "chat_template.jinja",
                "generation_config.json",
                "tokenizer_config.json",
                "config.json",
            )
        ]
        inputs.extend(paths)
        templates[name] = paths[1].read_text()
        configs[name] = json.loads(paths[2].read_text())
        special = json.loads(paths[3].read_text())
        architectures[name] = json.loads(paths[4].read_text())
        rendered, _ = render_jinja_template(
            conversations=[[{"role": "user", "content": "Scale study check."}]],
            chat_template=templates[name],
            add_generation_prompt=True,
            enable_thinking=False,
            **{k: v for k, v in special.items() if k.endswith("_token")},
        )
        prompts[name] = rendered[0]
        if prompts[name].count("<|channel>thought") != prompts[name].count("<channel|>"):
            raise ValueError(
                f"thinking-off chat template leaves a thought channel open: {directory}"
            )
    eos = configs["model"].get("eos_token_id")
    eos = eos if isinstance(eos, list) else [eos]
    reference_eos = configs["gemma31"].get("eos_token_id")
    reference_eos = reference_eos if isinstance(reference_eos, list) else [reference_eos]
    if (
        not eos
        or eos != reference_eos
        or any(type(t) is not int or not 0 <= t < tokenizer.get_vocab_size() for t in eos)
    ):
        raise ValueError(f"EOS tokens differ from Gemma31 or are invalid: {model}")
    return {
        "artifact_kind": "xgraph.mintqa_scale_compatibility",
        "version": 1,
        "family": family,
        "index": str(index),
        "tokenizer_sha256": raw_hash,
        "tokenizer_fingerprint": fingerprint,
        "generation": configs["model"],
        "architectures": {
            name: {
                k: v
                for k, v in config.items()
                if k in ("architectures", "model_type", "dtype", "quantization_config")
            }
            for name, config in architectures.items()
        },
        "differences": {
            "chat_template": "".join(
                difflib.unified_diff(
                    templates["gemma31"].splitlines(True),
                    templates["model"].splitlines(True),
                    fromfile="gemma31",
                    tofile=family,
                )
            ),
            "generation": {
                key: {name: config.get(key) for name, config in configs.items()}
                for key in sorted(configs["model"].keys() | configs["gemma31"].keys())
                if configs["model"].get(key) != configs["gemma31"].get(key)
            },
            "thinking_off_prompt": prompts,
        },
        "serving": {
            "enable_thinking": False,
            "ban_tokens": ["<|channel>"],
            "generation_config": "vllm",
            "temperature": 0.0,
            "eos_token_id": eos,
            "kv_cache_gib": 8 if family in SHARED_GEMMA else 16,
            "need_gib": FAMILIES[family][2],
        },
        "weight_bytes": sum(p.stat().st_size for p in model.glob("*.safetensors")),
        "provenance": provenance([*dict.fromkeys(inputs), *sorted(index.iterdir())]),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="action", required=True)
    qualify = subparsers.add_parser("qualify", help="CPU-only tokenizer and chat-template checks")
    qualify.add_argument("family", choices=(*SHARED_GEMMA, "gemma31"))
    qualify.add_argument("--models", type=Path, default=Path("/home/jonas/models"))
    qualify.add_argument("--index", type=Path, required=True)
    qualify.add_argument("--out", type=Path, required=True)
    for action in ("run", "score"):
        command = subparsers.add_parser(action)
        command.add_argument("family", choices=(*SCALE_FAMILIES, "all"), nargs="?", default="all")
        command.add_argument("--out", type=Path, default=REPO / "target/mintqa-scale")
        command.add_argument("--source-root", type=Path, default=Path("/home/jonas/git/xgraph"))
        command.add_argument("--models", type=Path, default=Path("/home/jonas/models"))
        command.add_argument("--resamples", type=int, default=10_000)
        command.add_argument("--dry-run", action="store_true")
    summary = subparsers.add_parser("summary", help="CPU-only table across Gemma and Qwen families")
    summary.add_argument("--out", type=Path, default=REPO / "target/mintqa-scale")
    summary.add_argument("--source-root", type=Path, default=Path("/home/jonas/git/xgraph"))
    summary.add_argument("--source", action="append", default=[], metavar="FAMILY:ARM=JSONL")
    summary.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    args.out = args.out.resolve()
    if not (args.out.is_relative_to(REPO) or args.out.is_relative_to(Path("/tmp"))):
        parser.error("outputs must be in this checkout or /tmp")
    if args.action == "qualify":
        write_json(
            args.out, qualify_gemma(args.family, args.models.resolve(), args.index.resolve())
        )
        return
    if args.action == "summary":
        from xgraph.mintqa_scale_summary import parse_sources, write_summary

        sources = parse_sources(args.source)
        if args.dry_run:
            print(
                shlex.join(
                    cpu(
                        "xgraph.mintqa_scale",
                        "summary",
                        "--out",
                        args.out,
                        "--source-root",
                        args.source_root,
                        *(arg for spec in args.source for arg in ("--source", spec)),
                    )
                )
            )
        else:
            write_summary(args.out, args.source_root.resolve(), sources)
        return
    if args.resamples < 1:
        parser.error("--resamples must be positive")
    args.source_root, args.models = args.source_root.resolve(), args.models.resolve()
    prepared = args.source_root / "target/mintqa-rematch"
    if args.out.is_relative_to(prepared) or prepared.is_relative_to(args.out):
        parser.error("scale outputs must be separate from the read-only prepared inputs")
    families = SCALE_FAMILIES if args.family == "all" else [args.family]
    steps = [
        step
        for family in families
        for step in (
            scale_steps(args, family) if args.action == "run" else [scale_score_step(args, family)]
        )
    ]
    if args.dry_run:
        for step in steps:
            print(shlex.join(step.command))
        return
    if args.action == "run":
        check_observe_support(REPO / "scripts/generate.sh", [3], required=True, no_loop=True)
        for item in json.loads((prepared / "panels/panels.json").read_text())["outputs"]:
            path = Path(item["path"])
            if path.parent.name == "development" and sha256_file(path) != item["sha256"]:
                raise ValueError(f"frozen development panel changed: {path}")
    args.out.mkdir(parents=True, exist_ok=True)
    with (args.out / "queue.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        for step in steps:
            run_step(step)


if __name__ == "__main__":
    main()
