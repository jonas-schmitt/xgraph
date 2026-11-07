"""Owner-only repair-visibility exports; dry runs never read sealed inputs.

Materializes sources without changing the registered study. Historical source
pins remain in immutable parent registrations.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import stat
import subprocess
import tempfile
from datetime import UTC, datetime
from pathlib import Path

from xgraph import repair_visibility as study
from xgraph.menu_control import identity
from xgraph.mintqa_panels import indexed, read_jsonl, write_json
from xgraph.provenance import provenance, sha256_file


def _one(values, description):
    unique = {json.dumps(v, sort_keys=True): v for v in values}
    if len(unique) != 1:
        raise ValueError(f"expected one unambiguous {description}; found {len(unique)}")
    return next(iter(unique.values()))


def _pin(row):
    return {k: row[k] for k in ("path", "bytes", "sha256")}


def _check(pin):
    if identity(pin["path"]) != _pin(pin):
        raise ValueError(f"changed pinned input: {pin['path']}")


def _json(pin):
    _check(pin)
    return json.loads(Path(pin["path"]).read_text())


def _option(argv, key):
    if argv.count(key) != 1:
        raise ValueError(f"expected one {key} in registered argv")
    values = []
    for value in argv[argv.index(key) + 1 :]:
        if value.startswith("--"):
            break
        values.append(value)
    if not values:
        raise ValueError(f"missing {key} value in registered argv")
    return values


def _metadata(out, *, clean=True):
    registration = study.verify(out, require_clean=clean)
    web_pin = registration["confirmation"]["eligible_population_registration"]
    web = _json(web_pin)
    if sorted(web["population"]["intersection"]) != registration["confirmation"]["eligible_ids"]:
        raise ValueError("WebQSP registration has a different eligible population")
    audit = web["question_cluster_audit"]
    if set(audit["columns"]) != {"id", "question", "q_entity"}:
        raise ValueError("registered projection must contain only id/question/q_entity")
    declared = _option(audit["command"], "--questions")
    questions = []
    for name in declared:
        matches = [
            _pin(p)
            for p in audit["inputs"]
            if Path(p["path"]).parts[-len(Path(name).parts) :] == Path(name).parts
        ]
        questions.append(_one(matches, f"question projection {name}"))
    if len({p["path"] for p in questions}) != len(questions):
        raise ValueError("duplicate pinned projection paths")
    execution = Path(web["config"]["out"]) / "execution/execution.json"
    return registration, web, questions, execution


def cohort_projection(registration_path, execution, questions, destination):
    out = Path(registration_path).resolve().parent
    if Path(registration_path).name != "registration.json":
        raise ValueError("expected the repair visibility registration.json")
    registration, web, pins, expected = _metadata(out)
    if Path(execution).resolve() != expected.resolve():
        raise ValueError("execution path differs from the registered WebQSP destination")
    if [str(Path(p).resolve()) for p in questions] != [p["path"] for p in pins]:
        raise ValueError("questions must name exactly the pinned projection shards in order")
    rows = []
    for pin in pins:
        _check(pin)
        rows.extend(read_jsonl(Path(pin["path"])))
    if not rows or any(
        set(r) != {"id", "question", "q_entity"}
        or not isinstance(r["question"], str)
        or not isinstance(r["q_entity"], list)
        or any(not isinstance(t, str) for t in r["q_entity"])
        for r in rows
    ):
        raise ValueError(
            "projection needs only id/question/q_entity; gold, answers and outputs forbidden"
        )
    table = indexed(rows)
    if set(table) != set(web["population"]["rog"]):
        raise ValueError("missing or extra ids in the pinned RoG question projection")
    core = json.loads(Path(execution).read_text())
    ids = core["ids"]
    if (
        core.get("artifact_kind") != "xgraph.webqsp_test_execution"
        or core.get("registration")
        != registration["confirmation"]["eligible_population_registration"]
        or ids != study.select_ids(registration["confirmation"]["eligible_ids"], len(ids))
        or not set(ids) <= set(table)
    ):
        raise ValueError("execution must name the registered complete core ids and registration")
    destination = Path(destination)
    if destination.exists():
        if json.loads(destination.read_text()) != [table[q] for q in ids]:
            raise ValueError("cohort projection is immutable")
        return
    write_json(destination, [table[q] for q in ids])


def _input_pin(inputs, path):
    return _one([_pin(p) for p in inputs if p["path"] == str(path)], f"pin for {path}")


def _pilot_answer_sources(registration):
    """Annotation identities from panel metadata; never read scoring gold here."""
    panels = _json(registration["pilot"]["population"])
    return [
        _one(
            [
                _pin(p)
                for p in panels["provenance"]["inputs"]
                if Path(p["path"]).name == f"{split}.jsonl"
            ],
            f"registered {split} answer source",
        )
        for split in ("ti", "pop")
    ]


def _source_plan(out, *, clean=True):
    """Follow only metadata edges; never traverse WebQSP jobs or sealed outputs."""
    registration, web, questions, execution = _metadata(out, clean=clean)
    final_pin = _one(
        [_pin(p) for p in web["inputs"] if Path(p["path"]).name == "freeze-manifest.json"],
        "final-validation recipe manifest",
    )
    final = _json(final_pin)
    if final["artifact_kind"] != "xgraph.mintqa_validation.freeze":
        raise ValueError("expected the frozen final-validation recipes")
    panels_pin = registration["pilot"]["population"]
    if (
        final["panels"] != panels_pin["path"]
        or _input_pin(final["inputs"], final["panels"]) != panels_pin
    ):
        raise ValueError("final-validation recipes name a different pilot population")
    panels = _json(panels_pin)
    specs, golds = {}, []
    for family in registration["families"]:
        first = registration["pilot"]["existing_W"][family]
        recipe = final["families"][family]
        inventory = [*final["inputs"], *recipe["model"]["files"]]
        step = _one([s for s in recipe["steps"] if s["output"] == first], f"{family} W recipe")
        requests = step["requests"]
        if requests != recipe["prompt_demo_set"]["base_requests"]:
            raise ValueError("W recipe must use the original registered requests")
        spec = {
            "first": {"path": first},
            "first_provenance": {"path": first + ".provenance.json"},
            "requests": _input_pin(final["inputs"], requests),
        }
        for field, variable in (
            ("db", "XG_TOOLS_DB"),
            ("relations", "XG_RELATION_LABELS"),
            ("tokenizer", "XG_TOKENIZER"),
            ("index", "XG_KG_INDEX"),
        ):
            value = _one(
                [w.split("=", 1)[1] for w in step["command"] if w.startswith(variable + "=")],
                f"{family} {variable}",
            )
            if not Path(value).is_absolute():
                raise ValueError(f"recipe path must be absolute: {value}")
            spec[field] = value if field == "index" else _input_pin(inventory, value)
        root = Path(spec["index"])
        spec["index_files"] = [
            _pin(p)
            for p in final["inputs"]
            if Path(p["path"]).parent == root
            and (Path(p["path"]).name == "kg-index.json" or Path(p["path"]).suffix == ".npy")
        ]
        expected = {str(root / "kg-index.json"), *map(str, root.glob("*.npy"))}
        if {p["path"] for p in spec["index_files"]} != expected:
            raise ValueError("index seal must cover its header and every native table")
        gold = _one(
            [
                _input_pin(final["inputs"], p["path"])
                for p in panels["outputs"]
                if Path(p["path"]).parent == Path(requests).parent
                and Path(p["path"]).suffix == ".json"
                and p["path"] in step["inputs"]
                and p["sha256"] == _input_pin(final["inputs"], p["path"])["sha256"]
            ],
            f"{family} registered gold path",
        )
        golds.append(gold)
        specs[family] = spec
    gold = _one(golds, "common pilot gold")
    metadata = [
        registration["confirmation"]["eligible_population_registration"],
        final_pin,
        panels_pin,
    ]
    return registration, specs, gold, metadata, questions, execution


def sources(out):
    """Owner invocation only: hash W histories in place; retain the registered gold pin."""
    out = Path(out).resolve()
    if (out / "pilot/sources.json").exists():
        raise ValueError("source export is immutable")
    registration, specs, gold, metadata, _, _ = _source_plan(out)
    cohort_pin = identity(out / "confirmation/cohort.json")
    study._read_cohort(out, registration, cohort_pin)
    answer_sources = _pilot_answer_sources(registration)
    for pin in [gold, *answer_sources]:
        if Path(pin["path"]).stat().st_size != pin["bytes"]:
            raise ValueError("registered gold size changed")
    checked = set()
    for spec in specs.values():
        for field in ("first", "first_provenance"):
            spec[field] = identity(spec[field]["path"])
        for pin in [spec[k] for k in ("requests", "db", "relations", "tokenizer")] + spec[
            "index_files"
        ]:
            if pin["path"] not in checked:
                _check(pin)
                checked.add(pin["path"])
    write_json(
        out / "pilot/sources.json",
        {
            "artifact_kind": "xgraph.repair_visibility_sources",
            "registration_sha256": sha256_file(out / "registration.json"),
            "population_ids": registration["pilot"]["population_ids"],
            "members": registration["pilot"]["members"],
            "gold": gold,
            "answer_sources": answer_sources,
            "families": specs,
            "confirmation_cohort": cohort_pin,
            "source_metadata": metadata,
            "provenance": provenance(
                [out / "registration.json", Path(__file__), *(p["path"] for p in metadata)]
            ),
        },
    )


def _validated_sources(out):
    registration, expected, gold, metadata, _, _ = _source_plan(out)
    seal = json.loads((out / "pilot/sources.json").read_text())
    if seal["registration_sha256"] != sha256_file(out / "registration.json"):
        raise ValueError("source export names a different registration")
    if (
        seal["population_ids"] != registration["pilot"]["population_ids"]
        or seal["members"] != registration["pilot"]["members"]
        or seal["gold"] != gold
        or seal["answer_sources"] != _pilot_answer_sources(registration)
        or seal["source_metadata"] != metadata
        or set(seal["families"]) != set(expected)
    ):
        raise ValueError("source export differs from the registered pilot metadata")
    if seal["confirmation_cohort"] != identity(out / "confirmation/cohort.json"):
        raise ValueError("confirmation cohort pin changed")
    study._read_cohort(out, registration, seal["confirmation_cohort"])
    checked = set()
    for family, spec in expected.items():
        actual = seal["families"][family]
        for field, pin in spec.items():
            if field in ("first", "first_provenance"):
                value = actual[field]
                if (
                    value["path"] != pin["path"]
                    or type(value["bytes"]) is not int
                    or len(value["sha256"]) != 64
                    or Path(value["path"]).stat().st_size != value["bytes"]
                ):
                    raise ValueError(f"sealed source identity changed: {field}")
                _check(value)
            elif actual[field] != pin:
                raise ValueError(f"source {family}/{field} differs from its registered recipe")
        for pin in [spec[k] for k in ("requests", "db", "relations", "tokenizer")] + spec[
            "index_files"
        ]:
            if pin["path"] not in checked:
                _check(pin)
                checked.add(pin["path"])
    for pin in [gold, *seal["answer_sources"]]:
        if Path(pin["path"]).stat().st_size != pin["bytes"]:
            raise ValueError("registered gold size changed")
    return registration, seal


def _template(out, registration):
    row = json.loads((out / "owner-decision.template.json").read_text())
    gate = registration["pilot_authorization"]
    expected = {
        "registration_sha256": sha256_file(out / "registration.json"),
        "stride_score_sha256": gate["score"]["sha256"],
        "stride_receipt_sha256": gate["receipt"]["sha256"],
    }
    if (
        any(row.get(k) != v for k, v in expected.items())
        or any(k.endswith("_sha256") and k not in {*expected, "sources_sha256"} for k in row)
        or not str(row.get("sources_sha256", "")).startswith("<")
    ):
        raise ValueError("owner decision template hashes changed")
    if (
        row.get("approved") is not False
        or row.get("owner") != ""
        or row.get("phase") != "pilot"
        or row.get("stride_passes") is not False
    ):
        raise ValueError("owner decision template defaults changed")
    return row


def decision(out, owner, *, approve=False):
    if not approve:
        raise ValueError("decision requires explicit --approve")
    if not owner or not owner.strip():
        raise ValueError("decision requires the actual owner name")
    out = Path(out).resolve()
    destination = out / "owner-decision.json"
    if destination.exists():
        raise ValueError("owner decision is immutable")
    registration, _ = _validated_sources(out)
    row = _template(out, registration)
    row.update(
        owner=owner.strip(),
        approved=True,
        decided_at=datetime.now(UTC).isoformat(),
        sources_sha256=sha256_file(out / "pilot/sources.json"),
    )
    # Validate before publication; failed authorization never leaves an approval.
    with tempfile.TemporaryDirectory(prefix=".owner-decision-", dir=out) as temporary:
        pending = Path(temporary) / "decision.json"
        write_json(pending, row)
        study.authorize(out, "pilot", pending)
        destination.hardlink_to(pending)


def _commands(out, questions, execution, owner):
    python = [str(study.REPO / ".venv/bin/python"), "-I", "-m"]
    export = [*python, "xgraph.repair_visibility_export"]
    projection = out / "confirmation/questions.json"
    return [
        [
            *export,
            "cohort-projection",
            "--registration",
            str(out / "registration.json"),
            "--execution",
            str(execution),
            "--questions",
            *(p["path"] for p in questions),
            "--out",
            str(projection),
        ],
        [
            *python,
            "xgraph.repair_visibility",
            "freeze-cohort",
            "--out",
            str(out),
            "--questions",
            str(projection),
            "--core-execution",
            str(execution),
        ],
        [*export, "sources", "--out", str(out), "--phase", "pilot"],
        [*export, "decision", "--out", str(out), "--owner", owner or "<owner>", "--approve"],
    ]


def preview(out, owner=None):
    out = Path(out).resolve()
    registration, specs, gold, metadata, questions, execution = _source_plan(out, clean=False)
    _template(out, registration)
    print("# STAT-ONLY preview: no sealed W/provenance, gold or execution seal is opened or hashed")
    paths = {p["path"] for p in [gold, *_pilot_answer_sources(registration), *metadata, *questions]}
    paths.update(p["path"] for p in registration["inputs"])
    paths.update(registration["pilot_authorization"][key]["path"] for key in ("score", "receipt"))
    paths.add(str(Path(__file__).resolve()))
    paths.update(
        str(out / name)
        for name in (
            "registration.json",
            "registration.sha256",
            "owner-decision.template.json",
            "confirmation/questions.json",
            "confirmation/cohort.json",
            "pilot/sources.json",
            "owner-decision.json",
            "owner-argv.txt",
            "owner-argv.json",
            "rvpilotq.sh",
        )
    )
    paths.add(str(execution))
    for spec in specs.values():
        paths.add(spec["index"])
        paths.update(p["path"] for p in spec["index_files"])
        paths.update(
            spec[k]["path"]
            for k in ("first", "first_provenance", "requests", "db", "relations", "tokenizer")
        )
    for path in sorted(paths):
        try:
            info = Path(path).stat()
            state = f"{'file' if stat.S_ISREG(info.st_mode) else 'directory'} bytes={info.st_size}"
        except FileNotFoundError:
            state = "MISSING (owner prerequisite or prospective output)"
        print(f"# stat {path}: {state}")
    for command in _commands(out, questions, execution, owner):
        print(shlex.join(["nice", "-n", "10", "prlimit", "--as=4000000000", *command]))
    print(shlex.join(["bash", str(out / "rvpilotq.sh")]))
    print("# owner-argv.txt (scheduler exec after prerequisite verification)")
    print((out / "owner-argv.txt").read_text().strip())
    if "stages" in registration:
        study.queue(out)


def all_steps(out, owner, *, approve=False):
    if not approve or not owner or not owner.strip():
        raise ValueError("all requires --owner and explicit --approve")
    out = Path(out).resolve()
    registration, _, _, _, questions, execution = _source_plan(out)
    _template(out, registration)
    projection = out / "confirmation/questions.json"
    cohort_projection(
        out / "registration.json", execution, [p["path"] for p in questions], projection
    )
    cohort = out / "confirmation/cohort.json"
    if cohort.exists():
        frozen = study._read_cohort(out, registration, identity(cohort))
        if frozen["question_only_input"] != identity(projection) or frozen[
            "core_execution"
        ] != identity(execution):
            raise ValueError("existing cohort binds a different projection or execution")
    else:
        study.freeze_cohort(out, projection, execution)
    if (out / "pilot/sources.json").exists():
        _validated_sources(out)
    else:
        sources(out)
    if (out / "owner-decision.json").exists():
        row = study.authorize(out, "pilot", out / "owner-decision.json")
        if row["owner"] != owner.strip():
            raise ValueError("existing decision names a different owner")
    else:
        decision(out, owner, approve=approve)


def _owner_argv(out):
    argv = shlex.split((out / "owner-argv.txt").read_text())
    owner_decision = str(out / "owner-decision.json")
    expected = [
        "env",
        f"XG_REPAIR_VISIBILITY_DECISION={owner_decision}",
        "bash",
        str(out / "repair-visibility.sh"),
        "--phase",
        "pilot",
        "--run",
        "--decision",
        owner_decision,
    ]
    if argv != expected or argv != json.loads((out / "owner-argv.json").read_text()):
        raise ValueError("owner argv differs from the exact registered pilot wrapper")
    return argv


def launch(out, *, dry_run=False):
    out = Path(out).resolve()
    if dry_run:
        preview(out)
        print("# scheduler exec: " + shlex.join(_owner_argv(out)))
        return
    try:
        registration = study.verify(out)
        _template(out, registration)
        study.authorize(out, "pilot", out / "owner-decision.json")
        _validated_sources(out)
        argv = _owner_argv(out)
    except (
        ValueError,
        OSError,
        KeyError,
        TypeError,
        AttributeError,
        subprocess.CalledProcessError,
    ) as error:
        log = study.SOURCE / "target/queue/repair-visibility.log"
        log.parent.mkdir(parents=True, exist_ok=True)
        detail = " ".join(str(error).splitlines())
        with log.open("a") as stream:
            stream.write(
                f"repair-visibility pilot REFUSED {datetime.now(UTC).isoformat()} {detail}\n"
            )
        raise SystemExit(3) from None
    os.execvpe(argv[0], argv, os.environ.copy())


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    projection = commands.add_parser("cohort-projection")
    projection.add_argument("--registration", type=Path, required=True)
    projection.add_argument("--execution", type=Path, required=True)
    projection.add_argument("--questions", type=Path, nargs="+", required=True)
    projection.add_argument("--out", type=Path, required=True)
    source = commands.add_parser("sources")
    source.add_argument("--out", type=Path, required=True)
    source.add_argument("--phase", choices=("pilot",), required=True)
    approval = commands.add_parser("decision")
    approval.add_argument("--out", type=Path, required=True)
    approval.add_argument("--owner", required=True)
    approval.add_argument("--approve", action="store_true")
    whole = commands.add_parser("all")
    whole.add_argument("--out", type=Path, required=True)
    whole.add_argument("--owner")
    whole.add_argument("--approve", action="store_true")
    whole.add_argument("--dry-run", "--check", dest="dry_run", action="store_true")
    scheduler = commands.add_parser("launch")
    scheduler.add_argument("--out", type=Path, required=True)
    scheduler.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if args.command == "cohort-projection":
        cohort_projection(args.registration, args.execution, args.questions, args.out)
    elif args.command == "sources":
        sources(args.out)
    elif args.command == "decision":
        decision(args.out, args.owner, approve=args.approve)
    elif args.command == "launch":
        launch(args.out, dry_run=args.dry_run)
    elif args.dry_run:
        preview(args.out, args.owner)
    else:
        all_steps(args.out, args.owner, approve=args.approve)


if __name__ == "__main__":
    main()
