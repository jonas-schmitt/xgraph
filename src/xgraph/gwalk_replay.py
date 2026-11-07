"""The decompose-then-execute replay: GWalk's recorded plans executed again, on the CPU.

A GWalk run (python -m xgraph.gwalk) records, per question, the subject it extracted, the
relations its breakdown named, and per hop the relation template the contriever snapped
the relation to, the subject the hop asked about, and the model's generated answer. The
replay keeps that decomposition and executes it again without a model call: from GWalk's
fitted subject, each hop's object comes from the first of the sources (--sources, in
order) that has one:

- `bank`: GWalk's edit bank for the case's setting (get_fact_form_kg, upstream's lookup),
  the all-edit bank with upstream's per-case gold-path mask (--bank all, as python -m
  xgraph.gwalk builds it), or the bank of the case's world, unmasked (--bank worlds,
  gwalk.world_banks, as xgraph.gwalk --worlds builds it);
- `kg`: the case's KG store (--kg-db; with '{}' in the path standing for the case's
  world, as for xgraph.gwalk --kg-db), or with --kg-versions one store read through the
  KG version of the case's world (toolkg.VersionedStore; the worlds as path mode reads
  them from one KG index), read by gwalk.KGLookup: the subject by label or alias, the
  template by the property the bank's id2rel gives it, of several entities or objects the
  best-connected one;
- `model`: GWalk's own generated answer for that hop, as upstream's loop parses it.

The default order, bank then KG then model, is what xgraph.gwalk --kg-db does at each
hop; `kg` alone is decompose-then-execute without the edit bank or the per-hop answers,
and bank then model replays the recorded walk itself. The object is fitted onto the
bank's entities as upstream's fit_subject_on_kg does and becomes the next hop's subject;
the question's answer is the last hop's. A hop no source answers ends the walk, and, as
upstream's loop leaves it at a failed hop, the answer is what the hop before reached
(none at the first hop).

A hop whose recorded answer failed to parse: upstream's order, which the real --kg-db
loop keeps, parses the model's answer before any lookup, so the hop ends the walk there
(--failed-hops stop, the default); with --failed-hops lookup the hop still gets the bank
and KG lookups, and the walk goes on past it. The hops the recorded walk never reached
take their templates from any record of the file that snapped the same relation (the
snap depends on the relation alone); a relation no record snapped ends the walk.

This is a replay of the recorded run, sampled as it was (upstream's temperatures, the
subject extracted once per case from all its questions), not a greedy run. It forecasts
the real --kg-db run but can differ from it in three ways: where the replay's object
differs from the recorded walk's, the later hops ask about another subject, whose model
answer the recording does not have; the replay then takes the recorded answer to the
recorded subject, marked stale. The contriever is not run: a name that is no bank
entity is fitted to the bank entity whose dataset aliases hold it, reusing the recorded
walk's fit of the same name where this bank allows it, and of several such entities the
first by name, marked ambiguous. And the bank of --bank worlds need not be the recorded
run's bank, which decided the recorded subjects.

    python -m xgraph.gwalk_replay --data data/mquake/MQuAKE-Remastered-CF-3k.json \\
        --records target/mqr-cf-3k-worlds/gwalk-fixed.jsonl --bank worlds \\
        --worlds target/mqr-cf-3k-worlds/worlds.json \\
        --kg-db 'data/wikidata/mqrw-kg-w{}.sqlite' --out target/mqr-cf-3k-worlds/replay.jsonl

Writes one record per recorded question, in the records' order and in GWalk's record
format (the transcript as text, one line per hop, ending with "Answer: ⟦ label⟧" when the
walk reached an answer), so that python -m xgraph.editing scores them like any arm's;
each hop names its object's source, and the record keeps the recorded answer, calls, and
tokens under `recorded`. <out>.summary.json counts hops by source, answers by the source
of their last hop, stale, diverged, unrecorded, and looked-up hops, subject fits, and
answers equal to the recorded ones, with the options and provenance.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path

from xgraph.gwalk import Bank, KGLookup

SOURCES = ("bank", "kg", "model")
FAILED_HOPS = ("stop", "lookup")
BANKS = ("all", "worlds")
ARM = "gwalk-replay"
NO_FACT = "<no fact>"


class Fitter:
    """Upstream's fit_subject_on_kg without the contriever: a name that is a bank entity
    is kept; otherwise the bank entity whose dataset aliases hold it, where the recorded
    walk's fit of the same name is taken when this bank allows it (the contriever chose
    it among its ten nearest entities), and of several the first by name."""

    def __init__(self, bank: Bank):
        self.bank = bank
        self.ents = set(bank.ents)
        self.by_alias: dict[str, list[str]] = {}
        for target in sorted(self.ents):
            for alias in bank.ent2alias.get(target, ()):
                self.by_alias.setdefault(alias, []).append(target)

    def fit(self, name: str, recorded: str | None = None) -> tuple[str, str]:
        """The fitted subject and how: entity, recorded, alias, ambiguous, or none."""
        if name in self.ents:
            return name, "entity"
        if recorded is not None and (
            recorded == name
            or (recorded in self.ents and name in self.bank.ent2alias.get(recorded, ()))
        ):
            return recorded, "recorded"
        found = self.by_alias.get(name, [])
        if found:
            return found[0], "alias" if len(found) == 1 else "ambiguous"
        return name, "none"


@dataclass
class CaseSetting:
    """What a case's hops read: its bank, the gold track the bank's mask hides another
    case's edits on (empty when unmasked), the bank's fitter, and its KG store, if any."""

    case_id: int
    bank: Bank
    track: list[tuple[str, str]]
    fitter: Fitter = field(repr=False)
    lookup: KGLookup | None = None

    @classmethod
    def of(
        cls,
        case: dict,
        bank: Bank,
        lookup: KGLookup | None,
        fitters: dict[int, Fitter] | None = None,
    ) -> CaseSetting:
        """A case's setting (upstream's case fields): every case is edited (all-edit), so
        the mask reads its edited chain, as upstream's get_correct_track does."""
        from xgraph.gwalk_upstream.mquake_utils import get_correct_track

        fitters = {} if fitters is None else fitters
        if id(bank) not in fitters:
            fitters[id(bank)] = Fitter(bank)
        track = get_correct_track(case, True, bank.id2rel) if bank.masked else []
        return cls(case["case_id"], bank, track, fitters[id(bank)], lookup)


def case_settings(
    dataset: list[dict],
    case_ids: Iterable[int],
    bank: str,
    worlds: dict[str, int] | None,
    kg_db: str | None,
    kg_versions: Path | None = None,
) -> dict[int, CaseSetting]:
    """Each case's setting: the all-edit bank, masked (bank "all"), or its world's bank
    (bank "worlds", which needs worlds), and its KG store (gwalk.kg_store_paths: '{}' in
    kg_db stands for the case's world), or with kg_versions (a version file, which needs
    worlds) the one store kg_db read through the version w<k> of the case's world k."""
    from xgraph.gwalk import build_bank, kg_store_paths, world_banks

    if bank not in BANKS:
        raise ValueError(f"bank {bank!r} is not one of {', '.join(BANKS)}")
    by_id = {case["case_id"]: case for case in dataset}
    wanted = sorted(set(case_ids))
    missing = [c for c in wanted if c not in by_id]
    if missing:
        raise ValueError(f"cases {missing[:5]} are not in the dataset")
    if bank == "worlds":
        if worlds is None:
            raise ValueError("world banks need the worlds map")
        banks = world_banks(dataset, worlds)
        outside = [c for c in wanted if c not in banks]
        if outside:
            raise ValueError(f"cases {outside[:5]} are in no world")
    else:
        everything = build_bank(dataset, set(by_id))
        banks = dict.fromkeys(wanted, everything)
    lookups: dict[int | None, KGLookup] = {}
    if kg_db is not None and kg_versions is not None:
        from xgraph.kgversions import read_kg_versions
        from xgraph.toolkg import KGStore, VersionedStore

        if worlds is None:
            raise ValueError("KG versions are the worlds' KGs: reading them needs the worlds map")
        versions = read_kg_versions(kg_versions)
        needed = sorted({worlds[f"mquake-{case}"] for case in wanted})
        absent = [f"w{world}" for world in needed if f"w{world}" not in versions]
        if absent:
            raise ValueError(f"{kg_versions} has no KG version {', '.join(absent)}")
        base = KGStore(kg_db)
        for world in needed:
            edits = [(int(s[1:]), int(p[1:]), int(o[1:])) for s, p, o in versions[f"w{world}"]]
            lookups[world] = KGLookup(kg_db, VersionedStore(base, edits))
    elif kg_db is not None:
        lookups = {w: KGLookup(path) for w, path in kg_store_paths(kg_db, worlds).items()}
    fitters: dict[int, Fitter] = {}
    settings = {}
    for case in wanted:
        lookup = None
        if lookups:
            lookup = lookups[None] if None in lookups else lookups[worlds[f"mquake-{case}"]]
        settings[case] = CaseSetting.of(by_id[case], banks[case], lookup, fitters)
    return settings


def record_case(record: dict) -> int:
    """The case of a record's question id, mquake-<case>-<k>."""
    parts = str(record.get("id", "")).split("-")
    if len(parts) != 3 or parts[0] != "mquake" or not parts[1].isdigit():
        raise ValueError(f"record id {record.get('id')!r} is not mquake-<case>-<k>")
    return int(parts[1])


def relation_templates(records: Iterable[dict]) -> dict[str, str | None]:
    """The template every recorded hop snapped its relation to (None: none retrieved);
    records that snap one relation to two templates are an error."""
    templates: dict[str, str | None] = {}
    for record in records:
        for hop in record["hops"]:
            relation, template = hop["relation"], hop["template"]
            if templates.setdefault(relation, template) != template:
                raise ValueError(
                    f"relation {relation!r} snapped to {templates[relation]!r} and to "
                    f"{template!r}: the records come from different banks or contrievers"
                )
    return templates


def model_object(hop: dict) -> str | None:
    """The object of a recorded hop's generated answer, as upstream's loop parses it, or
    None when the hop failed."""
    if hop.get("failed") or hop.get("generated") is None:
        return None
    segments = hop["generated"].strip().split(". ")
    return ". ".join(segments[1:]) if len(segments) >= 2 else None


def recorded_object(hop: dict) -> str | None:
    """What a recorded hop continued with before fitting: the retrieved fact's object
    when it contradicted the answer, else the model's."""
    if hop.get("failed") or "subject_after" not in hop:
        return None
    if hop.get("contradicted"):
        head = hop["template"].format(hop["subject"]) + " "
        fact = hop.get("retrieved_fact", "")
        return fact[len(head) :] if fact.startswith(head) else None
    return model_object(hop)


def replay_record(
    record: dict,
    setting: CaseSetting,
    templates: dict[str, str | None],
    sources: tuple[str, ...] = SOURCES,
    failed_hops: str = "stop",
) -> dict:
    """One recorded question replayed (the module docstring): its record."""
    from xgraph.gwalk_upstream.mquake_utils import get_fact_form_kg

    recorded_hops = record["hops"]
    relations = record.get("relations") or []
    fitter = setting.fitter
    subject, how = fitter.fit(
        record["subject"], recorded_hops[0]["subject"] if recorded_hops else None
    )
    error = None if relations else "breakdown"
    answer = None
    hops: list[dict] = []
    for i, relation in enumerate(relations):
        mine = recorded_hops[i] if i < len(recorded_hops) else None
        if mine is not None:
            template = mine["template"]
        elif relation in templates:
            template = templates[relation]
        else:
            hops.append({"relation": relation, "template": None, "subject": subject,
                         "recorded": False, "failed": "no_template"})  # fmt: skip
            error = "no_template"
            break
        hop: dict = {"relation": relation, "template": template, "subject": subject}
        hop["recorded"] = mine is not None
        hop["diverged"] = mine is not None and mine["subject"] != subject
        hops.append(hop)
        failed = mine.get("failed") if mine is not None else None
        if failed and (failed_hops == "stop" or template is None):
            hop["failed"] = error = failed
            break
        if failed:
            hop["parse_failed"] = True
        found = None
        for source in sources:
            if source == "bank" and template is not None:
                fact, contradicts, obj = get_fact_form_kg(
                    subject, template, setting.bank.kg, setting.case_id, setting.track
                )
                if contradicts:
                    found = obj
                    hop["retrieved_fact"] = fact
            elif source == "kg" and template is not None and setting.lookup is not None:
                fact = setting.lookup.fact(subject, template, setting.bank.id2rel)
                if fact is not None:
                    hop["retrieved_fact"], found, hop["kg_fact"] = fact
            elif source == "model" and mine is not None:
                found = model_object(mine)
                if found is not None and hop["diverged"]:
                    hop["stale"] = True
            if found is not None:
                hop["source"], hop["object"] = source, found
                break
        if found is None:
            hop["failed"] = error = "no_object"
            break
        same = mine is not None and recorded_object(mine) == found
        subject, how = fitter.fit(found, mine["subject_after"] if same else None)
        hop["subject_after"], hop["fit"] = subject, how
        answer = subject
    lines = [
        f"Hop {i + 1}: {h['subject']} · {h['template'] or h['relation']} → "
        + (f"{h['object']} ({h['source']})" if "object" in h else f"none ({h['failed']})")
        for i, h in enumerate(hops)
    ]
    return {
        "id": record["id"],
        "arm": ARM,
        "text": "\n".join(lines + ([f"Answer: ⟦ {answer}⟧"] if answer else [])),
        "answer": answer,
        "subject": record["subject"],
        "subject_fit": how,
        "relations": relations,
        "hops": hops,
        "error": error,
        "recorded": {
            key: record.get(key)
            for key in ("answer", "error", "calls", "prompt_tokens", "generated_tokens")
        },
    }


def replay(
    records: list[dict],
    settings: dict[int, CaseSetting],
    sources: tuple[str, ...] = SOURCES,
    failed_hops: str = "stop",
    templates: dict[str, str | None] | None = None,
) -> list[dict]:
    """Every recorded question replayed, in order. templates: each relation's template
    for hops the recorded walk never reached (default: relation_templates(records))."""
    if not sources or len(set(sources)) != len(sources) or not set(sources) <= set(SOURCES):
        raise ValueError(f"sources {sources}: each of {', '.join(SOURCES)} at most once")
    if failed_hops not in FAILED_HOPS:
        raise ValueError(f"failed_hops {failed_hops!r} is not one of {', '.join(FAILED_HOPS)}")
    templates = relation_templates(records) if templates is None else templates
    return [
        replay_record(record, settings[record_case(record)], templates, sources, failed_hops)
        for record in records
    ]


def summarize(replayed: list[dict]) -> dict:
    """Counts by source and of what the replay could not take from the recording."""
    hops = [hop for record in replayed for hop in record["hops"]]
    answered = [r for r in replayed if r["answer"]]
    last = Counter(
        next(h["source"] for h in reversed(r["hops"]) if "source" in h) for r in answered
    )
    return {
        "cases": len({record_case(r) for r in replayed}),
        "questions": len(replayed),
        "answered": len(answered),
        "same_answer_as_recorded": sum(r["answer"] == r["recorded"]["answer"] for r in replayed),
        "answers_by_source": {s: last[s] for s in SOURCES},
        "errors": dict(sorted(Counter(str(r["error"]) for r in replayed).items())),
        "hops": {
            "walked": len(hops),
            "by_source": {s: sum(h.get("source") == s for h in hops) for s in SOURCES},
            "failed": dict(sorted(Counter(h["failed"] for h in hops if "failed" in h).items())),
            "diverged": sum(bool(h.get("diverged")) for h in hops),
            "stale": sum(bool(h.get("stale")) for h in hops),
            "unrecorded": sum(not h["recorded"] for h in hops),
            "parse_failed_looked_up": sum(bool(h.get("parse_failed")) for h in hops),
        },
        "fits": dict(
            sorted(
                Counter(
                    [r["subject_fit"] for r in replayed] + [h["fit"] for h in hops if "fit" in h]
                ).items()
            )
        ),
    }


def main(argv: list[str] | None = None) -> None:
    from xgraph.gwalk import kg_store_paths, store_identity, upstream_case
    from xgraph.kgversions import version_files
    from xgraph.provenance import provenance

    parser = argparse.ArgumentParser(prog="python -m xgraph.gwalk_replay", description=__doc__)
    parser.add_argument("--data", type=Path, required=True, help="MQuAKE-Remastered JSON")
    parser.add_argument(
        "--records", type=Path, required=True, help="a GWalk run's records (xgraph.gwalk)"
    )
    parser.add_argument("--out", type=Path, required=True, help="replayed records (JSONL)")
    parser.add_argument(
        "--bank",
        choices=BANKS,
        default="all",
        help="the all-edit bank with upstream's mask, or each case's world bank (--worlds)",
    )
    parser.add_argument("--worlds", type=Path, help="worlds.json (python -m xgraph.mquake)")
    parser.add_argument(
        "--kg-db",
        help="a KG store (python -m xgraph.toolkg build); with --worlds, '{}' in the path "
        "stands for the case's world",
    )
    parser.add_argument(
        "--kg-versions",
        type=Path,
        help="with --worlds: a KG version file (python -m xgraph.mquake --kg-versions); each "
        "case reads the --kg-db store through its world's version w<k>",
    )
    parser.add_argument(
        "--sources",
        default=",".join(SOURCES),
        help="where a hop's object comes from, in order: some of bank, kg, model, comma "
        "separated (default: %(default)s)",
    )
    parser.add_argument(
        "--failed-hops",
        choices=FAILED_HOPS,
        default="stop",
        help="a hop whose recorded answer failed to parse ends the walk (stop, upstream's "
        "order) or still gets the lookups (lookup)",
    )
    args = parser.parse_args(argv)
    sources = tuple(s.strip() for s in args.sources.split(","))
    if len(set(sources)) != len(sources) or not set(sources) <= set(SOURCES):
        raise SystemExit(
            f"--sources {args.sources}: name each of {', '.join(SOURCES)} at most once"
        )
    if args.bank == "worlds" and args.worlds is None:
        raise SystemExit("--bank worlds needs --worlds")
    if ("kg" in sources) != (args.kg_db is not None):
        raise SystemExit("--sources names kg exactly when --kg-db gives the KG store")
    worlds = json.loads(args.worlds.read_text(encoding="utf-8")) if args.worlds else None
    if args.kg_versions is not None:
        if worlds is None:
            raise SystemExit("--kg-versions holds the worlds' KG versions: it needs --worlds")
        if args.kg_db is None:
            raise SystemExit("--kg-versions reads its versions over the --kg-db store")
        if "{}" in args.kg_db:
            raise SystemExit(
                f"--kg-db {args.kg_db}: with --kg-versions every case reads one store through "
                "its world's version, so the path holds no '{}'"
            )
        stores = {None: Path(args.kg_db)}
        if not stores[None].is_file():
            raise SystemExit(f"--kg-db: {args.kg_db}, the KG store, is missing")
    else:
        stores = kg_store_paths(args.kg_db, worlds) if args.kg_db else {}

    dataset = [upstream_case(case) for case in json.loads(args.data.read_text(encoding="utf-8"))]
    records = [json.loads(line) for line in args.records.read_text(encoding="utf-8").splitlines()]
    known = {case["case_id"] for case in dataset}
    unknown = [r["id"] for r in records if record_case(r) not in known]
    if unknown:
        raise SystemExit(f"{args.records}: records {unknown[:5]} name cases not in {args.data}")
    try:
        settings = case_settings(
            dataset,
            {record_case(r) for r in records},
            args.bank,
            worlds,
            args.kg_db,
            args.kg_versions,
        )
    except ValueError as error:
        raise SystemExit(f"{args.records}: {error}") from error
    replayed = replay(records, settings, sources, args.failed_hops)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as out:
        for record in replayed:
            out.write(json.dumps(record, ensure_ascii=False) + "\n")
    summary = {
        "artifact_kind": "xgraph.gwalk_replay",
        "version": 1,
        "arm": ARM,
        **summarize(replayed),
        "sources": list(sources),
        "failed_hops": args.failed_hops,
        "bank": args.bank,
        "worlds": str(args.worlds) if args.worlds else None,
        "kg_db": args.kg_db,
        "kg_versions": str(args.kg_versions) if args.kg_versions else None,
        "kg_stores": [store_identity(path) for path in stores.values()],
        "provenance": provenance(
            [args.data, args.records]
            + ([args.worlds] if args.worlds else [])
            + (version_files(args.kg_versions) if args.kg_versions else [])
        ),
    }
    Path(f"{args.out}.summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    print(json.dumps({k: summary[k] for k in ("questions", "answered", "answers_by_source")}))


if __name__ == "__main__":
    main()
