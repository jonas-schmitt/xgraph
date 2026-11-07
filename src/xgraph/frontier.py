"""The accuracy-cost frontier: every arm on one benchmark, accuracy against measured cost.

Each arm is given as `name=records.jsonl`, the records a generation run (`xgraph.generate`,
`scripts/generate.sh`) or a CPU stage over runs wrote, optionally followed by further cost
sources: `name=records.jsonl+run.jsonl+...`. Per arm the frontier reports

- accuracy under the benchmark's scorer, over all questions and per split and group;
- cost per question: the sums over the arm's cost sources divided by the questions the
  arm's records answer (one record each);
- the paired difference against the reference arm (`--reference`), arm minus reference,
  with a bootstrap interval that resamples whole clusters of correlated questions;
- whether the arm is on the Pareto front for the cost axis (`--axis`): no other arm is at
  least as accurate and at least as cheap, and better on one of the two.

**Cost sources.** The records file is the first cost source, the `+`-joined files the rest.
Each source's `<source>.summary.json` says what it cost:

- a generation summary with `seconds` is a run: its `seconds` (workload wall time,
  model loading excluded), `prompt_tokens` and `generated_tokens` (a number, or per
  stage with a `total`; without them, the sums of the records' fields: `generated_tokens`,
  else the length of `token_ids`), and, from its records, the model calls (`calls` with a
  `total` in the summary, else per record `rounds`, a GWalk record's `calls`, a walk
  record's `replies`, else one call, two for a fallback record) and the tool calls (an
  agent record's `tool_calls`, an iterative record's `lookups`). Multi-turn records already
  sum their tokens over rounds, and so do their summaries;
- a summary with `cost_kind: cpu_stage` adds its own `seconds`, `cpu_seconds` and
  `peak_rss_mib`, without counting the generation fields copied into its records. Peak
  RSS is the process high-water mark; a composite takes the largest recorded peak;
- a summary without `seconds` is an untimed stage: its time is not counted, and the arm
  says so in `not_counted`. Historical summaries remain readable;
- records without a summary (a concatenation of runs, a matched execution that wrote
  none) are derived: their cost is that of the runs they were built from, declared as
  further sources; without any, the arm's cost is not measured. A further source without a
  summary is refused.

A composite arm thus lists the runs it consumed: matched execution of free paths
(`free=<free>.matched.jsonl+<free>.jsonl`), delivered plans with the generation that
delivered them, a concatenation with its parts. Per question costs are `seconds`,
`cpu_seconds`, `prompt_tokens`, `generated_tokens`, `tokens` (both), `model_calls`, and
`tool_calls` (null
where no source's records name any); `budget_spent_share` is the share of records saying
their call budget ran out, among those that say.

`distributions` gives p50/p95 and counts of known per-question prompt tokens, generated
tokens and model calls. The sources' per-question costs are joined by id and summed
before taking quantiles, including calls made for only part of the questions. Missing
record usage stays unknown. When a summary total disagrees with complete record usage,
or source ids cannot be assigned to the arm's questions, `distribution_notes` explains
why that metric's composite distribution is unavailable. Workload wall seconds and
uncontrolled or unknown cache state do not establish controlled GPU-seconds or latency.
An aggregate model-usage metric also stays unknown if any contributing generation run
lacks it; CPU stages contribute no model usage. Unknown costs are not Pareto-classified.

**Benchmarks.** A scorer gives accuracy per question set and the paired comparison;
`mintqa` is MINTQA's own measure (`xgraph.mintqa_bench.paper_accuracy` and `compare_arms`,
clustered by relation sequence).

    python -m xgraph.frontier mintqa \\
        --arm paths=<paths>.jsonl --arm closed=<closed-book>.jsonl \\
        --arm free=<free>.matched.jsonl+<free>.jsonl --reference paths \\
        --questions target/mintqa/questions.json \\
        --split ti=data/external/mintqa-full/questions/ti.jsonl \\
        --split pop=data/external/mintqa-full/questions/pop.jsonl \\
        --out target/frontier/mintqa.json

writes the JSON report to `--out` and one row per arm to the same path with `.csv`.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from xgraph.cost import COST_DISTRIBUTIONS, distributions, request_cost
from xgraph.score import record_text

HISTORICAL_AXES = ("seconds", "prompt_tokens", "generated_tokens", "tokens", "model_calls")
SERVING_AXES = (
    "serving_seconds",
    "gpu_seconds",
    "latency_p50",
    "latency_p95",
    "latency_p99",
    "ttft_p50",
    "ttft_p95",
    "ttft_p99",
)
AXES = (*HISTORICAL_AXES, *SERVING_AXES)
"""Cost axes the Pareto front can be taken on."""
MEASURES = (*HISTORICAL_AXES, "cpu_seconds", "tool_calls")


@dataclass(frozen=True)
class Arm:
    name: str
    records: Path
    costs: tuple[Path, ...]


def parse_arm(spec: str) -> Arm:
    """`name=records.jsonl[+cost-source.jsonl...]` as an Arm."""
    name, _, files = spec.partition("=")
    paths = files.split("+")
    if not name or not files or not all(paths):
        raise ValueError(f"an arm is name=records.jsonl[+cost-source.jsonl...], not {spec!r}")
    return Arm(name, Path(paths[0]), tuple(Path(p) for p in paths[1:]))


def read_records(path: Path) -> list[dict]:
    """A records file's records, refusing two records for one question."""
    with path.open(encoding="utf-8") as lines:
        records = [json.loads(line) for line in lines if line.strip()]
    seen: set[str] = set()
    for record in records:
        if record["id"] in seen:
            raise ValueError(f"{path}: question {record['id']!r} has more than one record")
        seen.add(record["id"])
    return records


def _summary(path: Path) -> dict | None:
    found = Path(f"{path}.summary.json")
    return json.loads(found.read_text(encoding="utf-8")) if found.exists() else None


def _total(value: int | dict | None) -> int | None:
    return value.get("total") if isinstance(value, dict) else value


def _source_kind(summary: dict | None) -> str:
    if summary is None:
        return "none"
    if summary.get("seconds") is None:
        return "derivation"
    return "cpu_stage" if summary.get("cost_kind") == "cpu_stage" else "run"


def _tool_calls(record: dict) -> int | None:
    if "tool_calls" in record:  # the KG agent
        return record["tool_calls"]
    if isinstance(record.get("lookups"), list):  # iterative retrieval
        return len(record["lookups"])
    return None


def source_cost(path: Path, records: list[dict] | None = None) -> dict:
    """What one cost source cost in total (see the module docstring); `kind` is `run`,
    `cpu_stage`, `derivation` (a summary without seconds), or `none` (no summary)."""
    summary = _summary(path)
    kind = _source_kind(summary)
    if kind == "none":
        return {"path": str(path), "kind": "none"}
    if kind == "derivation":
        artifact = summary.get("artifact_kind", "a summary without artifact_kind")
        return {"path": str(path), "kind": kind, "artifact_kind": artifact}
    if kind == "cpu_stage":
        return {
            "path": str(path),
            "kind": "cpu_stage",
            "artifact_kind": summary.get("artifact_kind"),
            "seconds": summary["seconds"],
            "cpu_seconds": summary.get("cpu_seconds"),
            "peak_rss_mib": summary.get("peak_rss_mib"),
            "memory_scope": summary.get("memory_scope"),
        }
    records = read_records(path) if records is None else records
    costs = [request_cost(r) for r in records]

    def record_total(key: str) -> int | None:
        values = [r[key] for r in costs]
        return None if None in values else sum(values)

    prompt = _total(summary.get("prompt_tokens"))
    if prompt is None:
        prompt = record_total("prompt_tokens")
    generated = _total(summary.get("generated_tokens"))
    if generated is None:
        generated = record_total("generated_tokens")
    calls = _total(summary.get("model_calls", summary.get("calls")))
    tools = [t for t in map(_tool_calls, records) if t is not None]
    budget = [bool(r["budget_spent"]) for r in records if "budget_spent" in r]
    return {
        "path": str(path),
        "kind": "run",
        "records": len(records),
        "requests": summary.get("requests", summary.get("questions")),
        "seconds": summary["seconds"],
        "timing_kind": summary.get("timing_kind", "workload_wall"),
        "cache_state": summary.get("cache_state", "unknown"),
        "prefix_caching": summary.get("engine_args", {}).get("enable_prefix_caching"),
        "cpu_seconds": summary.get("cpu_seconds"),
        "peak_rss_mib": summary.get("peak_rss_mib"),
        "prompt_tokens": prompt,
        "generated_tokens": generated,
        "model_calls": calls if calls is not None else record_total("model_calls"),
        "tool_calls": sum(tools) if tools else None,
        "budget_spent": sum(budget) if budget else None,
        "budget_known": len(budget),
        "distributions": distributions(costs),
    }


def arm_cost(arm: Arm, records: list[dict] | None = None) -> dict:
    """The arm's cost per question over its cost sources (see the module docstring)."""
    records = read_records(arm.records) if records is None else records
    sources = [source_cost(arm.records, records)]
    requests = [(sources[0], {r["id"]: request_cost(r) for r in records})]
    for path in arm.costs:
        loaded = read_records(path) if _source_kind(_summary(path)) == "run" else []
        cost = source_cost(path, loaded)
        if cost["kind"] == "none":
            raise ValueError(f"cost source {path} of arm {arm.name!r} has no summary")
        sources.append(cost)
        requests.append((cost, {r["id"]: request_cost(r) for r in loaded}))
    not_counted = []
    for source in sources:
        if source["kind"] == "none":
            not_counted.append(
                f"{source['path']} has no summary: its own cost is not counted, only that "
                "of the runs declared after it"
            )
        elif source["kind"] == "derivation":
            not_counted.append(
                f"{source['path']} ({source['artifact_kind']}) records no seconds: its CPU "
                "time is not counted"
            )
    runs = [s for s in sources if s["kind"] == "run"]
    measured = [s for s in sources if s["kind"] in ("run", "cpu_stage")]
    questions = len(records)

    def total(key: str) -> float | None:
        if key in COST_DISTRIBUTIONS and any(s[key] is None for s in runs):
            return None
        values = [s[key] for s in measured if s.get(key) is not None]
        return sum(values) if values else None

    def per_question(key: str) -> float | None:
        value = total(key)
        return value / questions if value is not None and questions else None

    per = {key: per_question(key) for key in MEASURES if key != "tokens"}
    prompt, generated = per["prompt_tokens"], per["generated_tokens"]
    per["tokens"] = None if prompt is None or generated is None else prompt + generated
    known = sum(s["budget_known"] for s in runs)
    spent = sum(s["budget_spent"] or 0 for s in runs)
    question_costs = {
        r["id"]: dict.fromkeys(COST_DISTRIBUTIONS, 0 if runs else None) for r in records
    }
    distribution_notes = []
    for source, usage in requests:
        if source["kind"] != "run":
            continue
        for key in COST_DISTRIBUTIONS:
            values = [r[key] for r in usage.values()]
            mismatch = None not in values and source[key] is not None and sum(values) != source[key]
            foreign = usage.keys() - question_costs.keys()
            if mismatch or foreign:
                distribution_notes.append(
                    f"{source['path']}: {key} distribution unavailable: "
                    + (
                        "record sum differs from summary total"
                        if mismatch
                        else "source has question ids outside this arm"
                    )
                )
            for qid, totals in question_costs.items():
                value = usage.get(qid, {}).get(key, 0)
                if mismatch or foreign or value is None:
                    totals[key] = None
                elif totals[key] is not None:
                    totals[key] += value
    return {
        "questions": questions,
        "sources": sources,
        "totals": {
            **{
                key: total(key)
                for key in (
                    "seconds",
                    "cpu_seconds",
                    "prompt_tokens",
                    "generated_tokens",
                    "model_calls",
                )
            },
            "peak_rss_mib": max(
                (s["peak_rss_mib"] for s in measured if s.get("peak_rss_mib") is not None),
                default=None,
            ),
        },
        "per_question": {key: per[key] for key in MEASURES},
        "budget_spent_share": spent / known if known else None,
        "distributions": distributions(question_costs.values()),
        "distribution_notes": distribution_notes,
        "not_counted": not_counted,
    }


def pareto(points: dict[str, tuple[float, float | None]]) -> dict[str, bool | None]:
    """Per name of (accuracy, cost): on the front unless another point is at least as
    accurate and at least as cheap and better on one; None where the cost is unknown."""
    known = {n: p for n, p in points.items() if p[1] is not None}
    front: dict[str, bool | None] = {}
    for name, (accuracy, cost) in points.items():
        if cost is None:
            front[name] = None
            continue
        front[name] = not any(
            other != name
            and a >= accuracy
            and c <= cost  # type: ignore[operator]
            and (a > accuracy or c < cost)  # type: ignore[operator]
            for other, (a, c) in known.items()
        )
    return front


@dataclass
class Scorer:
    """A benchmark's scoring: accuracy (`all`, `by_split`, `by_group` as
    xgraph.mintqa_bench.paper_accuracy gives them) and the paired comparison of two arms'
    texts by question id (per split and `all`, each with `difference`, `low`, `high`)."""

    name: str
    accuracy: Callable[[dict[str, str]], dict]
    compare: Callable[[dict[str, str], dict[str, str]], dict]
    inputs: list[Path]


def mintqa_scorer(
    questions: Path, splits: list[str], resamples: int, ids: set[str] | None = None
) -> Scorer:
    """MINTQA's own measure, compared clustered by relation sequence."""
    from xgraph.mintqa_bench import compare_arms, dataset_rows, paper_accuracy, relation_clusters

    rows = dataset_rows(splits)
    answers = {qid: row["answer"] for qid, row in rows.items()}
    asked = json.loads(questions.read_text(encoding="utf-8"))
    if ids is not None:
        asked = [q for q in asked if q["id"] in ids]
        if {q["id"] for q in asked} != ids:
            raise ValueError("controlled serving panel is not contained in --questions")
    clusters = relation_clusters(rows)
    return Scorer(
        "mintqa",
        lambda texts: paper_accuracy(texts, answers, asked),
        lambda a, b: compare_arms(a, b, answers, asked, resamples, clusters),
        [questions, *(Path(s.partition("=")[2]) for s in splits)],
    )


def controlled_cost(report: dict, arm: str, ids: set[str], cache: str, concurrency: int) -> dict:
    """Merge exactly one repeated serving cell; never add repetitions as serial stages."""
    if (
        report.get("artifact_kind") != "xgraph.serving_study_report"
        or report.get("status") != "completed"
    ):
        raise ValueError("a controlled cost merge needs a completed serving study report")
    if set(report["panel_ids"]) != ids:
        raise ValueError(f"arm {arm}: controlled serving and accuracy must use the same panel ids")
    cells = [
        c
        for c in report["cells"]
        if (c["arm"], c["cache"], c["concurrency"]) == (arm, cache, concurrency)
    ]
    if len(cells) != 1 or cells[0]["status"] != "completed":
        raise ValueError(
            f"arm {arm}: need one completed serving cell for "
            f"cache={cache}, concurrency={concurrency}"
        )
    cell = cells[0]
    if cell["metrics"].get("serving_seconds", {}).get("trials", 0) < 2:
        raise ValueError(f"arm {arm}: controlled serving needs repeated workload measurements")
    per = {key: None for key in (*MEASURES, *SERVING_AXES)}
    for key, stats in cell["metrics"].items():
        if key in per:
            per[key] = stats["mean"]
    if per["prompt_tokens"] is not None and per["generated_tokens"] is not None:
        per["tokens"] = per["prompt_tokens"] + per["generated_tokens"]
    return {
        "questions": len(ids),
        "per_question": per,
        "controlled_serving": cell,
        "sources": [
            {"path": p, "kind": "controlled_serving", "cache_state": cache}
            for p in cell["trial_outputs"]
        ],
        "totals": {"peak_rss_mib": None},
        "budget_spent_share": None,
        "distributions": {},
        "distribution_notes": [],
        "not_counted": [],
        "accuracy_scope": (
            "declared frozen answer records on the timing panel; "
            "replay does not re-evaluate policy accuracy"
        ),
    }


def frontier(
    arms: list[Arm],
    scorer: Scorer,
    reference: str,
    axis: str,
    *,
    serving_report: dict | None = None,
    serving_cache: str = "off",
    serving_concurrency: int = 1,
) -> dict:
    """Accuracy, cost, the comparison with the reference arm, and the Pareto front."""
    names = [a.name for a in arms]
    if len(set(names)) != len(names):
        raise ValueError(f"arm names repeat: {names}")
    if reference not in names:
        raise ValueError(f"reference {reference!r} is not one of the arms {names}")
    if axis not in AXES:
        raise ValueError(f"unknown cost axis {axis!r}; expected one of {AXES}")
    texts, rows = {}, []
    for arm in arms:
        records = read_records(arm.records)
        if serving_report is not None:
            if arm.costs:
                raise ValueError(
                    "controlled serving needs one frozen trajectory; "
                    "historical composite sources cannot be added"
                )
            panel = set(serving_report["panel_ids"])
            records = [r for r in records if r["id"] in panel]
        texts[arm.name] = {r["id"]: record_text(r) for r in records}
        rows.append(
            {
                "name": arm.name,
                "records": str(arm.records),
                "cost_sources": [str(p) for p in arm.costs],
                "accuracy": scorer.accuracy(texts[arm.name]),
                "cost": arm_cost(arm, records)
                if serving_report is None
                else controlled_cost(
                    serving_report,
                    arm.name,
                    set(texts[arm.name]),
                    serving_cache,
                    serving_concurrency,
                ),
            }
        )
    for row in rows:
        name = row["name"]
        row["versus_reference"] = (
            None if name == reference else scorer.compare(texts[name], texts[reference])
        )
    front = pareto(
        {
            r["name"]: (r["accuracy"]["all"]["accuracy"], r["cost"]["per_question"].get(axis))
            for r in rows
        }
    )
    for row in rows:
        row["on_front"] = front[row["name"]]
    return {
        "artifact_kind": "xgraph.frontier",
        "version": 1,
        "benchmark": scorer.name,
        "reference": reference,
        "axis": axis,
        "arms": rows,
    }


def table(result: dict) -> list[dict]:
    """One flat row per arm, for plotting."""
    splits = sorted({s for arm in result["arms"] for s in arm["accuracy"]["by_split"]})
    out = []
    for arm in result["arms"]:
        versus = arm["versus_reference"] or {}
        overall = versus.get("all", {})
        row = {
            "arm": arm["name"],
            "questions": arm["accuracy"]["all"]["questions"],
            "accuracy": arm["accuracy"]["all"]["accuracy"],
            **{
                f"accuracy_{s}": arm["accuracy"]["by_split"].get(s, {}).get("accuracy")
                for s in splits
            },
            "difference": overall.get("difference"),
            "low": overall.get("low"),
            "high": overall.get("high"),
            **{f"{k}_per_question": v for k, v in arm["cost"]["per_question"].items()},
            **{
                f"{key}_{stat}": value
                for key, stats in arm["cost"]["distributions"].items()
                for stat, value in stats.items()
            },
            "peak_rss_mib": arm["cost"]["totals"]["peak_rss_mib"],
            "budget_spent_share": arm["cost"]["budget_spent_share"],
            "on_front": arm["on_front"],
            "cost_measured": arm["cost"]["per_question"].get(result["axis"]) is not None,
        }
        out.append({k: "" if v is None else v for k, v in row.items()})
    return out


def main(argv: list[str] | None = None) -> None:
    from xgraph.provenance import provenance

    parser = argparse.ArgumentParser(prog="python -m xgraph.frontier", description=__doc__)
    benchmarks = parser.add_subparsers(dest="benchmark", required=True)
    mintqa = benchmarks.add_parser("mintqa", help="MINTQA's own measure")
    mintqa.add_argument("--questions", type=Path, required=True, help="questions.json (all)")
    mintqa.add_argument(
        "--split", action="append", required=True, help="<ti|pop>=<questions (JSON lines)>"
    )
    for sub in (mintqa,):
        sub.add_argument(
            "--arm",
            action="append",
            required=True,
            help="name=records.jsonl[+cost-source.jsonl...]",
        )
        sub.add_argument("--reference", required=True, help="the arm the others are compared with")
        sub.add_argument("--axis", default="seconds", choices=AXES, help="the Pareto front's cost")
        sub.add_argument("--resamples", type=int, default=10_000, help="bootstrap resamples")
        sub.add_argument("--out", type=Path, required=True, help="report (JSON); .csv beside it")
        sub.add_argument(
            "--serving-report",
            type=Path,
            help="controlled study report.json; restrict accuracy to its panel",
        )
        sub.add_argument("--serving-cache", choices=("off", "cold", "warm"), default="off")
        sub.add_argument("--serving-concurrency", type=int, default=1)
    args = parser.parse_args(argv)
    try:
        arms = [parse_arm(spec) for spec in args.arm]
    except ValueError as error:
        parser.error(str(error))
    if args.reference not in {a.name for a in arms}:
        parser.error(f"--reference {args.reference!r} names no --arm")
    serving = json.loads(args.serving_report.read_text()) if args.serving_report else None
    if args.axis in SERVING_AXES and serving is None:
        parser.error(f"--axis {args.axis} needs --serving-report")
    scorer = mintqa_scorer(
        args.questions,
        args.split,
        args.resamples,
        set(serving["panel_ids"]) if serving is not None else None,
    )
    result = frontier(
        arms,
        scorer,
        args.reference,
        args.axis,
        serving_report=serving,
        serving_cache=args.serving_cache,
        serving_concurrency=args.serving_concurrency,
    )
    inputs = []
    for arm in arms:
        for path in (arm.records, *arm.costs):
            inputs.append(path)
            if Path(f"{path}.summary.json").exists():
                inputs.append(Path(f"{path}.summary.json"))
    result["provenance"] = provenance(
        list(
            dict.fromkeys(
                [*inputs, *scorer.inputs, *([args.serving_report] if args.serving_report else [])]
            )
        )
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=1) + "\n", encoding="utf-8")
    rows = table(result)
    with args.out.with_suffix(".csv").open("w", newline="", encoding="utf-8") as written:
        writer = csv.DictWriter(written, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    for row in rows:
        print(
            json.dumps({k: row[k] for k in ("arm", "accuracy", "seconds_per_question", "on_front")})
        )


if __name__ == "__main__":
    main()
