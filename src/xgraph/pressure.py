"""The rejected mass: how much of the model's probability the constraint takes away.

At each step the constraint renormalizes the model's next-token distribution over the
allowed set. The log of the allowed mass, log Z_t = log sum_{v allowed} p(v), is the
step's KL divergence from that renormalization (KL(q || p) = -log Z_t), and its sum
over a span is the log importance weight of the span under the unconstrained model. A
step where the model wants a token the KG rejects has log Z_t far below zero: the
constraint is pushing it somewhere it would not go.

Method pilot 3 (docs/research-plan.md) asks whether this pressure detects unsupported
fact spans, wrong answers, and loops better than the model's own confidence. This module
replays generated outputs through the model, masks each reply position with the engine
(xgraph.policy.mask_reply), and records per token the log allowed mass, the log
probability of the token taken under the constrained distribution, and the entropy of
the unconstrained one, with the token range of every span. The forward pass runs where
the model is (scripts/pressure-dump.sh, the pinned vLLM container).

    python -m xgraph.pressure dump --model /model --requests requests.jsonl \\
        --outputs B.jsonl --index /index --tokenizer /tokenizer.json --out B.pressure.jsonl
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from xgraph.syntax import SpanSyntax

DEFAULT_SYNTAX = SpanSyntax()


def step_scores(engine, logits, reply: list[int]) -> dict:
    """Per reply position i (predicting reply[i] from reply[:i]): the log allowed mass,
    the constrained log probability of reply[i], the unconstrained entropy, and whether
    the model's own top token is allowed.

    logits is [len(reply), columns] float on the GPU; it is left unchanged.
    """
    import torch

    from xgraph.policy import mask_reply

    raw = logits.float()
    log_p = raw.log_softmax(-1)
    masked = mask_reply(engine, raw.clone(), reply)
    targets = torch.tensor(reply, device=logits.device)
    return {
        "log_mass": masked.logsumexp(-1) - raw.logsumexp(-1),
        "log_prob": masked.log_softmax(-1).gather(1, targets[:, None])[:, 0],
        "entropy": -(log_p.exp() * log_p).nan_to_num().sum(-1),
        "top_allowed": ~masked.gather(1, raw.argmax(-1)[:, None])[:, 0].isneginf(),
    }


def spans(tokenizer, reply: list[int], syntax: SpanSyntax) -> list[dict]:
    """Every closed span of a reply: its kind (entity, fact, or escaped), its text, and
    the token range [start, stop) from the token holding its opener to the one ending
    its closer."""
    ends, data = [], b""
    for token in reply:
        data += tokenizer.token_bytes(token) or b""
        ends.append(len(data))
    opener, closer = syntax.opener.encode(), syntax.closer.encode()
    separator, escape = syntax.separator.encode(), syntax.escape.encode()
    pattern = re.compile(re.escape(opener) + b".*?" + re.escape(closer), re.S)
    # Keep the original token coordinates for pressure, while ignoring auxiliary
    # displays/decisions. Rewriting the path text would invalidate those coordinates.
    headers = [
        f"\n{heading}: ".encode()
        for heading in ("Executed", "Relations (terminal)", "Relations (penultimate)", "Decision")
    ]
    audit_at = data.find(headers[0] + opener)
    from xgraph.menus import auxiliary_ranges, semantic_text

    text = data.decode("utf-8", "replace")
    searchable = bytearray(data)
    for lo, hi in auxiliary_ranges(text, syntax):
        first, last = len(text[:lo].encode()), len(text[:hi].encode())
        searchable[first:last] = b" " * (last - first)
    found = []
    for match in pattern.finditer(searchable):
        at = match.start()
        if (
            audit_at >= 0
            and at >= audit_at
            and any(data[max(0, at - len(h)) : at] == h for h in headers)
        ):
            continue
        body = semantic_text(
            data[match.start() : match.end()].decode("utf-8", "replace"), syntax
        ).encode()
        kind = "escaped" if escape in body else "fact" if separator in body else "entity"
        start = next(i for i, end in enumerate(ends) if end > match.start())
        stop = next(i for i, end in enumerate(ends) if end >= match.end()) + 1
        found.append(
            {"kind": kind, "text": body.decode("utf-8", "replace"), "start": start, "stop": stop}
        )
    return found


def auroc(positives: list[float], negatives: list[float]) -> float | None:
    """The probability that a positive scores above a negative, ties counting half."""
    if not positives or not negatives:
        return None
    ranked = sorted([(v, 1) for v in positives] + [(v, 0) for v in negatives])
    rank_sum, i = 0.0, 0
    while i < len(ranked):  # average ranks over ties
        j = i
        while j < len(ranked) and ranked[j][0] == ranked[i][0]:
            j += 1
        rank_sum += (i + j + 1) / 2 * sum(label for _, label in ranked[i:j])
        i = j
    n, m = len(positives), len(negatives)
    return (rank_sum - n * (n + 1) / 2) / (n * m)


def _triple(text: str, syntax: SpanSyntax) -> tuple[str, ...]:
    body = text[len(syntax.opener) : -len(syntax.closer)]
    return tuple(part.strip() for part in body.split(syntax.separator.strip()))


def span_rows(record: dict, facts: dict, syntax: SpanSyntax = DEFAULT_SYNTAX) -> list[dict]:
    """The fact spans of one replayed output, each with its pressure scores, whether
    the judge found it unsupported (None if not judged), and whether it repeats an
    earlier fact span of the reply."""
    verdicts = {tuple(d["span"]): d["supported"] for d in facts.get(record["id"], [])}
    rows, seen = [], set()
    for span in sorted(record["spans"], key=lambda s: s["start"]):
        if span["kind"] != "fact":
            continue
        tokens = range(span["start"], span["stop"])
        log_mass = [record["log_mass"][i] for i in tokens]
        triple = _triple(span["text"], syntax)
        supported = verdicts.get(triple)
        rows.append(
            {
                "id": record["id"],
                "triple": triple,
                "unsupported": None if supported is None else not supported,
                "repeat": triple in seen,
                "pressure": -sum(log_mass),
                "min_log_mass": min(log_mass),
                "mean_log_prob": sum(record["log_prob"][i] for i in tokens) / len(tokens),
                "mean_entropy": sum(record["entropy"][i] for i in tokens) / len(tokens),
                "top_rejected": sum(1 - record["top_allowed"][i] for i in tokens) / len(tokens)
                if "top_allowed" in record
                else None,
            }
        )
        seen.add(triple)
    return rows


def _separation(rows: list[dict], label: str, scores: dict) -> dict:
    kept = [r for r in rows if r[label] is not None]
    result = {"rows": len(kept), "positives": sum(bool(r[label]) for r in kept)}
    for name, score in scores.items():
        value = auroc(
            [score(r) for r in kept if r[label]], [score(r) for r in kept if not r[label]]
        )
        result[name] = None if value is None else round(value, 4)
    return result


def release_triggers(rows: list[dict], thresholds=(0.01, 0.05, 0.1, 0.2, 0.3, 0.5)) -> dict:
    """For each release rule, how many judged fact spans it would have touched (some
    step's allowed mass below the threshold, or the model's top token disallowed),
    and the unsupported share among those spans and among the others."""
    import math

    judged = [r for r in rows if r["unsupported"] is not None]
    rules = {f"mass<{t:g}": (lambda r, t=t: r["min_log_mass"] < math.log(t)) for t in thresholds}
    if judged and all(r.get("top_rejected") is not None for r in judged):
        rules["top"] = lambda r: r["top_rejected"] > 0
    result = {}
    for name, rule in rules.items():
        hit = [r for r in judged if rule(r)]
        rest = [r for r in judged if not rule(r)]
        result[name] = {
            "spans": len(judged),
            "triggered": len(hit),
            "unsupported_triggered": sum(r["unsupported"] for r in hit) / len(hit) if hit else None,
            "unsupported_other": sum(r["unsupported"] for r in rest) / len(rest) if rest else None,
        }
    return result


SPAN_SCORES = {
    "pressure": lambda r: r["pressure"],
    "min_log_mass": lambda r: -r["min_log_mass"],
    "mean_log_prob": lambda r: -r["mean_log_prob"],
    "mean_entropy": lambda r: r["mean_entropy"],
    "top_rejected": lambda r: r["top_rejected"] or 0.0,
}
"""Higher means more suspect: the rejected mass, then the model's own confidence."""


def analyze(records: list[dict], facts: dict, judged: dict) -> dict:
    """AUROC of each score for unsupported fact spans, repeated fact spans, and wrong
    answers (a reply scored by its most pressed fact span, or its whole length)."""
    spans_all = [row for record in records for row in span_rows(record, facts)]
    answers = []
    for record in records:
        rows = span_rows(record, facts)
        decision = judged.get(record["id"])
        if decision is None or not record.get("log_mass"):
            continue
        n = len(record["log_mass"])
        answers.append(
            {
                "wrong": not decision["correct"],
                "max_pressure": max((r["pressure"] for r in rows), default=0.0),
                "total_pressure": -sum(record["log_mass"]),
                "mean_log_prob": sum(record["log_prob"]) / n,
                "mean_entropy": sum(record["entropy"]) / n,
            }
        )
    answer_scores = {
        "max_pressure": lambda r: r["max_pressure"],
        "total_pressure": lambda r: r["total_pressure"],
        "mean_log_prob": lambda r: -r["mean_log_prob"],
        "mean_entropy": lambda r: r["mean_entropy"],
    }
    return {
        "unsupported_spans": _separation(spans_all, "unsupported", SPAN_SCORES),
        "repeated_spans": _separation(spans_all, "repeat", SPAN_SCORES),
        "wrong_answers": _separation(answers, "wrong", answer_scores),
        "release_triggers": release_triggers(spans_all),
    }


def _prompt_ids(tokenizer, messages: list[dict]) -> list[int]:
    """The prompt as vLLM's chat renders it: the template's text, tokenized as is."""
    text = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, enable_thinking=False, tokenize=False
    )
    return tokenizer(text, add_special_tokens=False)["input_ids"]


def dump(model_dir: str, requests: dict, outputs: list[dict], index_dir, tokenizer_json):
    """Replay each output through the model and yield its per-token scores and spans."""
    import torch
    from transformers import AutoTokenizer

    from xgraph.engine import Engine
    from xgraph.hfload import load_model
    from xgraph.index import load_kg_index
    from xgraph.tokenizer import ServedTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = load_model(model_dir)
    index = load_kg_index(index_dir)
    served = ServedTokenizer.from_file(tokenizer_json)
    engine = Engine(index, served, index.span_syntax, max_sequences=1)
    for output in outputs:
        reply = output["token_ids"]
        prompt = _prompt_ids(tokenizer, requests[output["id"]]["messages"])
        record = {
            "id": output["id"],
            "prompt_tokens": len(prompt),
            "prompt_tokens_generated": output.get("prompt_tokens"),
            "finish_reason": output.get("finish_reason"),
            "spans": spans(served, reply, index.span_syntax),
        }
        if reply:
            ids = torch.tensor([prompt + reply], device="cuda")
            with torch.no_grad():
                logits = model(input_ids=ids).logits[0, len(prompt) - 1 : -1]
                scores = step_scores(engine, logits[:, : served.vocab_size], reply)
            for name, values in scores.items():
                if values.dtype == torch.bool:
                    record[name] = [int(v) for v in values.cpu()]
                else:
                    record[name] = [round(float(v), 4) for v in values.cpu()]
        yield record


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="python -m xgraph.pressure", description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("dump", help="replay outputs and record the rejected mass")
    run.add_argument("--model", required=True, help="bf16 model directory")
    run.add_argument("--requests", type=Path, required=True, help="chat requests (JSONL)")
    run.add_argument("--outputs", type=Path, required=True, help="constrained outputs (JSONL)")
    run.add_argument("--index", type=Path, required=True, help="the KG index they ran with")
    run.add_argument("--tokenizer", type=Path, required=True, help="its tokenizer.json")
    run.add_argument("--out", type=Path, required=True, help="per-token scores (JSONL)")
    report = commands.add_parser("analyze", help="AUROC of the pressure against the judge")
    report.add_argument("--pressure", type=Path, required=True, help="dump output (JSONL)")
    report.add_argument("--facts", type=Path, required=True, help="python -m xgraph.gate wwq-facts")
    report.add_argument(
        "--judged", type=Path, required=True, help="python -m xgraph.gate wwq-judged"
    )
    report.add_argument("--out", type=Path, required=True, help="result (JSON)")
    args = parser.parse_args(argv)
    if args.command == "analyze":
        from xgraph.provenance import provenance

        records = [json.loads(line) for line in args.pressure.read_text().splitlines() if line]
        facts = json.loads(args.facts.read_text())["decisions"]
        judged = json.loads(args.judged.read_text())["decisions"]
        result = {"artifact_kind": "xgraph.pressure_analysis", "version": 1}
        result.update(analyze(records, facts, judged))
        result["provenance"] = provenance([args.pressure, args.facts, args.judged])
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=1) + "\n")
        print(json.dumps({k: v for k, v in result.items() if k != "provenance"}, indent=1))
        return
    requests = {r["id"]: r for r in map(json.loads, args.requests.read_text().splitlines()) if r}
    outputs = [json.loads(line) for line in args.outputs.read_text().splitlines() if line]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as out:
        for n, record in enumerate(
            dump(args.model, requests, outputs, args.index, args.tokenizer), 1
        ):
            out.write(json.dumps(record) + "\n")
            if n % 100 == 0:
                print(f"{n}/{len(outputs)}", flush=True)


if __name__ == "__main__":
    main()
