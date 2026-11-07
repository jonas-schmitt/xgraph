"""Data for method pilot 2 (docs/research-plan.md): hidden states at span decisions.

Retrieval inside the mask scores the KG-valid candidates of a span segment against
the model's hidden state. Its decisive offline test needs, for fact spans the model
writes, the points where a relation or an object label begins, the candidates the KG
allows there, the label actually written, and the model's hidden state and logits at
the token before the label. This module finds those points in text and maps them to
tokens; the forward pass runs where the model is (the training container).
"""

from __future__ import annotations

import re

from xgraph.syntax import SpanSyntax

_S = SpanSyntax()
_FACT = re.compile(re.escape(_S.opener) + "(.*?)" + re.escape(_S.closer), re.S)


def segment_points(text: str) -> list[dict]:
    """For each fact span with three labels and no escape: the relation and the object
    points, each with the character index just after its separator, the subject, the
    relation, and the label written there (gold)."""
    points = []
    for match in _FACT.finditer(text):
        body = match.group(1)
        if _S.escape.strip() in body:
            continue
        parts = body.split(_S.separator)
        if len(parts) != 3:
            continue
        subject, relation, obj = (part.strip() for part in parts)
        start = match.start(1)
        first = start + body.index(_S.separator) + len(_S.separator)
        second = first + body[first - start :].index(_S.separator) + len(_S.separator)
        points.append(
            {
                "kind": "relation",
                "char": first,
                "subject": subject,
                "relation": None,
                "gold": relation,
            }
        )
        points.append(
            {
                "kind": "object",
                "char": second,
                "subject": subject,
                "relation": relation,
                "gold": obj,
            }
        )
    return points


def token_before(offsets: list[tuple[int, int]], char: int) -> int:
    """The index of the last token that ends at or before the character index."""
    best = 0
    for i, (_, end) in enumerate(offsets):
        if end <= char:
            best = i
        else:
            break
    return best


def candidates(store, point: dict) -> tuple[dict[int, str], set[int]]:
    """The KG-valid candidates of a point (id to label) and the gold ids among them."""
    from xgraph.score import normalize

    items = store.items_named(point["subject"])
    if point["kind"] == "relation":
        found = store.relations_of(items)
        gold = set(store.relation_named(point["gold"])) & set(found)
    else:
        found = {}
        for prop in store.relation_named(point["relation"]):
            found.update(store.objects_of(items, prop))
        gold = {i for i, label in found.items() if normalize(label) == normalize(point["gold"])}
    return found, gold


def dump(model_dir: str, examples: list[dict], store, max_candidates: int = 256) -> dict:
    """Run the model over the examples and collect, per decision point with at least two
    candidates and a gold one among them: the hidden state at the token before the
    label, the candidates, the gold ids, and the model's log-probability of each
    candidate's first label token. Candidate vectors are the mean input embeddings of
    their label tokens, stored once per label."""
    import torch
    from transformers import AutoTokenizer

    from xgraph.hfload import load_model

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = load_model(model_dir)
    embeddings = model.get_input_embeddings().weight
    label_vectors: dict[str, torch.Tensor] = {}
    records = []
    for example in examples:
        text = tokenizer.apply_chat_template(
            example["messages"], tokenize=False, enable_thinking=False
        )
        reply = example["messages"][-1]["content"]
        offset = text.rindex(reply)
        points = []
        for point in segment_points(reply):
            found, gold = candidates(store, point)
            if len(found) < 2 or not gold or len(found) > max_candidates:
                continue
            points.append((point, found, gold))
        if not points:
            continue
        encoded = tokenizer(text, return_offsets_mapping=True, add_special_tokens=False)
        ids = torch.tensor([encoded["input_ids"]], device="cuda")
        with torch.no_grad():
            out = model(input_ids=ids, output_hidden_states=True)
        hidden, logits = out.hidden_states[-1][0], out.logits[0].float().log_softmax(-1)
        for point, found, gold in points:
            at = token_before(encoded["offset_mapping"], offset + point["char"])
            labels = [found[i] for i in sorted(found)]
            first = []
            for label in labels:
                tokens = tokenizer.encode(_S.label_prefix + label, add_special_tokens=False)
                first.append(float(logits[at, tokens[0]]))
                if label not in label_vectors:
                    label_vectors[label] = embeddings[tokens].float().mean(0).half().cpu()
            records.append(
                {
                    "example": example["id"],
                    "kind": point["kind"],
                    "hidden": hidden[at].float().half().cpu(),
                    "candidates": sorted(found),
                    "labels": labels,
                    "gold": sorted(gold),
                    "lm_first_token": first,
                }
            )
    return {"points": records, "label_vectors": label_vectors}


def main(argv: list[str] | None = None) -> None:
    import argparse
    import json
    from pathlib import Path

    import torch

    from xgraph.toolkg import KGStore

    parser = argparse.ArgumentParser(prog="python -m xgraph.hidden", description=__doc__)
    parser.add_argument("--model", required=True, help="bf16 model directory")
    parser.add_argument("--data", type=Path, required=True, help="chat examples (JSONL)")
    parser.add_argument("--db", type=Path, required=True, help="KG store (xgraph.toolkg)")
    parser.add_argument("--out", type=Path, required=True, help="points (torch file)")
    args = parser.parse_args(argv)
    examples = [json.loads(line) for line in args.data.read_text().splitlines() if line]
    result = dump(args.model, examples, KGStore(args.db))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(result, args.out)
    kinds = [p["kind"] for p in result["points"]]
    print(json.dumps({k: kinds.count(k) for k in sorted(set(kinds))}))


if __name__ == "__main__":
    main()
