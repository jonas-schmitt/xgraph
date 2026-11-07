"""The retrieved-edits arm: MQuAKE questions with retrieved edits in the prompt.

The fair counterpart of the oracle arm (`requests-edits.jsonl`, each question with its
own case's edits): each question is asked, unconstrained and in one pass, with the k
edits a retriever ranks first from the edit bank of the question's interference-free
world (`python -m xgraph.mquake --worlds`), which holds the same edits as that world's
KG. The bank keys each edit by MeLLo's edit sentence (the rewrite's prompt filled with
its subject and new object, "Ellie Kemper is a citizen of Croatia"); the retriever is
GWalk's and MeLLo's, contriever-msmarco with mean pooling, and its query is the
question. The prompt lists the retrieved edits as labelled facts, as the oracle arm
does, so only the retrieval differs. With --every-edit, each question instead gets every
edit of its world, in one order (the long-context editing arm; no retriever); with
--own-edits, its own case's edits (the oracle). The prompt frames the edits as updates
that replace what the model remembers, leaving the rest to its own knowledge
(xgraph.rag.edit_messages); --plain-facts lists them as plain Wikidata facts instead,
which the model reads as the only facts to answer from.

Needs torch and transformers (the vLLM image holds both; `scripts/retrieve-edits.sh`
runs it there):

    python -m xgraph.edit_retrieval --data data/mquake/MQuAKE-Remastered-CF-3k.json \\
        --cases 1000 --seed 0 --per-case 3 --worlds target/mqr-cf-3k-worlds/worlds.json \\
        --k 4 --out target/mqr-cf-3k-worlds/requests-retrieved-edits-4.jsonl
"""

from __future__ import annotations

import argparse
import json
import random
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

CONTRIEVER = "facebook/contriever-msmarco"
CONTRIEVER_REVISION = "abe8c1493371369031bcb1e02acb754cf4e162fa"

Key = tuple[int, int, int]


def edit_bank(cases: list[dict]) -> dict[Key, dict]:
    """Every case's edits by (subject, property, object) number: MeLLo's edit sentence
    and the labelled fact, from the case's edited chain, or for an edit off that chain
    from its rewrite (subject and new object) and the property's label in any chain."""
    from xgraph.mquake import _number

    relation: dict[str, str] = {}
    for case in cases:
        for ids, names in (("triples", "triples_labeled"), ("new_triples", "new_triples_labeled")):
            for t, n in zip(case["orig"].get(ids, []), case["orig"].get(names, []), strict=True):
                relation.setdefault(t[1], n[1])
    bank: dict[Key, dict] = {}
    for case in cases:
        chain = [tuple(t) for t in case["orig"]["new_triples"]]
        labelled = case["orig"]["new_triples_labeled"]
        rewrites = case["requested_rewrite"]
        for edit, rewrite in zip(case["orig"]["edit_triples"], rewrites, strict=True):
            key: Key = (_number(edit[0]), _number(edit[1]), _number(edit[2]))
            if key in bank:
                continue
            new = rewrite["target_new"]["str"]
            sentence = f"{rewrite['prompt'].format(rewrite['subject'])} {new}"
            if tuple(edit) in chain:
                fact = tuple(labelled[chain.index(tuple(edit))])
            else:
                fact = (rewrite["subject"], relation[edit[1]], new)
            bank[key] = {"sentence": sentence, "fact": fact}
    return bank


def retrieved_requests(
    asked: list[dict],
    world_of: dict[str, int],
    world_edits: list[np.ndarray],
    bank: dict[Key, dict],
    rank: Callable[[str, list[str]], list[int]],
    k: int,
    per_case: int = 1,
    messages: Callable[[str, list[tuple[str, str, str]]], list[dict]] | None = None,
) -> list[dict]:
    """Each question with the k edits rank puts first among its world's edits.
    rank(question, sentences) returns the sentences' indices, best first; messages
    builds the prompt (default xgraph.rag.edit_messages: the edits as updates)."""
    from xgraph.rag import edit_messages

    messages = messages or edit_messages

    banks = [[bank[(int(s), int(p), int(o))] for s, p, o in edits] for edits in world_edits]
    requests = []
    for case in asked:
        world = banks[world_of[f"mquake-{case['case_id']}"]]
        sentences = [e["sentence"] for e in world]
        for j, utterance in enumerate(case["questions"][:per_case]):
            qid = f"mquake-{case['case_id']}" + (f"-{j}" if per_case > 1 else "")
            facts = [list(world[i]["fact"]) for i in rank(utterance, sentences)[:k]]
            prompt = messages(utterance, [tuple(f) for f in facts])
            requests.append({"id": qid, "messages": prompt, "facts": facts})
    return requests


def every_edit_requests(
    asked: list[dict],
    world_of: dict[str, int],
    world_edits: list[np.ndarray],
    bank: dict[Key, dict],
    per_case: int = 1,
    messages: Callable[[str, list[tuple[str, str, str]]], list[dict]] | None = None,
) -> list[dict]:
    """Each question with every edit of its world in the prompt, ordered by (subject,
    property, object) number, so a world's questions share the prompt up to the
    question and a prefix cache prefills the edits once (the long-context editing arm)."""
    ordered = [edits[np.lexsort(edits.T[::-1])] if len(edits) else edits for edits in world_edits]
    longest = max((len(edits) for edits in ordered), default=0)
    return retrieved_requests(
        asked, world_of, ordered, bank, lambda _, sentences: list(range(len(sentences))),
        longest, per_case, messages,
    )  # fmt: skip


def own_edit_requests(
    asked: list[dict],
    per_case: int = 1,
    messages: Callable[[str, list[tuple[str, str, str]]], list[dict]] | None = None,
) -> list[dict]:
    """The oracle: each question with its own case's edits (labelled from the edited
    chain, as xgraph.mquake's requests-edits.jsonl), framed as the other arms are."""
    from xgraph.rag import edit_messages

    messages = messages or edit_messages
    requests = []
    for case in asked:
        chain, labelled = case["orig"]["new_triples"], case["orig"]["new_triples_labeled"]
        edited = [tuple(t) for t in case["orig"]["edit_triples"]]
        facts = [list(labelled[i]) for i, t in enumerate(chain) if tuple(t) in edited]
        for j, utterance in enumerate(case["questions"][:per_case]):
            qid = f"mquake-{case['case_id']}" + (f"-{j}" if per_case > 1 else "")
            prompt = messages(utterance, [tuple(f) for f in facts])
            requests.append({"id": qid, "messages": prompt, "facts": facts})
    return requests


def contriever_ranker(name: str, revision: str) -> Callable[[str, list[str]], list[int]]:
    """GWalk's and MeLLo's retriever: contriever embeddings, mean pooled, by dot product;
    sentence embeddings are cached across calls."""
    import torch
    from transformers import AutoModel, AutoTokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModel.from_pretrained(name, revision=revision).to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained(name, revision=revision)
    cache: dict[str, Any] = {}

    def embed(texts: list[str]) -> Any:
        out = []
        for i in range(0, len(texts), 64):
            batch = tokenizer(texts[i : i + 64], padding=True, truncation=True, return_tensors="pt")
            batch = batch.to(device)
            with torch.no_grad():
                tokens = model(**batch)[0]
            mask = batch["attention_mask"][..., None].bool()
            out.append((tokens.masked_fill(~mask, 0.0).sum(1) / mask.sum(1)).cpu())
        return torch.vstack(out)

    def rank(question: str, sentences: list[str]) -> list[int]:
        missing = [s for s in dict.fromkeys(sentences) if s not in cache]
        if missing:
            cache.update(zip(missing, embed(missing), strict=True))
        matrix = torch.vstack([cache[s] for s in sentences])
        scores = matrix @ embed([question])[0]
        return [int(i) for i in torch.argsort(scores, descending=True, stable=True)]

    return rank


def main(argv: list[str] | None = None) -> None:
    from xgraph.mquake import worlds
    from xgraph.provenance import provenance

    parser = argparse.ArgumentParser(prog="python -m xgraph.edit_retrieval", description=__doc__)
    parser.add_argument("--data", type=Path, required=True, help="MQuAKE-Remastered-CF-3k.json")
    parser.add_argument("--cases", type=int, default=1000, help="cases asked (as xgraph.mquake)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--per-case", type=int, default=1, help="questions asked per case (1-3)")
    parser.add_argument("--worlds", type=Path, required=True, help="xgraph.mquake's worlds.json")
    parser.add_argument("--k", type=int, default=4, help="edits retrieved per question")
    parser.add_argument(
        "--every-edit", action="store_true", help="every edit of the world, no retrieval"
    )
    parser.add_argument(
        "--own-edits", action="store_true", help="the oracle: each case's own edits"
    )
    parser.add_argument(
        "--plain-facts",
        action="store_true",
        help="list the edits as plain Wikidata facts (xgraph.rag.rag_messages), not as updates",
    )
    parser.add_argument("--contriever", default=CONTRIEVER)
    parser.add_argument("--contriever-revision", default=CONTRIEVER_REVISION)
    parser.add_argument("--out", type=Path, required=True, help="requests (JSONL)")
    args = parser.parse_args(argv)
    cases = json.loads(args.data.read_text())
    asked = sorted(random.Random(args.seed).sample(cases, args.cases), key=lambda c: c["case_id"])
    found = worlds(asked, cases)
    world_of = {f"mquake-{c}": k for k, w in enumerate(found) for c in w["cases"]}
    if world_of != json.loads(args.worlds.read_text()):
        raise SystemExit(f"error: the worlds differ from {args.worlds}; check --cases and --seed")
    from xgraph.rag import edit_messages, rag_messages

    frame = rag_messages if args.plain_facts else edit_messages
    edits, bank = [w["edits"] for w in found], edit_bank(cases)
    if args.own_edits:
        requests = own_edit_requests(asked, args.per_case, frame)
    elif args.every_edit:
        requests = every_edit_requests(asked, world_of, edits, bank, args.per_case, frame)
    else:
        rank = contriever_ranker(args.contriever, args.contriever_revision)
        requests = retrieved_requests(
            asked, world_of, edits, bank, rank, args.k, args.per_case, frame
        )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as out:
        for request in requests:
            out.write(json.dumps(request, ensure_ascii=False) + "\n")
    manifest = {
        "artifact_kind": "xgraph.retrieved_edits",
        "version": 1,
        "requests": len(requests),
        "k": None if args.every_edit or args.own_edits else args.k,
        "framing": "plain facts" if args.plain_facts else "updates",
        "mode": "own edits" if args.own_edits else "every edit" if args.every_edit else "retrieved",
        "worlds": len(found),
        "bank_sizes": [len(w["edits"]) for w in found],
        "contriever": {"name": args.contriever, "revision": args.contriever_revision},
        "provenance": provenance([args.data, args.worlds]),
    }
    Path(f"{args.out}.manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")
    print(json.dumps({k: v for k, v in manifest.items() if k != "provenance"}))


if __name__ == "__main__":
    main()
