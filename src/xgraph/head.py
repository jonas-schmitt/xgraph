"""Method pilot 2's offline test: does a learned head rank KG candidates better than the
model's own next-token scores?

Points come from `python -m xgraph.hidden` (a hidden state at a span decision, the
KG-valid candidates there, the gold ones, and the model's log-probability of each
candidate's first label token). The head scores a candidate as (A h) . (B c), with h
the hidden state and c the mean input embedding of the candidate's label tokens, both
projected to `width` dimensions, trained with cross-entropy over each point's
candidates. Reported on held-out points: top-1 accuracy of the model's first-token
scores, of the head, and of their sum after standardizing each over the candidates.

    python -m xgraph.head --train target/pilot2/train-points.pt \\
        --test target/pilot2/dev-points.pt --out target/pilot2/head.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def gold_positions(point: dict) -> list[int]:
    """The positions among the point's candidates of its gold ids (both KG ids)."""
    return [point["candidates"].index(g) for g in point["gold"]]


def _tensors(point: dict, vectors: dict):
    import torch

    hidden = point["hidden"].float()
    cands = torch.stack([vectors[label].float() for label in point["labels"]])
    return hidden, cands


def train_head(
    points: list[dict],
    vectors: dict,
    dim: int,
    width: int = 256,
    epochs: int = 10,
    lr: float = 1e-3,
    seed: int = 0,
):
    import torch

    torch.manual_seed(seed)
    head = torch.nn.ModuleDict(
        {
            "a": torch.nn.Linear(dim, width, bias=False),
            "b": torch.nn.Linear(dim, width, bias=False),
        }
    )
    optimizer = torch.optim.Adam(head.parameters(), lr=lr)
    for _ in range(epochs):
        order = torch.randperm(len(points)).tolist()
        for i in order:
            hidden, cands = _tensors(points[i], vectors)
            scores = head["b"](cands) @ head["a"](hidden)
            target = torch.zeros(len(cands))
            gold = gold_positions(points[i])
            target[gold] = 1.0 / len(gold)
            loss = -(target * scores.log_softmax(0)).sum()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    return head


def _standard(values):
    return (values - values.mean()) / (values.std() + 1e-6)


def evaluate(head, points: list[dict], vectors: dict) -> dict:
    import torch

    hits = {"lm": 0, "head": 0, "combined": 0}
    by_kind: dict[str, dict[str, int]] = {}
    with torch.no_grad():
        for point in points:
            hidden, cands = _tensors(point, vectors)
            lm = torch.tensor(point["lm_first_token"])
            learned = head["b"](cands) @ head["a"](hidden)
            for name, scores in (
                ("lm", lm),
                ("head", learned),
                ("combined", _standard(lm) + _standard(learned)),
            ):
                hit = int(int(scores.argmax()) in gold_positions(point))
                hits[name] += hit
                kind = by_kind.setdefault(
                    point["kind"], {"points": 0, "lm": 0, "head": 0, "combined": 0}
                )
                kind[name] += hit
            by_kind[point["kind"]]["points"] += 1
    n = max(len(points), 1)
    return {
        "points": len(points),
        "lm_accuracy": hits["lm"] / n,
        "head_accuracy": hits["head"] / n,
        "combined_accuracy": hits["combined"] / n,
        "by_kind": by_kind,
    }


def main(argv: list[str] | None = None) -> None:
    import torch

    parser = argparse.ArgumentParser(prog="python -m xgraph.head", description=__doc__)
    parser.add_argument("--train", type=Path, required=True, help="points (xgraph.hidden)")
    parser.add_argument("--test", type=Path, required=True, help="held-out points")
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, required=True, help="result (JSON)")
    args = parser.parse_args(argv)
    train, test = torch.load(args.train), torch.load(args.test)
    vectors = {**train["label_vectors"], **test["label_vectors"]}
    dim = len(next(iter(vectors.values())))
    head = train_head(train["points"], vectors, dim, args.width, args.epochs, seed=args.seed)
    result = {
        "settings": vars(args)
        | {"train": str(args.train), "test": str(args.test), "out": str(args.out)}
    }
    result.update(evaluate(head, test["points"], vectors))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=1, default=str) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "settings"}, indent=1))


if __name__ == "__main__":
    main()
