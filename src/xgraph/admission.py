"""The admission contrast: does the model escape an answer because it contradicts it?

Four path-mode runs ask the same questions: on the edited and the unedited KG, each with
escapes on and with escapes off (forced: the answer after a path must name an entity it
reached). Taking the questions on which all four write the same path, with no escape
inside it, the written tokens are the same and only the entity the KG reaches can
differ; the forced runs' answers tell whether it does. The contrast is the share of
answers escaped with escapes on, on the edited and the unedited KG, where the reached
entity differs and, as a control, where it is the same; a missing answer (no answer
line) is counted apart from an escaped one, and the questions whose reached entity
differs are also split by the path's last relation (by_relation).

    python -m xgraph.admission --edited target/mqr-cf-3k/paths2-anchor.jsonl \\
        --unedited target/mqr-cf-3k/paths2-anchor-unedited.jsonl \\
        --edited-forced target/mqr-cf-3k/paths2-anchor-noescape.jsonl \\
        --unedited-forced target/mqr-cf-3k/paths2-anchor-noescape-unedited.jsonl \\
        --out target/mqr-cf-3k/admission.json
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from xgraph.score import record_text

ESCAPE = "⁇"
_PATH = re.compile(r"⟦[^⟧]*·[^⟧]*⟧")
_ANSWER = re.compile(r"^Answer:\s*⟦\s*(.*?)⟧", re.MULTILINE)


def _path(text: str) -> str | None:
    match = _PATH.search(text)
    return match.group(0) if match else None


def _answer(text: str) -> str | None:
    found = _ANSWER.findall(text)
    return found[-1] if found else None


def _last_relation(path: str) -> str:
    return path.strip("⟦⟧ ").rsplit("·", 1)[-1].strip()


def _shares(rows: list[tuple[str | None, str | None]]) -> dict:
    """Shares of escaped and of missing answers (escapes on), on the edited and the
    unedited KG, over rows of (edited answer, unedited answer); None is missing."""
    n = len(rows)

    def share(k: int, test) -> float:
        return sum(test(row[k]) for row in rows) / n if n else 0.0

    return {
        "questions": n,
        "edited_escaped": share(0, lambda a: a is not None and ESCAPE in a),
        "unedited_escaped": share(1, lambda a: a is not None and ESCAPE in a),
        "edited_missing": share(0, lambda a: a is None),
        "unedited_missing": share(1, lambda a: a is None),
    }


def admission_contrast(
    edited: dict[str, str],
    unedited: dict[str, str],
    edited_forced: dict[str, str],
    unedited_forced: dict[str, str],
) -> dict:
    """Escape shares on the edited and unedited KG (escapes on) over the questions whose
    four runs write one path without an escape, split by whether the forced runs reach
    different entities; a missing answer (no answer line) is counted apart from an
    escaped one. by_relation splits the questions whose reached entity differs by the
    path's last relation."""
    groups: dict[str, list] = {"reached_differs": [], "reached_same": []}
    relations: dict[str, list] = {}
    for qid, text in edited.items():
        runs = [
            text,
            unedited.get(qid, ""),
            edited_forced.get(qid, ""),
            unedited_forced.get(qid, ""),
        ]
        paths = {_path(t) for t in runs}
        if len(paths) != 1 or None in paths or ESCAPE in next(iter(paths)):
            continue
        forced_edited, forced_unedited = _answer(runs[2]), _answer(runs[3])
        if forced_edited is None or forced_unedited is None:
            continue
        row = (_answer(runs[0]), _answer(runs[1]))
        if forced_edited == forced_unedited:
            groups["reached_same"].append(row)
        else:
            groups["reached_differs"].append(row)
            relations.setdefault(_last_relation(next(iter(paths))), []).append(row)
    result = {name: _shares(rows) for name, rows in groups.items()}
    result["by_relation"] = {r: _shares(rows) for r, rows in sorted(relations.items())}
    return result


def main(argv: list[str] | None = None) -> None:
    from xgraph.provenance import provenance

    parser = argparse.ArgumentParser(prog="python -m xgraph.admission", description=__doc__)
    for name in ("edited", "unedited", "edited-forced", "unedited-forced"):
        parser.add_argument(f"--{name}", type=Path, required=True, help="path-mode outputs")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    paths = [args.edited, args.unedited, args.edited_forced, args.unedited_forced]

    def texts(path: Path) -> dict[str, str]:
        rows = (json.loads(line) for line in path.read_text().splitlines() if line)
        return {r["id"]: record_text(r) for r in rows}

    result = {"artifact_kind": "xgraph.admission", "version": 1}
    result.update(admission_contrast(*(texts(p) for p in paths)))
    result["provenance"] = provenance(paths)
    args.out.write_text(json.dumps(result, indent=1) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "provenance"}))


if __name__ == "__main__":
    main()
