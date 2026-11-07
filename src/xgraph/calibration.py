"""The registered calibration panel and exact candidate-selection order.

Both agreement reports and nested assurance call this selector. Multiplicities
represent whole repeated calibration cases; population weights never select arms.
"""

from __future__ import annotations

import numpy as np


def case_of(qid: str) -> str:
    return qid.rsplit("-", 1)[0]


def select_arms(
    panel: Panel,
    arms: dict,
    candidates: list[str],
    costs: dict,
    *,
    case_multiplicities: np.ndarray | None = None,
) -> dict:
    """Exact panel counts, then measured total tokens; X matches P's delivery."""
    counts = (
        np.ones(len(panel.cases), dtype=int)
        if case_multiplicities is None
        else np.asarray(case_multiplicities)
    )
    if (
        counts.shape != (len(panel.cases),)
        or not np.isfinite(counts).all()
        or (counts < 0).any()
        or not np.equal(counts, np.floor(counts)).all()
        or not counts.any()
    ):
        raise ValueError("case multiplicities must be nonnegative integers with nonzero total")
    groups = {role: [] for role in ("P", "X", "A")}
    seen = set()
    for spec in candidates:
        kind, separator, name = spec.partition("=")
        role, _, delivery = kind.partition(":")
        if not separator or role not in groups or name not in arms or name in seen:
            raise ValueError(f"invalid or duplicate candidate {spec!r}")
        if delivery not in ({"reader"} if role == "A" else {"engine", "executor"}):
            raise ValueError(f"invalid candidate delivery {spec!r}")
        seen.add(name)
        right = arms[name]
        groups[role].append(
            {
                "arm": name,
                "delivery": delivery,
                "correct_questions": int(panel.by_case(right).sum(axis=1) @ counts),
                "all_three_cases": int(panel.by_case(right).all(axis=1) @ counts),
                "total_tokens": costs[name]["total_tokens"],
            }
        )
    selected = {}
    for role in ("P", "X", "A"):
        pool = groups[role]
        if role == "X":
            pool = [c for c in pool if c["delivery"] == selected["P"]["delivery"]]
        if not pool:
            raise ValueError(f"no {role} candidates with the required answer delivery")
        best = max((c["correct_questions"], c["all_three_cases"]) for c in pool)
        tied = [c for c in pool if (c["correct_questions"], c["all_three_cases"]) == best]
        if len(tied) > 1 and any(c["total_tokens"] is None for c in tied):
            raise ValueError(
                f"{role}: accuracy tie needs measured total-token cost for every tied candidate"
            )
        selected[role] = min(tied, key=lambda c: (c["total_tokens"] or 0, c["arm"]))
    return selected


class Panel:
    """Questions grouped by case and cases by relation sequence."""

    def __init__(self, qids: list[str], clusters: dict[str, str], weights: dict[str, float]):
        if not qids or len(set(qids)) != len(qids):
            raise ValueError("panel questions must be nonempty and have unique ids")
        self.qids = qids
        self.cases = list(dict.fromkeys(case_of(q) for q in qids))
        index = {c: i for i, c in enumerate(self.cases)}
        self.case_of_question = np.array([index[case_of(q)] for q in qids])
        self.sequences = sorted({clusters[c] for c in self.cases})
        seq_index = {s: i for i, s in enumerate(self.sequences)}
        self.sequence_of_case = np.array([seq_index[clusters[c]] for c in self.cases])
        self.case_weight = np.array([weights[c] for c in self.cases])
        if not np.isfinite(self.case_weight).all() or (self.case_weight <= 0).any():
            raise ValueError("panel weights must be finite and positive")
        sizes = np.bincount(self.case_of_question, minlength=len(self.cases))
        if len(set(sizes.tolist())) != 1:
            raise ValueError(f"cases ask different numbers of questions: {sorted(set(sizes))}")
        self.per_case = int(sizes[0])
        if self.per_case != 3:
            raise ValueError("the calibration panel needs exactly three questions per case")

    def by_case(self, values: np.ndarray) -> np.ndarray:
        """Questions' values as a (cases, questions per case) array, in question order."""
        order = np.argsort(self.case_of_question, kind="stable")
        return values[order].reshape(len(self.cases), self.per_case)
