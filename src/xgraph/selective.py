"""Gold-free selective verification of plan repair, including offline MINTQA replay.

Feature version 1 uses the actual verifier menu (which may be the whole candidate
pool). It never reads an endpoint's identity, answer, verifier reply, or gold.
Only an explicitly recorded path_logprob is a likelihood feature: whole-answer
likelihoods would also score answer delivery and are not interchangeable.
"""

from __future__ import annotations

import json
import math
import operator
from pathlib import Path

FEATURE_VERSION = 1
FEATURES = (
    "lexical_margin",
    "candidate_count",
    "shortlist_size",
    "plan_length",
    "original_frontier",
    "best_alternative_frontier",
    "best_alternative_length_delta",
    "alternative_frontier_min",
    "path_logprob",
)
OPERATORS = {">": operator.gt, ">=": operator.ge, "<": operator.lt, "<=": operator.le}


def features(record: dict, shown: list[int] | None = None, output: dict | None = None) -> dict:
    """Pre-verification scalars; unavailable margins/frontiers/logprobs are None.

    Tied lexical alternatives retain candidate order, as repair.ranked_choice does.
    A supplied output contributes only its explicit path_logprob scalar.
    """
    from xgraph.repair import shortlist_scores

    candidates = record["candidates"]
    shown = list(range(len(candidates))) if shown is None else shown
    if len(set(shown)) != len(shown) or any(k < 0 or k >= len(candidates) for k in shown):
        raise ValueError(f"question {record['id']!r} has an invalid verifier menu: {shown}")
    if shown and 0 not in shown:
        raise ValueError(f"question {record['id']!r} has no original in its verifier menu")
    scores = shortlist_scores(record)
    alternatives = [k for k in shown if k != 0]
    best = max(alternatives, key=lambda k: (scores[k], -k)) if alternatives else None
    original = candidates[0] if candidates and candidates[0]["kind"] == "original" else None
    other = candidates[best] if best is not None else None
    logprob = (output if output is not None else record).get("path_logprob")
    if logprob is not None and (
        not isinstance(logprob, (float, int)) or not math.isfinite(logprob)
    ):
        raise ValueError(f"question {record['id']!r} has invalid path_logprob {logprob!r}")
    return {
        "lexical_margin": scores[best] - scores[0] if best is not None and original else None,
        "candidate_count": len(candidates),
        "shortlist_size": len(shown),
        "plan_length": len(original["relations"]) if original else 0,
        "original_frontier": original["reached"] if original else None,
        "best_alternative_frontier": other["reached"] if other else None,
        "best_alternative_length_delta": (
            len(other["relations"]) - len(original["relations"]) if other and original else None
        ),
        "alternative_frontier_min": min(
            (candidates[k]["reached"] for k in alternatives), default=None
        ),
        "path_logprob": logprob,
    }


def validate_rule(rule: dict) -> dict:
    """A frozen rule is a constant, or at most two scalar thresholds, AND/OR joined."""
    if set(rule) == {"constant"} and isinstance(rule["constant"], bool):
        return rule
    if set(rule) != {"operator", "conditions"} or rule["operator"] not in ("and", "or"):
        raise ValueError(f"invalid selective rule: {rule!r}")
    if not isinstance(rule["conditions"], list) or not 1 <= len(rule["conditions"]) <= 2:
        raise ValueError("a selective rule needs one or two threshold conditions")
    for condition in rule["conditions"]:
        if (
            set(condition) != {"feature", "op", "threshold"}
            or condition["feature"] not in FEATURES
            or condition["op"] not in OPERATORS
            or not isinstance(condition["threshold"], (int, float))
            or not math.isfinite(condition["threshold"])
        ):
            raise ValueError(f"invalid selective threshold: {condition!r}")
    return rule


def triggered(values: dict, rule: dict) -> bool:
    """Apply a validated rule. A missing feature never satisfies its condition."""
    if "constant" in rule:
        return rule["constant"]
    matches = []
    for condition in rule["conditions"]:
        value = values.get(condition["feature"])
        matches.append(
            value is not None and OPERATORS[condition["op"]](value, condition["threshold"])
        )
    return all(matches) if rule["operator"] == "and" else any(matches)


def load_rule(path: Path, family: str) -> dict:
    """Load the all-development fit for a subsequent frozen confirmation."""
    frozen = json.loads(path.read_text(encoding="utf-8"))
    if frozen.get("version") != 1 or frozen.get("feature_version") != FEATURE_VERSION:
        raise ValueError(f"unsupported selective rule version in {path}")
    if family not in frozen.get("families", {}):
        raise ValueError(f"no selective rule for family {family!r} in {path}")
    return validate_rule(frozen["families"][family]["rule"])
