"""Scoring the quality gate (`docs/quality-gate.md`).

- Label matching: a span names an entity when its text equals one of the entity's
  English labels or aliases after Unicode NFC normalization and case folding; a
  label several entities share counts as correct if any of them is gold.
- WWQ answer accuracy: the share of scored questions whose final entity span
  (`Answer: ⟦ label⟧`, see xgraph.prompts) names a gold entity. Questions without a
  gold entity are excluded and counted.
- Spans in text, by kind, for the judge and the reported span counts.
- The paired bootstrap the pass rule uses for confidence intervals. Given clusters (a
  label per pair), it resamples whole clusters with replacement instead of pairs, both
  arms together, for pairs that are not independent (questions sharing a relation
  sequence): a resample's mean is its pairs' sum over its pairs' count.
- The arm by stratum interaction: the paired difference on one stratum minus that on
  another, with the same (optionally clustered) bootstrap over both strata together.
"""

from __future__ import annotations

import json
import re
import unicodedata
from collections.abc import Callable, Hashable, Iterable
from pathlib import Path

import numpy as np

from xgraph.prompts import final_answer
from xgraph.syntax import SpanSyntax

_SYNTAX = SpanSyntax()


def normalize(text: str) -> str:
    return unicodedata.normalize("NFC", text).casefold().strip()


def spans(text: str, syntax: SpanSyntax = _SYNTAX) -> list[tuple[str, tuple[str, ...]]]:
    """The closed spans of a text, in order, as (kind, segment texts).

    Kind is "escaped" when the span holds the escape marker, "fact" when it has
    separators, and "entity" otherwise. Segment texts are stripped of the label prefix.
    """
    from xgraph.audit import semantic_text

    text = semantic_text(text, syntax)
    found = []
    escape = syntax.escape.strip()
    pattern = re.escape(syntax.opener) + "(.*?)" + re.escape(syntax.closer)
    for body in re.findall(pattern, text, re.DOTALL):
        if escape in body:
            found.append(("escaped", (body.strip(),)))
        elif syntax.separator.strip() in body:
            from xgraph.observations import path_segments

            parts = tuple(path_segments(body, syntax.separator, syntax.opener, syntax.escape))
            found.append(("fact" if len(parts) > 1 else "entity", parts))
        else:
            found.append(("entity", (body.strip(),)))
    return found


def replay_failure(record: dict) -> bool:
    """Declined or unconfirmed masking is never a constrained success."""
    status = record.get("replay_status")
    return (
        (
            bool(record.get("declined"))
            and not (
                record.get("arm") == "agent" and record.get("finish_reason") == "context_limit"
            )
        )
        or status not in (None, "not_requested", "replayed")
        or (bool(record.get("prefill")) and status != "replayed")
    )


def record_text(record: dict) -> str:
    """Apply replay admission failures before legacy text-only answer matching."""
    text = record.get("text")
    return text if isinstance(text, str) and not replay_failure(record) else ""


def record_failure(record: dict, *, require_text: bool = True) -> bool:
    """The common calibration/registered analysis failure policy, before matching."""
    text = record.get("text")
    return (
        (require_text and not isinstance(text, str))
        or replay_failure(record)
        or any(record.get(k) for k in ("declined", "truncated", "escaped", "malformed", "error"))
        or record.get("finish_reason") in {"length", "max_tokens", "error", "abort", "declined"}
        or (isinstance(text, str) and any(kind == "escaped" for kind, _ in spans(text)))
    )


def record_outputs(records: Iterable[dict], qids: Iterable[str]) -> dict[str, str]:
    """Failure-filtered text for every declared question; missing records abstain."""
    rows = {}
    for record in records:
        qid = record["id"]
        if qid in rows:
            raise ValueError(f"duplicate record id {qid!r}")
        rows[qid] = record
    return {q: "" if record_failure(rows.get(q, {})) else rows[q]["text"] for q in qids}


def load_labels(paths: Iterable[str | Path], wanted: set[str]) -> dict[str, set[str]]:
    """Normalized labels and aliases of the wanted items, from label JSON lines."""
    labels: dict[str, set[str]] = {}
    for path in paths:
        with Path(path).open(encoding="utf-8") as lines:
            for line in lines:
                item, label, _ = json.loads(line)
                if item in wanted:
                    labels.setdefault(item, set()).add(normalize(label))
    return labels


def names_gold(answer: str | None, gold: Iterable[str], labels: dict[str, set[str]]) -> bool:
    if answer is None:
        return False
    text = normalize(answer)
    return any(text in labels.get(item, ()) for item in gold)


def wwq_accuracy(
    outputs: dict[str, str],
    gold: dict,
    labels: dict[str, set[str]],
    syntax: SpanSyntax | None = _SYNTAX,
) -> dict:
    """Per-question correctness of the final answer, and the excluded questions."""
    correct: dict[str, bool] = {}
    excluded: dict[str, str] = {}
    for question, failure in gold["failed"].items():
        excluded[question] = f"gold query failed: {failure}"
    for question, answers in gold["answers"].items():
        if not answers["entities"]:
            excluded[question] = "no gold entity"
            continue
        text = outputs[question]
        correct[question] = names_gold(final_answer(text, syntax), answers["entities"], labels)
    return {"correct": correct, "excluded": excluded}


def paired_bootstrap(
    treated: Iterable[float],
    control: Iterable[float],
    *,
    resamples: int = 10_000,
    seed: int = 0,
    level: float = 0.95,
    clusters: Iterable[Hashable] | None = None,
) -> dict[str, float]:
    """Mean of treated minus control, with a percentile bootstrap interval over pairs, or
    over clusters of pairs when each pair's cluster is given."""
    a = np.asarray(list(treated), dtype=np.float64)
    b = np.asarray(list(control), dtype=np.float64)
    if a.shape != b.shape:
        raise ValueError(f"paired samples differ in length: {len(a)} and {len(b)}")
    diff = a - b
    if clusters is not None:
        columns = np.stack([diff, np.ones_like(diff)], axis=1)
        means = _resampled(columns, clusters, lambda s: s[:, 0] / s[:, 1], resamples, seed)
        low, high = _interval(means, level)
        return {"difference": float(diff.mean()), "low": low, "high": high}
    rng = np.random.default_rng(seed)
    means = np.empty(resamples)
    for start in range(0, resamples, 1000):  # bounded memory for large samples
        stop = min(start + 1000, resamples)
        picks = rng.integers(0, len(diff), size=(stop - start, len(diff)))
        means[start:stop] = diff[picks].mean(axis=1)
    tail = (1 - level) / 2
    return {
        "difference": float(diff.mean()),
        "low": float(np.quantile(means, tail)),
        "high": float(np.quantile(means, 1 - tail)),
    }


def paired_ratio_bootstrap(
    treated: Iterable[tuple[float, float]],
    control: Iterable[tuple[float, float]],
    *,
    resamples: int = 10_000,
    seed: int = 0,
    level: float = 0.95,
    clusters: Iterable[Hashable] | None = None,
) -> dict[str, float]:
    """Pooled share sum(numerator) / sum(denominator) per arm, and its paired difference.

    Each item is one question's (numerator, denominator) per arm, such as unsupported
    fact spans and fact spans; questions are resampled with both arms together, or whole
    clusters of them when each item's cluster is given.
    """
    a = np.asarray(list(treated), dtype=np.float64).reshape(-1, 2)
    b = np.asarray(list(control), dtype=np.float64).reshape(-1, 2)
    if a.shape != b.shape:
        raise ValueError(f"paired samples differ in length: {len(a)} and {len(b)}")
    share_a, share_b = a[:, 0].sum() / a[:, 1].sum(), b[:, 0].sum() / b[:, 1].sum()
    if clusters is not None:
        diffs = _resampled(
            np.concatenate([a, b], axis=1),
            clusters,
            lambda s: s[:, 0] / s[:, 1] - s[:, 2] / s[:, 3],
            resamples,
            seed,
        )
        low, high = _interval(diffs, level)
        return {
            "treated": float(share_a),
            "control": float(share_b),
            "difference": float(share_a - share_b),
            "low": low,
            "high": high,
        }
    rng = np.random.default_rng(seed)
    diffs = np.empty(resamples)
    for start in range(0, resamples, 1000):
        stop = min(start + 1000, resamples)
        picks = rng.integers(0, len(a), size=(stop - start, len(a)))
        sa, sb = a[picks].sum(axis=1), b[picks].sum(axis=1)
        with np.errstate(invalid="ignore", divide="ignore"):
            diffs[start:stop] = sa[:, 0] / sa[:, 1] - sb[:, 0] / sb[:, 1]
    tail = (1 - level) / 2
    return {
        "treated": float(share_a),
        "control": float(share_b),
        "difference": float(share_a - share_b),
        "low": float(np.nanquantile(diffs, tail)),
        "high": float(np.nanquantile(diffs, 1 - tail)),
    }


def interaction_bootstrap(
    treated: Iterable[tuple[float, float]],
    control: Iterable[tuple[float, float]],
    first: Iterable[bool],
    second: Iterable[bool],
    *,
    resamples: int = 10_000,
    seed: int = 0,
    level: float = 0.95,
    clusters: Iterable[Hashable] | None = None,
) -> dict:
    """The arm by stratum interaction: the paired difference of pooled shares (treated
    minus control, as paired_ratio_bootstrap) on the first stratum minus that on the
    second, with a percentile bootstrap interval. Each item is one (numerator,
    denominator) per arm, a mean's item being (value, 1); first and second say which
    stratum it is in, and items in neither are left out. Items (or, given each item's
    cluster, whole clusters) are drawn from both strata together, both arms together."""
    a = np.asarray(list(treated), dtype=np.float64).reshape(-1, 2)
    b = np.asarray(list(control), dtype=np.float64).reshape(-1, 2)
    one = np.asarray(list(first), dtype=bool)
    two = np.asarray(list(second), dtype=bool)
    if not a.shape == b.shape or not len(a) == len(one) == len(two):
        raise ValueError(
            f"paired samples and strata differ in length: {len(a)}, {len(b)}, "
            f"{len(one)} and {len(two)}"
        )
    if not one.any() or not two.any():
        raise ValueError(f"a stratum has no item: {one.sum()} and {two.sum()} items")
    kept = one | two
    labels = None if clusters is None else list(clusters)
    if labels is not None:
        if len(labels) != len(a):
            raise ValueError(f"{len(labels)} clusters given for {len(a)} paired items")
        labels = [label for label, keep in zip(labels, kept, strict=True) if keep]
    a, b, one, two = a[kept], b[kept], one[kept], two[kept]
    columns = np.concatenate(
        [a * one[:, None], b * one[:, None], a * two[:, None], b * two[:, None]], axis=1
    )

    def statistic(s: np.ndarray) -> np.ndarray:
        return (s[:, 0] / s[:, 1] - s[:, 2] / s[:, 3]) - (s[:, 4] / s[:, 5] - s[:, 6] / s[:, 7])

    sums = columns.sum(axis=0)

    def stratum(s: np.ndarray) -> dict[str, float]:
        share_a, share_b = s[0] / s[1], s[2] / s[3]
        return {
            "treated": float(share_a),
            "control": float(share_b),
            "difference": float(share_a - share_b),
        }

    values = _resampled(columns, labels, statistic, resamples, seed)
    low, high = _interval(values, level)
    return {
        "first": stratum(sums[:4]),
        "second": stratum(sums[4:]),
        "difference": float(statistic(sums[None, :])[0]),
        "low": low,
        "high": high,
    }


def _resampled(
    columns: np.ndarray,
    clusters: Iterable[Hashable] | None,
    statistic: Callable[[np.ndarray], np.ndarray],
    resamples: int,
    seed: int,
) -> np.ndarray:
    """The statistic of each bootstrap resample. columns holds one row of sums per item;
    a resample draws as many clusters as there are, with replacement (each item its own
    cluster when clusters is None), and the statistic maps each resample's column sums
    (resamples, columns) to its values (nan where a share has no denominator)."""
    if clusters is None:
        groups = columns
    else:
        labels = list(clusters)
        if len(labels) != len(columns):
            raise ValueError(f"{len(labels)} clusters given for {len(columns)} paired items")
        index: dict[Hashable, int] = {}
        codes = np.array([index.setdefault(label, len(index)) for label in labels], dtype=np.int64)
        groups = np.zeros((len(index), columns.shape[1]))
        np.add.at(groups, codes, columns)
    rng = np.random.default_rng(seed)
    values = np.empty(resamples)
    for start in range(0, resamples, 1000):  # bounded memory for large samples
        stop = min(start + 1000, resamples)
        picks = rng.integers(0, len(groups), size=(stop - start, len(groups)))
        sums = np.stack([column[picks].sum(axis=1) for column in groups.T], axis=1)
        with np.errstate(invalid="ignore", divide="ignore"):
            values[start:stop] = statistic(sums)
    return values


def _interval(values: np.ndarray, level: float) -> tuple[float, float]:
    """The percentile interval of bootstrap values, ignoring undefined (nan) ones."""
    tail = (1 - level) / 2
    return float(np.nanquantile(values, tail)), float(np.nanquantile(values, 1 - tail))
