"""Streaming CPU evidence for the hypothesis-relative initial-case design."""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
from collections import Counter
from functools import lru_cache
from pathlib import Path

from xgraph.provenance import sha256_file

ID = re.compile(rb"mintqa-(?:ti|pop)-[0-9]+(?![0-9])")
TEXT = {".json", ".jsonl", ".md", ".txt", ".log", ".csv", ".tsv", ".yaml", ".yml", ".py", ".sh"}
FORBIDDEN = ("postq", "post-queue", "post_queue", "webqsp")


def exposure_inventory(worktrees, population, output):
    """Inventory every target file, projecting only IDs, never retaining outcomes.

    Unclassified ID-bearing artifacts are conservatively exposed. Only complete
    release inputs, unused prompts and prospective case-alias artifacts are
    metadata, and the owner's precise Qwen2.5 full-run exception is separate.
    Forbidden benchmark/serving subtrees are listed without opening their files.
    """
    from xgraph.case_alias import id_digest

    population = set(population)
    exposed, prior, counts = set(), set(), Counter()
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w") as manifest:
        for worktree in sorted(map(Path, worktrees)):
            target = worktree / "target"
            for directory, children, files in os.walk(target, followlinks=False):
                children.sort()
                for child in list(children):
                    path = Path(directory) / child
                    if any(s in child.lower() for s in FORBIDDEN) or child in (
                        "uv-cache",
                        "cpu-bin",
                        ".venv",
                        "node_modules",
                    ):
                        children.remove(child)
                        manifest.write(
                            json.dumps({"path": str(path), "classification": "excluded subtree"})
                            + "\n"
                        )
                        counts["excluded subtree"] += 1
                for name in sorted(files):
                    path = Path(directory) / name
                    relative = path.relative_to(target)
                    row = {"path": str(path)}
                    if path.is_symlink() or path.suffix not in TEXT:
                        row["classification"] = "symlink or nontext"
                    elif any(s in str(relative).lower() for s in FORBIDDEN):
                        row["classification"] = "forbidden path"
                    elif path == output:
                        row["classification"] = "current audit output"
                    else:
                        found, digest = set(), hashlib.sha256()
                        before = path.stat()
                        with path.open("rb") as stream:
                            tail = b""
                            while block := stream.read(1024 * 1024):
                                digest.update(block)
                                data = tail + block
                                found.update(
                                    m[0].decode() for m in ID.finditer(data) if m.end() < len(data)
                                )
                                tail = data[-64:]
                            found.update(m[0].decode() for m in ID.finditer(tail))
                        after = path.stat()
                        if (before.st_size, before.st_mtime_ns) != (
                            after.st_size,
                            after.st_mtime_ns,
                        ):
                            raise ValueError(f"exposure source changed during scan: {path}")
                        found &= population
                        row.update(
                            bytes=after.st_size,
                            sha256=digest.hexdigest(),
                            ids=sorted(found),
                            ids_sha256=id_digest(found),
                        )
                        key = str(relative)
                        if not found:
                            category = "no population IDs"
                        elif re.fullmatch(r"mintqa/qwen25-.*-all\.jsonl", key):
                            category = "disclosed Qwen2.5 different-model prior"
                            prior |= found
                        elif key.startswith(("case-alias/", "queue-case-alias/")) and not any(
                            p in relative.parts for p in ("WT", "WT+alias")
                        ):
                            category = "prospective case-alias preparation"
                        elif key in ("mintqa/questions.json", "mintqa/chains.json"):
                            category = "complete release metadata"
                        elif (
                            name.startswith("requests-")
                            and key[:7] == "mintqa/"
                            and "panel" not in name
                        ):
                            category = "complete release prompts"
                        else:
                            category = "exposed or conservative exclusion"
                            exposed |= found
                        row["classification"] = category
                    counts[row["classification"]] += 1
                    manifest.write(json.dumps(row, sort_keys=True) + "\n")
    return {
        "roots": [str(Path(w) / "target") for w in worktrees],
        "manifest": {"path": str(output), "sha256": sha256_file(output)},
        "classifications": dict(counts),
        "exposed_ids": sorted(exposed),
        "different_model_prior_ids": sorted(prior),
        "scope": (
            "every target file enumerated; textual IDs projected in streaming blocks; "
            "exclusions listed"
        ),
    }


def worktree_roots(repo):
    roots = {
        Path(line.removeprefix("worktree "))
        for line in subprocess.check_output(
            ["git", "-C", str(repo), "worktree", "list", "--porcelain"], text=True
        ).splitlines()
        if line.startswith("worktree ")
    }
    roots.update(p for p in (Path(repo) / ".claude/worktrees").iterdir() if (p / "target").is_dir())
    return sorted(roots)


def question_reachability(questions, baseline, alias, tokenizer, label_prefix=" "):
    """Exact, case-preserving question substrings at Unicode word boundaries.

    A witness is a token prefix with a nonempty alias row range and an empty
    base row range. It measures potential access, not actual model use or rescue.
    No answers, linked entities, gold chains or model outcomes are consulted.
    """
    strata, affected, witnesses = {}, [], {}

    @lru_cache(maxsize=100_000)
    def witness(surface):
        tokens = tokenizer.encode(label_prefix + surface)
        lo, hi = alias.prefix_range(tokens)
        if lo == hi:
            return None
        for end in range(1, len(tokens) + 1):
            prefix = tokens[:end]
            lo, hi = baseline.prefix_range(prefix)
            if lo == hi:
                return {"surface": surface, "first_alias_only_prefix": prefix}
        return None

    for q in sorted(questions, key=lambda q: q["id"]):
        group = strata.setdefault(q["complexity"], {"questions": 0, "affected": 0})
        group["questions"] += 1
        text = q["utterance"]
        words = list(re.finditer(r"\w+", text))
        found = None
        for start in range(len(words)):
            for end in range(start, len(words)):
                found = witness(text[words[start].start() : words[end].end()])
                if found:
                    break
            if found:
                break
        if found:
            group["affected"] += 1
            affected.append(q["id"])
            witnesses[q["id"]] = found
    for group in strata.values():
        group["fraction"] = group["affected"] / group["questions"]
    return {
        "questions": len(questions),
        "affected": len(affected),
        "fraction": len(affected) / len(questions) if questions else None,
        "affected_ids": affected,
        "by_stratum": strata,
        "witnesses": witnesses,
        "definition": (
            "case-preserving question word-boundary substring reaches alias-only token prefix"
        ),
        "limitation": (
            "potential reachability, not a measured policy flip; "
            "homonym/ranking spillovers remain unmodeled"
        ),
    }


def label_arrays(directory):
    """Mmap the label index without loading millions of unused owner strings."""
    import numpy as np

    from xgraph.index import LabelIndex

    arrays = [
        np.load(Path(directory) / f"labels.{name}.npy", mmap_mode="r")
        for name in ("tokens", "offsets", "owners")
    ]
    return LabelIndex(*arrays, (), "", ())


def wide_prefix_count(labels):
    """Count native wide-mask groups in bounded chunks, including split groups."""
    import numpy as np

    from xgraph.engine import MAX_WIDE_DEPTH, WIDE_ROWS

    count = 0
    for depth in range(1, MAX_WIDE_DEPTH + 1):
        previous, run = None, 0
        for lo in range(0, labels.num_rows, 65536):
            hi = min(labels.num_rows, lo + 65536)
            offsets = labels.offsets[lo:hi]
            lengths = labels.offsets[lo + 1 : hi + 1] - offsets
            prefix = np.full((hi - lo, depth), -1, dtype=np.int32)
            for column in range(depth):
                valid = lengths > column
                prefix[valid, column] = labels.tokens[offsets[valid] + column]
            changes = np.ones(hi - lo, dtype=bool)
            changes[1:] = (prefix[1:] != prefix[:-1]).any(axis=1)
            starts = np.flatnonzero(changes)
            sizes = np.diff(np.append(starts, hi - lo))
            keys = prefix[starts]
            if previous is not None:
                if tuple(keys[0]) == previous:
                    sizes[0] += run
                elif min(previous) >= 0 and run >= WIDE_ROWS:
                    count += 1
            count += int(np.count_nonzero((keys[:-1] >= 0).all(axis=1) & (sizes[:-1] >= WIDE_ROWS)))
            previous, run = tuple(keys[-1]), int(sizes[-1])
        if previous is not None and min(previous) >= 0 and run >= WIDE_ROWS:
            count += 1
    return count


def index_size_delta(baseline, alias, *, vocab_size):
    """Saved bytes and exact changing native-table bytes; no CUDA allocation.

    Native tables additionally hold one entity-row position and one cumulative
    unquoted-row count per added row. Owner offsets and relation tables retain
    their lengths. Wide-mask groups use the native implementation's thresholds.
    All remaining table shapes are unchanged for this registered WT recipe.
    """
    import numpy as np

    totals = {}
    for name, directory in (("WT", Path(baseline)), ("WT+alias", Path(alias))):
        arrays = {p.name: np.load(p, mmap_mode="r") for p in directory.glob("*.npy")}
        labels = label_arrays(directory)
        totals[name] = {
            "rows": labels.num_rows,
            "saved_bytes": sum(
                p.stat().st_size
                for p in directory.iterdir()
                if p.name == "kg-index.json" or p.suffix == ".npy"
            ),
            "array_payload_bytes": sum(a.nbytes for a in arrays.values()),
            "wide_prefix_groups": wide_prefix_count(labels),
            "arrays": {
                k: {"shape": a.shape, "dtype": str(a.dtype), "bytes": a.nbytes}
                for k, a in arrays.items()
            },
        }
    base, variant = totals["WT"], totals["WT+alias"]
    rows = variant["rows"] - base["rows"]
    payload = variant["array_payload_bytes"] - base["array_payload_bytes"]
    wide = variant["wide_prefix_groups"] - base["wide_prefix_groups"]
    return {
        "indexes": totals,
        "added_rows": rows,
        "saved_delta_bytes": variant["saved_bytes"] - base["saved_bytes"],
        "array_payload_delta_bytes": payload,
        "gpu_table_delta_bytes": payload + 8 * rows + wide * (8 + 4 * ((vocab_size + 31) // 32)),
        "definition": (
            "NPY payload delta + 8 bytes/added row + native wide-key/mask delta; "
            "identical WT recipe"
        ),
        "native_source": "src/xgraph/engine.py:native_tables, entity_rows, wide_masks",
    }


def validation_alias_prior(rows, affected_ids, cases):
    """Disclosed exposed all-new calibration, with a discounted rescue ceiling.

    Count only a WT error independently attributed to the frozen initial-case
    rule and reached by the gold-free surface screen. Half the weighted ceiling
    is the gain prior; a five-percent per-affected harm prior is a stress choice.
    Neither is an observed WT/WT+alias correctness transition.
    """
    affected = set(affected_ids)
    rows = [r for r in rows if r["id"] in affected]
    repairable = {
        r["id"] for r in cases if r.get("evidence", {}).get("initial_case_alias_covers_surface")
    }
    mass = sum(r["weight"] for r in rows)
    gain_mass = sum(r["weight"] for r in rows if not r["correct"] and r["id"] in repairable)
    return {
        "affected_validation_items": len(rows),
        "affected_weight": mass,
        "repairable_wt_errors": sum(not r["correct"] and r["id"] in repairable for r in rows),
        "repairable_weight": gain_mass,
        "gain_rate": 0.5 * gain_mass / mass if mass else 0,
        "harm_rate": 0.05,
        "rescue_discount": 0.5,
        "definition": (
            "half the exposed validation all-new initial-case-attributed WT error mass "
            "per surface-affected item"
        ),
        "disclosure": (
            "hypothesis-derived exposed calibration; discounted ceiling and assumed harm, "
            "not alias-arm outcomes"
        ),
    }
