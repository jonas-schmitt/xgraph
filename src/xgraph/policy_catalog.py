"""Prospectively named complete policies shared by validation, cost and transfer."""

from __future__ import annotations

import hashlib
import json

FINALISTS = ("W", "M2", "MT", "WT", "B2", "BT")
CATALOG = (
    *FINALISTS,
    "menuobs-audit",
    "menuobs-audit-terminal",
    "blind5",
    *(f"agent-strong-{n}" for n in (1, 2, 4, 8)),
    "agent-feedback-8",
)
ALIASES = {
    "agent-strong-8-feedback": "agent-feedback-8",
    "menuobs-width8": "W",
    "menuobs-two-pass": "M2",
    "menuobs-two-pass-terminal": "MT",
    "menuobs-width8-two-pass-terminal": "WT",
    "blind5-two-pass": "B2",
    "blind5-two-pass-terminal": "BT",
}
GENERATION_DEFAULTS = {
    "--max-model-len": 8192,
    "--max-num-seqs": 64,
    "--kv-cache-gib": 8.0,
    "--repetition-penalty": 1.0,
    "--samples": 1,
    "--think-budget": 0,
}


def recipe(name, family=None):
    name = ALIASES.get(name, name)
    if name not in CATALOG:
        raise ValueError(f"unknown catalog policy {name!r}")
    agent = name.startswith("agent-")
    blind = name in ("blind5", "B2", "BT")
    menu = not agent and not blind
    terminal = name in ("MT", "WT", "BT", "menuobs-audit-terminal")
    second = (
        ("audit" if name.startswith("menuobs-audit") else "two-pass")
        + ("-terminal" if terminal else "")
        if name in ("M2", "MT", "WT", "B2", "BT", "menuobs-audit", "menuobs-audit-terminal")
        else None
    )
    spec = {
        "policy": name,
        "menu": 5 if menu else 0,
        "observe": 3 if menu else 0,
        "observe_width": 8 if name in ("W", "WT") else 0,
        "path_demos": None if agent else 5,
        "max_tokens": 256 if agent or (blind and family in ("gemma31", "qwen25")) else 1024,
        "seed": 17 if agent else 0,
        "second": second,
        "second_max_tokens": 256,
        "second_seed": 17,
        "terminal_width": 8 if terminal else 0,
        "agent_budget": int(name.rsplit("-", 1)[1]) if agent else None,
        "agent_feedback": name == "agent-feedback-8",
    }
    spec["generation"] = {}
    for stage in ("first", "second") if second else ("first",):
        extended = agent or stage == "second"
        argv = ["--max-tokens", str(spec["max_tokens"] if stage == "first" else 256)]
        if extended:
            argv += [
                "--max-model-len",
                "16384",
                "--max-num-seqs",
                "64",
                "--kv-cache-gib",
                "16" if family == "gemma31" else "8",
            ]
        argv += ["--seed", str(spec["seed"] if stage == "first" else 17)]
        if family == "gemma31":
            argv += ["--ban-tokens", "<|channel>"]
        if agent:
            argv += [
                "--max-calls",
                str(spec["agent_budget"]),
                "--matched-demos",
                "--matched-paths",
                "--pages",
                "--start-rank",
                "sitelinks",
                "--unquoted-starts",
            ]
            if spec["agent_feedback"]:
                argv.append("--agent-feedback")
        argv += ["--memory-share", "0.328" if family == "qwen25" else "0.450"]
        spec["generation"][stage] = {
            "argv": argv,
            "path_demos": 5 if stage == "first" and (blind or name in ("W", "WT")) else None,
            "omitted_defaults": {k: v for k, v in GENERATION_DEFAULTS.items() if k not in argv},
        }
    return spec


def fingerprint(spec):
    return hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest()


def selection(shared, overrides, families):
    result = dict.fromkeys(families, recipe(shared)["policy"] if shared else None)
    seen = set()
    for item in overrides or ():
        family, separator, name = item.partition("=")
        if not separator or family not in result or family in seen:
            raise ValueError(f"expected one FAMILY=POLICY declaration per known family: {item}")
        result[family] = recipe(name)["policy"]
        seen.add(family)
    if overrides and any(v is None for v in result.values()):
        raise ValueError("declare every family or supply a shared --policy fallback")
    return result if any(result.values()) else {}


def frozen_recipe(path, family, *, model=None):
    """Read the sealed policy through validation's full provenance checks."""
    from pathlib import Path

    from xgraph.mintqa_validation import load_frozen
    from xgraph.provenance import sha256_file

    manifest = load_frozen(path, policy_consumer=True)
    if family not in manifest.get("policy_selection", {}):
        raise ValueError("frozen policy needs an explicitly selected model family")
    selected = manifest["families"][family]
    spec = recipe(manifest["policy_selection"][family], family)
    if selected.get("selected_policy") != spec or selected.get("policy_sha256") != fingerprint(
        spec
    ):
        raise ValueError("frozen policy recipe changed")
    if model is not None and Path(selected["model"]["path"]).resolve() != Path(model).resolve():
        raise ValueError("frozen policy served model changed")
    return {
        "family": family,
        "recipe": spec,
        "comparators": selected.get("agent_comparators", []),
        "strongest_agent": manifest.get("strongest_agents", {}).get(family),
        "freeze_sha256": sha256_file(path),
    }
