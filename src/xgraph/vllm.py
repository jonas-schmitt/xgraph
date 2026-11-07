"""The engine in vLLM, as a Model Runner V2 logits processor.

    XGRAPH_KG_INDEX=<saved KG index> XGRAPH_TOKENIZER=<the model's tokenizer.json> \\
        vllm serve <model> --logits-processors xgraph.vllm:XGraphLogitsProcessor

A request opts in with the per-request argument `xgraph` set to 1, through the
OpenAI API as `extra_body={"vllm_xargs": {"xgraph": 1}}`; other requests are left
alone. With XGRAPH_ESCAPES=0 the escape marker is never allowed, so every span must
name the KG (an ablation). One engine state follows each of vLLM's request slots: vLLM keeps every
slot's token history on the GPU, and before each sampling step the engine advances
each opted-in slot over the tokens committed since the last step (from the end of
the prompt the first time, so a request resumed after preemption replays its
output), then sets every logit the slot's mask disallows to -inf, in place.

vLLM has no hook for a finished request; a slot is reset when its next request is
added. Under speculative decoding a request has one logits row per draft position,
and each row is masked as if the drafts before it had been committed (see
Engine.apply_mask); only accepted tokens reach the token history, so the slot's
state follows them. This module imports only the standard library at import
time, because vLLM also calls validate_params in its frontend process.

With XGRAPH_KG_VERSIONS=<version file> (xgraph.kgversions), the engine also holds KG
versions, named edit sets over the KG index, and a request reads one with the
per-request argument `xgraph_kg_version` (its name); requests without it read the KG
index itself, and requests of different versions share a batch (docs/kg-versions-adr.md).

A request with the per-request argument `xgraph_open` set to 1 starts inside a span (a
forced opener): its prompt ends with the span opener, as a prefilled assistant turn, and
its engine state starts as if the opener had just been written, so its first token
continues the span. It composes with an anchor and a KG version.

A request with the per-request argument `xgraph_prefill` set to a text replays that
prefill (a replayed prefill): its prompt ends with the text, as a prefilled assistant turn
such as a whole path span (`⟦ The Hobbit · author · place of birth⟧`), and the engine
advances over the prompt tokens that hold it, through the request's anchor and KG
version, exactly as if the model had written them, so that a path's frontier is bound for
what the model writes next. The replayed tokens are the fewest last prompt tokens whose
bytes cover the text; the first of them may also hold bytes of the text before it, and is
replayed whole, from free text. A prompt whose last tokens do not end with the text's
bytes (`tail mismatch`), or a replayed token the mask disallows (`disallowed`), declines
the request: it is left unmasked, never masked from a wrong state. The processor counts
each outcome (`replays`) and reports every decline on standard error. The replay runs
before the request's first mask, once its prompt is in vLLM's token history. It excludes
xgraph_open, which needs no tokens: the opener's state is set directly.

With XGRAPH_EXPRESSIONS=1 (which needs XGRAPH_PATHS=1 and XGRAPH_COMMITTED_PATHS=1, and an
index with the date table), a span may also be a date comparison,
`⟦ earlier: Jaws | Star Wars · publication date⟧`, whose result binds the answer after it.

XGRAPH_STOP=residual-goal (off by default) needs committed paths and each opted-in
request's question text as xgraph_question. The host compiles STOP relations once per
request and installs them in its slot before an opened span or a replayed prefill.
At a qualifying path relation boundary only the closer delimiter is allowed; comparisons
keep their expression rules (docs/engine-design.md).

XGRAPH_END_AFTER_ANSWER=1 (off by default, path mode) admits only the served model's
EOS ids after its answer span closes normally. Escaped answers keep existing behavior.
The EOS ids come from vLLM's served tokenizer and the model's generation config.

XGRAPH_RELATIONS=frontier|vocabulary|successor selects path relation support (frontier
by default). The weaker supports keep the start and span syntax, execute no path and
leave the answer unbound. They need path mode with STOP and expressions off.

XGRAPH_OBSERVE=k (0/unset off; 1..64 with frontier path mode and expressions off)
forces observations of the start and each relation's capped versioned frontier.
The engine emits canonical punctuation and indexed label tokens entirely on the GPU;
the adapter's sync, opened spans, replayed prefills and speculative rows use the same
slot protocol. A replay must contain observations tokenized in the canonical pieces
described in docs/observed-paths-adr.md; an incompatible prefill is declined as usual.

XGRAPH_AUDIT=1 (off by default) delays readback until a blind path closes, then
masks one terminal decision and forces a bound answer/EOS. XGRAPH_AUDIT_ENTITIES
and XGRAPH_AUDIT_RELATIONS bound endpoint and relation displays, default 20 and 8;
zero relations hides both menus. Each opted-in request supplies xgraph_question.
Its lexical relation ranks are compiled once on the host and installed after begin,
before an opened span or replay; decoding and speculative copies stay on the GPU.

Masked KG agent rounds use this same per-request opt-in. Each round is admitted as a
new request (begin resets its state), with its own KG version and, only in the first
forced round, xgraph_prefill="Action: execute_path ⟦". JSON lookups contain no spans;
tool path arguments use ordinary path spans. Earlier calls and observations are prompt
text, so no frontier or answer binding survives into the next round. Other requests
keep their existing opt-in behavior.
"""

from __future__ import annotations

import os
import sys
from typing import Any

try:
    from vllm.v1.worker.gpu.sample.logits_processor import LogitsProcessor as _Base
except ImportError:  # vLLM is not installed: the class still works with stand-ins
    _Base = object  # type: ignore[assignment,misc]

OPT_IN = "xgraph"
ANCHOR = "xgraph_anchor"
"""Per-request argument: the question's entity ids, where derivation mode starts."""
KG_VERSION = "xgraph_kg_version"
"""Per-request argument: the name of the KG version the request reads (XGRAPH_KG_VERSIONS)."""
VERSIONS = "XGRAPH_KG_VERSIONS"
OPEN = "xgraph_open"
"""Per-request argument: 1 starts the request inside a span (its prompt ends with the opener)."""
PREFILL = "xgraph_prefill"
"""Per-request argument: the text the prompt ends with, which the engine replays."""
REPLAY_ID = "xgraph_replay_id"
"""Opaque caller correlation key; a decline without a status consumer errors."""
QUESTION = "xgraph_question"
"""Per-request argument: the question text, required with XGRAPH_STOP=residual-goal."""
RELATION_RANKS = "xgraph_relation_ranks"
"""Optional (relation id, rank) pairs replacing lexical relation menu ranks.

At most 64 distinct known IDs, integer ranks in [0, 64); alias ranks stay lexical.
Ignored for menu/audit-free decoding after validation; never changes support.
"""
STOP = "xgraph_stop"
"""Per-request 0 disables STOP, for delivery of an already selected plan."""
PATH_OPTIONS = "xgraph_path_options"
"""Opt-in per-request menu/observation view and prompt demonstration count."""
REPLAYS = ("replayed", "tail mismatch", "disallowed")
"""The outcomes of a replayed prefill, which XGraphLogitsProcessor.replays counts."""


_REGISTERED: dict[str, list[str]] = {}
_REPLAY_RESULTS: dict[str, str] = {}


def collect_replay_status(_worker, keys: list[str]) -> dict[str, str]:
    """vLLM worker RPC: consume only this completed generation's replay outcomes."""
    # Resolve the worker's module even if the RPC callable is serialized by value.
    from xgraph.vllm import _REPLAY_RESULTS as results

    return {key: results.pop(key) for key in keys if key in results}


class XGraphWorkerExtension:
    """vLLM worker extension for replay receipts without callable serialization.

    Register as ``worker_extension_cls="xgraph.vllm.XGraphWorkerExtension"``
    when constructing an LLM that will consume replay statuses.
    """

    def xgraph_collect_replay_status(self, keys: list[str]) -> dict[str, str]:
        from xgraph.vllm import collect_replay_status

        return collect_replay_status(self, keys)


def replay_statuses(llm, keys: list[str], *, diagnostics: dict | None = None) -> dict[str, dict]:
    """A decline on any worker wins; absent/invalid receipts fail closed."""
    unavailable = "worker replay status unavailable"
    try:
        rpc = getattr(llm, "collective_rpc", None)
        # A string method crosses both the frontend Msgpack boundary (which
        # rejects functions by default) and single/multiprocess executors.
        results = (
            rpc("xgraph_collect_replay_status", args=(keys,), timeout=30) if callable(rpc) else []
        )
    except Exception as error:
        results = []
        unavailable = f"worker replay status failed: {type(error).__name__}: {str(error)[:512]}"
        if diagnostics is not None:
            diagnostics["rpc_error"] = unavailable
    if diagnostics is not None:
        from reprlib import repr as bounded_repr

        diagnostics.update(result_type=type(results).__name__, result_repr=bounded_repr(results))
    if not isinstance(results, (list, tuple)):
        results = []
        unavailable = "invalid worker replay status response"
    statuses = {}
    for key in keys:
        values = [item[key] for item in results if isinstance(item, dict) and key in item]
        declined = next(
            (v for v in values if isinstance(v, str) and v in ("tail mismatch", "disallowed")), None
        )
        if declined is not None:
            statuses[key] = {
                "replay_status": "declined",
                "replay_reason": declined,
                "declined": True,
            }
        elif (
            results
            and len(values) == len(results)
            and all(isinstance(v, str) and v == "replayed" for v in values)
        ):
            statuses[key] = {"replay_status": "replayed"}
        else:
            statuses[key] = {
                "replay_status": "unconfirmed",
                "replay_reason": unavailable,
                "error": True,
            }
    return statuses


def _checked_request_path_options(value):
    from xgraph.observations import PATH_OPTION_FIELDS, checked_path_options

    if isinstance(value, str):
        import json

        def unique_fields(pairs):
            result = {}
            for key, item in pairs:
                if key in result:
                    raise ValueError(f"xgraph_path_options repeats {key!r}")
                result[key] = item
            return result

        try:
            value = json.loads(value, object_pairs_hook=unique_fields)
        except ValueError as error:
            raise ValueError(f"xgraph_path_options must be a JSON object: {error}") from error
    capabilities = checked_path_options(
        {key: int(os.environ.get("XGRAPH_" + key.upper()) or "0") for key in PATH_OPTION_FIELDS},
        capability_union=True,
    )
    return checked_path_options(value, capabilities=capabilities)


def _version_names(path: str) -> list[str]:
    """The version names of a version file, read once per process."""
    if path not in _REGISTERED:
        from xgraph.kgversions import version_names

        _REGISTERED[path] = version_names(path)
    return _REGISTERED[path]


def replayed_tokens(tail: list[int], token_bytes, prefill: bytes) -> int | None:
    """How many of the last tokens of tail to replay for a prompt ending with prefill: the
    fewest whose bytes (token_bytes: a token's bytes, None for a special token or unused
    id) cover it, None if their bytes do not end with it."""
    written = b""
    for count, token in enumerate(reversed(tail), start=1):
        data = token_bytes(token)
        if data is None:
            return None
        written = data + written
        if len(written) >= len(prefill):
            return count if written.endswith(prefill) else None
    return None


def _setting(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(f"{name} is not set; the xgraph logits processor needs it")
    return value


class PendingRequest:
    """One slot's staged admission, followed by its delayed prefill replay."""

    def __init__(self):
        self.admissions: list[tuple[int, bool]] = []
        self.options: dict | None = None
        self.anchor: list[str] | None = None
        self.version: str | None = None
        self.question: str | None = None
        self.relation_question: str | None = None
        self.relation_ranks: list | None = None
        self.opened = False
        self.prefill: bytes | None = None
        self.replay_id: str | None = None
        self.order: dict[str, int] = {}

    def stage(self, name: str, value: Any, order: int) -> None:
        if value is None:
            self.order.pop(name, None)
        else:
            self.order.setdefault(name, order)
        setattr(self, name, value)


class XGraphLogitsProcessor(_Base):
    """vLLM Model Runner V2 adapter for the GPU constraint engine.

    Register ``xgraph.vllm:XGraphLogitsProcessor`` as a worker logits processor.
    Configure the saved index and tokenizer with ``XGRAPH_KG_INDEX`` and
    ``XGRAPH_TOKENIZER`` before worker construction. Each request opts in with
    ``extra_args={"xgraph": 1}`` offline, or ``vllm_xargs`` online.

    The worker owns this object and calls its lifecycle hooks. Request slots
    retain independent anchors, versions and committed history; prompt tokens
    are skipped. Replayed prefills additionally need correlated worker receipts
    from :func:`replay_statuses`. Import is CPU-safe, but construction requires
    a compatible runtime and the native GPU engine. Runtime upgrades require
    native agreement and real serving validation.
    """

    def __init__(self, vllm_config: Any, req_states: Any) -> None:
        import torch

        from xgraph.engine import Engine
        from xgraph.index import load_kg_index
        from xgraph.tokenizer import ServedTokenizer

        strict = os.environ.get("XGRAPH_STRICT_FINGERPRINT", "0") == "1"
        index = load_kg_index(_setting("XGRAPH_KG_INDEX"), strict=strict)
        tokenizer = ServedTokenizer.from_file(_setting("XGRAPH_TOKENIZER"))
        versions = None
        if os.environ.get(VERSIONS):  # KG versions a request can read (xgraph_kg_version)
            from xgraph.kgversions import load_kg_versions

            versions = load_kg_versions(os.environ[VERSIONS], index, strict=strict)
            for stats in versions.stats:
                print(f"xgraph: KG version {stats}", file=sys.stderr, flush=True)
        self._torch = torch
        self._req_states = req_states
        self._tokenizer = tokenizer
        self._relation_ids = frozenset(index.relations.owner_ids)
        # The span syntax is the one the index was built for (see KGIndex.span_syntax).
        # XGRAPH_ESCAPES=0 takes the escape marker away: every span must name the KG.
        # XGRAPH_REPEATS=0 keeps a sequence from closing a fact span it closed before.
        # XGRAPH_RELEASE_MASS=<p> and XGRAPH_RELEASE_TOP=1 release a span's step to the
        # escape marker when the model puts less than p on the allowed set, or when its
        # own top token is disallowed.
        escapes = os.environ.get("XGRAPH_ESCAPES", "1") != "0"
        repeats = os.environ.get("XGRAPH_REPEATS", "1") != "0"
        release_mass = float(os.environ.get("XGRAPH_RELEASE_MASS", "0"))
        release_top = os.environ.get("XGRAPH_RELEASE_TOP", "0") == "1"
        # XGRAPH_DERIVATION=1 binds spans by entity (Engine's derivation mode).
        derivation = os.environ.get("XGRAPH_DERIVATION", "0") == "1"
        # XGRAPH_ESCAPE_BIAS=<b> adds b to the logits of tokens that commit to the escape.
        escape_bias = float(os.environ.get("XGRAPH_ESCAPE_BIAS", "0"))
        # XGRAPH_PATHS=1: spans are paths of relations the KG walks (Engine's path mode).
        paths = os.environ.get("XGRAPH_PATHS", "0") == "1"
        # XGRAPH_BOUND_ESCAPES=0: a bound subject (an answer after a path) cannot escape.
        bound_escapes = os.environ.get("XGRAPH_BOUND_ESCAPES", "1") == "1"
        # XGRAPH_COMMITTED_PATHS=1: a path resolves to what its best start entity reaches.
        committed_paths = os.environ.get("XGRAPH_COMMITTED_PATHS", "0") == "1"
        # XGRAPH_START_RANK=sitelinks: best-connected order puts more sitelinks first (the
        # sitelink rank), for an index with sitelink counts; the default is degree.
        start_rank = os.environ.get("XGRAPH_START_RANK") or "degree"
        # XGRAPH_UNQUOTED_STARTS=1: an unbound subject cannot name a label that starts with
        # a quotation mark (unquoted starts).
        unquoted_starts = os.environ.get("XGRAPH_UNQUOTED_STARTS", "0") == "1"
        # XGRAPH_EXPRESSIONS=1 (with XGRAPH_PATHS=1 and XGRAPH_COMMITTED_PATHS=1): a span may
        # be a date comparison, for an index with the date table (Engine's expressions).
        expressions = os.environ.get("XGRAPH_EXPRESSIONS", "0") == "1"
        observe_select = os.environ.get("XGRAPH_OBSERVE_SELECT", "0") == "1"
        observe_answer = os.environ.get("XGRAPH_OBSERVE_ANSWER", "0") == "1"
        observe_width = int(os.environ.get("XGRAPH_OBSERVE_WIDTH") or "0")
        observe_defer = int(os.environ.get("XGRAPH_OBSERVE_DEFER") or "0")
        observe_terminal = int(os.environ.get("XGRAPH_OBSERVE_TERMINAL") or "0")
        terminal_order = os.environ.get("XGRAPH_TERMINAL_ORDER", "identity")
        terminal_seed = int(os.environ.get("XGRAPH_TERMINAL_SEED") or "17")
        menu_close_next = os.environ.get("XGRAPH_MENU_CLOSE_NEXT", "0") == "1"
        observe_answer = observe_answer or menu_close_next or bool(observe_defer)
        self._audit = os.environ.get("XGRAPH_AUDIT", "0") == "1"
        self._menu = int(os.environ.get("XGRAPH_MENU") or "0")
        self._request_path_options = os.environ.get("XGRAPH_REQUEST_PATH_OPTIONS") == "1"
        end_after_answer = (
            os.environ.get("XGRAPH_END_AFTER_ANSWER", "0") == "1"
            or observe_answer
            or self._audit
            or bool(observe_terminal)
        )
        eos_token_ids = []
        if end_after_answer:
            from vllm.tokenizers import cached_tokenizer_from_config

            # Match vLLM's input processor: the served tokenizer's EOS plus every
            # EOS in the model's generation config (also with generation_config="vllm").
            model_config = vllm_config.model_config
            served_tokenizer = cached_tokenizer_from_config(model_config)
            generation = model_config.try_get_generation_config()
            for ids in (
                getattr(served_tokenizer, "eos_token_id", None),
                generation.get("eos_token_id"),
            ):
                if ids is not None:
                    eos_token_ids.extend([ids] if isinstance(ids, int) else ids)
        self._stop = os.environ.get("XGRAPH_STOP") or None
        if expressions and not (paths and committed_paths):
            raise ValueError(
                "XGRAPH_EXPRESSIONS=1 needs path mode with committed paths: "
                "XGRAPH_PATHS=1 and XGRAPH_COMMITTED_PATHS=1"
            )
        self._engine = Engine(
            index,
            tokenizer,
            index.span_syntax,
            req_states.max_num_reqs,
            escapes=escapes,
            repeats=repeats,
            release_mass=release_mass,
            release_top=release_top,
            derivation=derivation,
            escape_bias=escape_bias,
            paths=paths,
            bound_escapes=bound_escapes,
            committed_paths=committed_paths,
            versions=versions,
            start_rank=start_rank,
            unquoted_starts=unquoted_starts,
            expressions=expressions,
            stop=self._stop,
            relations=os.environ.get("XGRAPH_RELATIONS", "frontier"),
            end_after_answer=end_after_answer,
            eos_token_ids=eos_token_ids,
            observe=int(os.environ.get("XGRAPH_OBSERVE") or "0"),
            **({"observe_width": observe_width} if observe_width else {}),
            **({"observe_defer": observe_defer} if observe_defer else {}),
            **({"observe_terminal": observe_terminal} if observe_terminal else {}),
            **(
                {"terminal_order": terminal_order, "terminal_seed": terminal_seed}
                if terminal_order != "identity" or terminal_seed != 17
                else {}
            ),
            **({"menu_close_next": True} if menu_close_next else {}),
            observe_select=observe_select,
            observe_answer=observe_answer,
            observe_required=os.environ.get("XGRAPH_OBSERVE_REQUIRED", "0") == "1",
            observe_no_loop=os.environ.get("XGRAPH_OBSERVE_NO_LOOP", "0") == "1",
            **({"menu": self._menu} if self._menu else {}),
            **({"request_path_options": True} if self._request_path_options else {}),
            **(
                {"relation_labels": os.environ["XGRAPH_RELATION_LABELS"]}
                if (self._audit or self._menu) and os.environ.get("XGRAPH_RELATION_LABELS")
                else {}
            ),
            **(
                {
                    "audit": True,
                    "audit_entities": int(os.environ.get("XGRAPH_AUDIT_ENTITIES") or "20"),
                    "audit_relations": int(os.environ.get("XGRAPH_AUDIT_RELATIONS") or "8"),
                }
                if self._audit
                else {}
            ),
        )
        self._pending: dict[int, PendingRequest] = {}
        self._replaying: dict[int, PendingRequest] = {}
        self._admission_order = 0
        self.replays = dict.fromkeys(REPLAYS, 0)

    @classmethod
    def validate_params(cls, sampling_params: Any) -> None:
        extra = sampling_params.extra_args or {}
        value = extra.get(OPT_IN)
        if value not in (None, 0, 1) or isinstance(value, float):
            raise ValueError(f"the per-request argument {OPT_IN!r} must be 0 or 1, got {value!r}")
        question = extra.get(QUESTION)
        if RELATION_RANKS in extra:
            from xgraph.audit import checked_relation_ranks

            if value != 1:
                raise ValueError(f"{RELATION_RANKS} needs xgraph=1")
            checked_relation_ranks(extra[RELATION_RANKS])
        if PATH_OPTIONS in extra:
            if value != 1 or os.environ.get("XGRAPH_REQUEST_PATH_OPTIONS") != "1":
                raise ValueError(f"{PATH_OPTIONS} needs xgraph=1 and XGRAPH_REQUEST_PATH_OPTIONS=1")
            options = _checked_request_path_options(extra[PATH_OPTIONS])
            if options["menu"] and (not isinstance(question, str) or not question.strip()):
                raise ValueError(f"menu needs the per-request argument {QUESTION!r}")
        stop = extra.get(STOP)
        if stop is not None and (type(stop) is not int or stop != 0):
            raise ValueError(f"the per-request argument {STOP!r} only accepts 0, got {stop!r}")
        if stop is not None and value != 1:
            raise ValueError(f"{STOP} needs the engine: set {OPT_IN!r} to 1")
        if (
            (
                value == 1
                and (
                    (stop != 0 and os.environ.get("XGRAPH_STOP"))
                    or os.environ.get("XGRAPH_AUDIT") == "1"
                )
            )
            or question is not None
        ) and (not isinstance(question, str) or not question.strip()):
            raise ValueError(f"the per-request argument {QUESTION!r} must be the question text")
        opened = extra.get(OPEN)
        if opened not in (None, 0, 1) or isinstance(opened, float):
            raise ValueError(f"the per-request argument {OPEN!r} must be 0 or 1, got {opened!r}")
        if opened and value != 1:
            raise ValueError(
                f"{OPEN} needs the engine: set the per-request argument {OPT_IN!r} to 1"
            )
        prefill = extra.get(PREFILL)
        if REPLAY_ID in extra and (
            not isinstance(extra[REPLAY_ID], str) or not extra[REPLAY_ID] or not prefill
        ):
            raise ValueError(f"{REPLAY_ID} needs a nonempty string and {PREFILL}")
        if prefill is not None:
            if not isinstance(prefill, str) or not prefill:
                raise ValueError(
                    f"the per-request argument {PREFILL!r} must be the text the prompt ends "
                    f"with, got {prefill!r}"
                )
            if value != 1:
                raise ValueError(
                    f"{PREFILL} needs the engine: set the per-request argument {OPT_IN!r} to 1"
                )
            if opened:
                raise ValueError(f"{PREFILL} and {OPEN} exclude each other: replay the opener")
        version = extra.get(KG_VERSION)
        if version is None:
            return
        if not isinstance(version, str):
            raise ValueError(
                f"the per-request argument {KG_VERSION!r} must be a KG version's name, "
                f"got {version!r}"
            )
        if not os.environ.get(VERSIONS):
            raise ValueError(
                f"{KG_VERSION} names {version!r}, but no KG versions are registered "
                f"({VERSIONS} is not set)"
            )
        names = _version_names(os.environ[VERSIONS])
        if version not in names:
            raise ValueError(f"no KG version {version!r}; the registered ones are {names}")

    def add_request(self, req_idx: int, sampling_params: Any) -> bool:
        extra = sampling_params.extra_args or {}
        on = bool(extra.get(OPT_IN))
        ranked = None
        if RELATION_RANKS in extra:
            from xgraph.audit import checked_relation_ranks

            if not on:
                raise ValueError(f"{RELATION_RANKS} needs xgraph=1")
            ranked = checked_relation_ranks(extra[RELATION_RANKS], self._relation_ids)
        order = self._admission_order
        self._admission_order += 1
        pending = self._pending.setdefault(req_idx, PendingRequest())
        pending.relation_ranks = ranked
        options = extra.get(PATH_OPTIONS)
        if PATH_OPTIONS in extra:
            if not on or not self._request_path_options:
                raise ValueError(f"{PATH_OPTIONS} needs xgraph=1 and XGRAPH_REQUEST_PATH_OPTIONS=1")
            options = _checked_request_path_options(options)
        pending.stage("options", options, order)
        menu = options["menu"] if options is not None else self._menu
        if on and (self._audit or menu):
            question = extra.get(QUESTION)
            if not isinstance(question, str) or not question.strip():
                mode = "audit" if self._audit else "menu"
                raise ValueError(f"{mode} needs the per-request argument {QUESTION!r}")
            pending.stage("relation_question", question, order)
        else:
            pending.stage("relation_question", None, order)
        if on and self._stop and extra.get(STOP) != 0:
            question = extra.get(QUESTION)
            if not isinstance(question, str) or not question.strip():
                raise ValueError(f"residual-goal STOP needs the per-request argument {QUESTION!r}")
            pending.stage("question", question, order)
        else:
            pending.stage("question", None, order)
        pending.admissions.append((order, on))
        if on and ANCHOR in extra:  # derivation mode: the question's entities
            pending.stage("anchor", [str(e) for e in extra[ANCHOR]], order)
        pending.stage(
            "version", str(extra[KG_VERSION]) if on and extra.get(KG_VERSION) else None, order
        )
        pending.opened = bool(on and extra.get(OPEN))
        if on and extra.get(PREFILL):  # its prompt ends with this text: replay it
            pending.prefill = str(extra[PREFILL]).encode("utf-8")
            pending.replay_id = extra.get(REPLAY_ID) or None
            self._replaying[req_idx] = pending
        else:
            self._replaying.pop(req_idx, None)
        return on

    def apply_staged_writes(self) -> None:
        admissions = sorted(
            (order, slot, on)
            for slot, pending in self._pending.items()
            for order, on in pending.admissions
        )
        for on, start in ((True, self._engine.begin), (False, self._engine.end)):
            slots = [slot for _, slot, enabled in admissions if enabled == on]
            if slots:
                start(self._slots(slots), self._stream())
                for pending in self._pending.values():
                    pending.admissions = [
                        (order, flag) for order, flag in pending.admissions if flag != on
                    ]
        for field, method in (
            ("options", "set_path_options"),
            ("anchor", "anchor"),
            ("version", "set_version"),
            ("question", "set_stop"),
            ("relation_question", "set_relation_question"),
        ):
            writes = sorted(
                (pending.order[field], slot, getattr(pending, field))
                for slot, pending in self._pending.items()
                if field in pending.order
            )
            for _, slot, value in writes:
                ranked = self._pending[slot].relation_ranks
                if field == "relation_question" and ranked is not None:
                    self._engine.set_relation_question(slot, value, self._stream(), ranked=ranked)
                else:
                    getattr(self._engine, method)(slot, value, self._stream())
            for _, slot, _ in writes:
                self._pending[slot].stage(field, None, 0)
        for slot in {slot for slot, pending in self._pending.items() if pending.opened}:
            self._engine.open_span(slot, self._stream())
        self._pending.clear()

    def apply(self, logits: Any, ctx: Any) -> Any:
        int32 = self._torch.int32
        state = self._req_states
        stream = self._stream()
        if self._replaying:  # before the slots' first sync: their cursors are still unset
            self._replay(stream)
        self._engine.sync(
            ctx.idx_mapping.to(int32),
            state.all_token_ids.gpu,
            state.prompt_len.gpu,
            state.total_len.gpu,
            stream,
        )
        rows = ctx.expanded_idx_mapping.to(int32)
        if rows.shape[0] == ctx.idx_mapping.shape[0]:  # one row per request: no drafts
            self._engine.apply_mask(logits, rows, stream)
        else:
            self._engine.apply_mask(
                logits,
                rows,
                stream,
                local_pos=ctx.expanded_local_pos.to(int32),
                input_ids=ctx.input_ids.to(int32),
            )
        return logits

    def _replay(self, stream: int) -> None:
        """Advance each slot with a pending prefill over the prompt tokens that hold it,
        then decline (end) those whose tail does not match or whose replay hit a token
        outside the mask (see the module docstring)."""
        from xgraph.engine import PHASE_ERROR
        from xgraph.vllm import _REPLAY_RESULTS as results

        torch, state = self._torch, self._req_states
        pending, self._replaying = self._replaying, {}
        history = state.all_token_ids.gpu
        start = torch.zeros_like(state.prompt_len.gpu)
        stop = torch.zeros_like(state.prompt_len.gpu)
        replayed, declined = [], []
        for slot, request in sorted(pending.items()):
            prefill = request.prefill
            end = int(state.prompt_len.gpu[slot])
            tail = history[slot, max(0, end - len(prefill)) : end].tolist()
            count = replayed_tokens(tail, self._tokenizer.token_bytes, prefill)
            if count is None:
                declined.append((slot, "tail mismatch"))
                continue
            start[slot], stop[slot] = end - count, end
            replayed.append(slot)
        if replayed:  # sync from start to stop: the cursors then stand at the prompt's end
            self._engine.sync(self._slots(replayed), history, start, stop, stream)
            phases = self._engine.states_host(max(replayed) + 1)["phase"]
            for slot in replayed:
                if phases[slot] == PHASE_ERROR:
                    declined.append((slot, "disallowed"))
                else:
                    self.replays["replayed"] += 1
                    if key := pending[slot].replay_id:
                        # A preempted/re-admitted request must never erase a decline.
                        results.setdefault(key, "replayed")
        if declined:
            self._engine.end(self._slots([slot for slot, _ in declined]), stream)
            for slot, reason in declined:
                self.replays[reason] += 1
                if key := pending[slot].replay_id:
                    results[key] = reason
                print(
                    f"xgraph: declined the replayed prefill of the request in slot {slot} "
                    f"({reason}); it is left unmasked. Replays so far: {self.replays}",
                    file=sys.stderr,
                    flush=True,
                )
            if any(pending[slot].replay_id is None for slot, _ in declined):
                raise RuntimeError("xgraph replay declined without a request status consumer")

    def _slots(self, slots: list[int]) -> Any:
        return self._torch.tensor(slots, dtype=self._torch.int32, device=self._req_states.device)

    def _stream(self) -> int:
        return self._torch.cuda.current_stream().cuda_stream
