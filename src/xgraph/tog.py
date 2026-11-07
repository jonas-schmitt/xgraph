"""Think-on-Graph: relation pruning, entity scoring, global width, then reasoning.

Official Wikidata prompts are loaded as literal data. No downloaded code is executed.
This is a SQLite/shared-backbone transfer, with explicit deviations in every record.
"""

from __future__ import annotations

import hashlib
import math
import random
import re
from copy import copy
from dataclasses import asdict, dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from xgraph.cor import parse_fallback
from xgraph.gcr import chat_token_ids
from xgraph.method_prompts import PromptSource
from xgraph.toolkg import store_for

SOURCE_COMMIT = "7ccbb92e17579f934bb778386230de47eca0ab67"
DEVIATIONS = (
    "MINTQA SQLite KG and linked topic entities replace the native KG service and dataset topics",
    "shared local vLLM backbone/chat template; thinking disabled; no published-score reproduction",
    "released per-entity relation width precedes global entity pruning; no global relation prune",
    "released sufficiency reply can supply the answer without the paper's separate answer call",
    "Wikidata label noise filter disabled to retain benchmark relations including country",
    "branch-local immediate inverse exclusion; stable PID/QID ties and seeded entity subsampling",
    "inverse triples retain factual orientation; release serializes traversal orientation",
    "context overflow returns Unknown with all issued calls charged; no prompt truncation",
    "final braced answer delivered as an xgraph Answer span without another model call",
)


@dataclass(frozen=True)
class ToGConfig:
    variant: str = "tog"
    width: int = 3
    depth: int = 3
    max_tokens: int = 256
    exploration_temperature: float = 0.4
    reasoning_temperature: float = 0.0
    seed: int = 17
    max_prompt_tokens: int | None = None
    max_calls: int | None = None

    def __post_init__(self):
        if self.variant not in ("tog", "tog-r"):
            raise ValueError("ToG variant must be tog or tog-r")
        for name in ("width", "depth", "max_tokens", "max_prompt_tokens", "max_calls"):
            value = getattr(self, name)
            if value is None and name in ("max_prompt_tokens", "max_calls"):
                continue
            if type(value) is not int or value <= 0:
                raise ValueError(f"ToG {name} must be a positive integer")
        for value in (self.exploration_temperature, self.reasoning_temperature):
            if not math.isfinite(value) or value < 0:
                raise ValueError("ToG temperatures must be finite and nonnegative")


def method_deviations(config: ToGConfig) -> list[str]:
    if config.variant == "tog":
        return list(DEVIATIONS)
    return [d for d in DEVIATIONS if not d.startswith(("released per-entity", "branch-local"))] + [
        "ToG-R paper Algorithm 2 with reconstructed relation-chain reasoning serialization",
        "branch-local inverse exclusion; stable ties and seeded entity/path random pruning",
        "ToG-R retains all endpoint evidence before random pruning; no LLM entity scoring",
    ]


@dataclass(frozen=True)
class ToGPrompts:
    source: PromptSource

    @classmethod
    def load(cls, path: str | Path) -> ToGPrompts:
        source = PromptSource.load(
            path,
            (
                "extract_relation_prompt_wiki",
                "score_entity_candidates_prompt_wiki",
                "prompt_evaluate_wiki",
                "answer_prompt_wiki",
                "cot_prompt",
            ),
        )
        try:
            source.strings["extract_relation_prompt_wiki"] % (3, 3)
            source.strings["score_entity_candidates_prompt_wiki"].format("q", "r")
        except (KeyError, ValueError, TypeError, IndexError) as error:
            raise ValueError(f"ToG prompt placeholders {path}: {error}") from error
        return cls(source)

    def render(self, stage: str, *, question: str, **values) -> str:
        strings = self.source.strings
        if stage == "relations":
            width = values["width"]
            return (
                strings["extract_relation_prompt_wiki"] % (width, width)
                + question
                + "\nTopic Entity: "
                + values["entity"]
                + "\nRelations:\n"
                + "\n".join(f"{i}. {r}" for i, r in enumerate(values["relations"], 1))
                + "A:"
            )
        if stage == "entities":
            return (
                strings["score_entity_candidates_prompt_wiki"].format(question, values["relation"])
                + "; ".join(values["entities"])
                + "\nScore: "
            )
        if stage == "fallback":
            return strings["cot_prompt"] + "\n\nQ: " + question + "\nA:"
        if "relation_chains" in values:
            groups = {}
            for row in values["relation_chains"]:
                chain = tuple([row["start_label"], *row["relation_labels"]])
                groups.setdefault(chain, set()).add(row["label"])
            chains = "\n".join(
                " -> ".join(chain) + " -> {" + "; ".join(sorted(ends)) + "}"
                for chain, ends in groups.items()
            )
            key = "prompt_evaluate_wiki" if stage == "reasoning" else "answer_prompt_wiki"
            return (
                strings[key]
                + question
                + "\nRelation Chains (prefix ^ means inverse traversal):\n"
                + chains
                + "A: "
            )
        triples = "\n".join(", ".join(triple) for triple in values["triples"])
        key = "prompt_evaluate_wiki" if stage == "reasoning" else "answer_prompt_wiki"
        return strings[key] + question + "\nKnowledge Triplets: " + triples + "A: "


def parse_relations(text: str, offered: list[str]) -> list[tuple[str, float]]:
    result, seen = [], set()
    for name, raw in re.findall(r"\{\s*([^{}()]+?)\s+\(Score:\s*([0-9.]+)\)\}", text):
        name = name.strip().removeprefix("wiki.relation.").replace("_", " ")
        try:
            score = float(raw)
        except ValueError:
            continue
        if name in offered and name not in seen and math.isfinite(score) and 0 <= score <= 1:
            result.append((name, score))
            seen.add(name)
    return result


def parse_entity_scores(text: str, count: int) -> list[float]:
    if count < 1:
        raise ValueError("ToG entity scores need nonempty candidates")
    scores = [float(s) for s in re.findall(r"\d+\.\d+", text)]
    if len(scores) != count or not any(scores) or any(not 0 <= s <= 1 for s in scores):
        return [1 / count] * count
    return scores


def parse_reasoning(text: str) -> tuple[bool, str | None]:
    braces = re.findall(r"\{([^{}]*)\}", text)
    stop = bool(braces and braces[0].strip().casefold().replace(" ", "") == "yes")
    return stop, (braces[-1].strip() if stop and len(braces) > 1 else None)


def run_tog(
    llm: Any,
    requests: list[dict],
    store: Any,
    sampling: Any,
    *,
    prompts: ToGPrompts,
    config: ToGConfig | None = None,
    versions: dict | None = None,
    lora_request: Any = None,
    batch_size: int = 32,
) -> list[dict]:
    """Advance independent question episodes one model call per scheduling wave.

    Each episode owns its RNG, search state and usage. A wave submits at most one
    pending call per question to the same engine; batch_size=1 is serial scheduling.
    """
    if type(batch_size) is not int or batch_size < 1:
        raise ValueError("ToG batch_size must be a positive integer")
    config = config or ToGConfig()
    tokenizer = llm.get_tokenizer() if config.max_prompt_tokens is not None else None
    records = [None] * len(requests)
    pending = []
    at = 0

    def advance(index, episode, output=None):
        try:
            messages, params = episode.send(output)
        except StopIteration as done:
            records[index] = done.value
        else:
            pending.append((index, episode, messages, params))

    while at < len(requests) or pending:
        while at < len(requests) and len(pending) < batch_size:
            advance(
                at, _episode(requests[at], store, sampling, prompts, config, versions, tokenizer)
            )
            at += 1
        if not pending:
            continue
        wave, pending = pending, []
        outputs = llm.chat(
            [messages for _, _, messages, _ in wave],
            [params for _, _, _, params in wave],
            use_tqdm=False,
            chat_template_kwargs={"enable_thinking": False},
            **({"lora_request": lora_request} if lora_request is not None else {}),
        )
        if len(outputs) != len(wave) or any(len(o.outputs) != 1 for o in outputs):
            raise ValueError("ToG needs exactly one completion per issued call")
        for (index, episode, _, _), output in zip(wave, outputs, strict=True):
            advance(index, episode, output)
    return records


def _episode(request, store, sampling, prompts, config, versions, tokenizer):
    question = request.get("question")
    if not isinstance(question, str) or not question.strip():
        raise ValueError(f"ToG request {request['id']!r} needs a nonempty question")
    topics = request.get("anchor") or request.get("topics") or []
    if not isinstance(topics, list) or any(
        not isinstance(s, str) or not re.fullmatch(r"Q\d+", s) for s in topics
    ):
        raise ValueError("ToG anchor/topics must be QID lists")
    read = store_for(request, store, versions)
    turns, trace, layers, caps = [], [], [], []
    status, final = "searching", None
    rng = random.Random(
        config.seed + int.from_bytes(hashlib.sha256(str(request["id"]).encode()).digest()[:8])
    )

    def issue(stage, question=question, request=request, turns=turns, trace=trace, **values):
        nonlocal status
        messages = [
            {
                "role": "system",
                "content": "You are an AI assistant that helps people find information.",
            },
            {"role": "user", "content": prompts.render(stage, question=question, **values)},
        ]
        if config.max_calls is not None and len(turns) >= config.max_calls:
            status = "call_limit"
            return None
        if (
            tokenizer is not None
            and len(chat_token_ids(tokenizer, messages)) > config.max_prompt_tokens
        ):
            status = "context_limit"
            trace.append({"event": status, "stage": stage})
            return None
        one = copy(sampling) if sampling is not None else SimpleNamespace()
        one.max_tokens = config.max_tokens
        one.temperature = (
            config.exploration_temperature
            if stage in ("relations", "entities")
            else config.reasoning_temperature
        )
        one.seed = request.get("seed", config.seed)
        output = yield messages, one
        completion = output.outputs[0]
        turns.append(
            {
                "stage": stage,
                "messages": messages,
                "text": completion.text,
                "token_ids": list(completion.token_ids),
                "prompt_tokens": len(output.prompt_token_ids),
                "generated_tokens": len(completion.token_ids),
                "finish_reason": completion.finish_reason,
            }
        )
        if completion.finish_reason == "abort":
            status = "aborted"
            return None
        return completion.text

    # A frontier entry retains its predecessor's signed relation for inverse exclusion.
    frontier = [(int(q[1:]), 0) for q in dict.fromkeys(topics) if read.label(int(q[1:]))]
    starts = [f"Q{item}" for item, _ in frontier]
    if config.variant == "tog-r":
        final = yield from _relation_search(read, frontier, issue, config, rng, layers, trace)
    else:
        for _ in range(config.depth):
            candidates = []
            for item, previous in frontier:
                menu = {
                    (-p if reverse else p): label
                    for reverse in (False, True)
                    for p, label in sorted(read.relations_of([item], reverse=reverse).items())
                    if (-p if reverse else p) != -previous
                }
                if not menu:
                    continue
                labels = sorted(set(menu.values()))
                reply = yield from issue(
                    "relations", entity=read.label(item), relations=labels, width=config.width
                )
                if reply is None:
                    break
                selected = parse_relations(reply, labels)[: config.width]
                if not selected or len(re.findall(r"\(Score:", reply)) != len(selected):
                    trace.append(
                        {"event": "invalid_relation", "entity": f"Q{item}", "reply": reply}
                    )
                for label, relation_score in selected:
                    signed = min(
                        (p for p in menu if menu[p] == label), key=lambda p: (p < 0, abs(p))
                    )
                    entities = [
                        (read.label(other), other)
                        for other in read.neighbours_of([item], abs(signed), reverse=signed < 0)
                        if read.label(other)
                        and read.label(other) not in ("N/A", "Unname_Entity", "UnName_Entity")
                    ]
                    if len(entities) >= 20:
                        caps.append(
                            {
                                "stage": "entity_subsample",
                                "entity": f"Q{item}",
                                "relation": signed,
                                "before": len(entities),
                                "kept": 10,
                            }
                        )
                        entities = rng.sample(entities, 10)
                    entities.sort()
                    if not entities:
                        continue
                    weights = [1.0]
                    if len(entities) > 1:
                        reply = yield from issue(
                            "entities", relation=label, entities=[e[0] for e in entities]
                        )
                        if reply is None:
                            break
                        weights = parse_entity_scores(reply, len(entities))
                    for (name, other), weight in zip(entities, weights, strict=True):
                        subject, obj = (other, item) if signed < 0 else (item, other)
                        candidates.append(
                            {
                                "entity": f"Q{other}",
                                "from": f"Q{item}",
                                "relation": f"P{abs(signed)}",
                                "reverse": signed < 0,
                                "triple": [read.label(subject), label, read.label(obj)],
                                "score": relation_score * weight,
                                "label": name,
                            }
                        )
                if status != "searching":
                    break
            if status != "searching":
                break
            chosen = sorted(
                candidates,
                key=lambda c: (
                    -c["score"],
                    int(c["from"][1:]),
                    int(c["relation"][1:]),
                    int(c["entity"][1:]),
                    c["reverse"],
                ),
            )[: config.width]
            chosen = [c for c in chosen if c["score"] > 0]
            if not chosen:
                reply = yield from issue(
                    "answer" if layers else "fallback",
                    triples=[c["triple"] for layer in layers for c in layer],
                )
                final = reply
                break
            layers.append(chosen)
            reply = yield from issue(
                "reasoning", triples=[c["triple"] for layer in layers for c in layer]
            )
            if reply is None:
                break
            stop, answer = parse_reasoning(reply)
            if stop:
                final = (
                    "{" + answer + "}"
                    if answer
                    else (
                        yield from issue(
                            "answer", triples=[c["triple"] for layer in layers for c in layer]
                        )
                    )
                )
                break
            frontier = list(
                dict.fromkeys(
                    (int(c["entity"][1:]), (-1 if c["reverse"] else 1) * int(c["relation"][1:]))
                    for c in chosen
                )
            )
        else:
            # Official main_wiki.py discards the explored paths after depth exhaustion.
            final = yield from issue("fallback")
            trace.append({"event": "depth_exhausted", "fallback": "released_cot"})
    braces = re.findall(r"\{([^{}]*)\}", final or "")
    answer, _ = parse_fallback(braces[-1] if braces else final or "")
    if status == "searching":
        status = "answered" if answer else "invalid_answer"
    return {
        "id": request["id"],
        "arm": config.variant,
        "text": answer or "Answer: ⟦ Unknown⟧",
        **({"kg_version": request["kg_version"]} if request.get("kg_version") else {}),
        "status": status,
        "start": starts,
        "layers": layers,
        "trace": trace,
        "cap_events": caps,
        "config": asdict(config),
        "prompts": prompts.source.metadata(),
        "source_commit": SOURCE_COMMIT,
        "deviations": method_deviations(config),
        "model_calls": len(turns),
        "rounds": len(turns),
        "turns": turns,
        "token_ids": [],
        "prompt_tokens": sum(t["prompt_tokens"] for t in turns),
        "generated_tokens": sum(t["generated_tokens"] for t in turns),
        "finish_reason": turns[-1]["finish_reason"] if turns else "length",
        "context_limited": status == "context_limit",
        "budget_spent": status == "call_limit",
    }


def _relation_search(read, frontier, issue, config, rng, layers, trace):
    """Paper Algorithm 2: reason over every endpoint, then random-prune entities."""
    frontier = [
        {
            "item": item,
            "previous": previous,
            "start_label": read.label(item),
            "relations": [],
            "relation_labels": [],
        }
        for item, previous in frontier
    ]
    for _ in range(config.depth):
        relations = []
        for node in frontier:
            item = node["item"]
            menu = {
                (-p if reverse else p): label
                for reverse in (False, True)
                for p, label in sorted(read.relations_of([item], reverse=reverse).items())
                if (-p if reverse else p) != -node["previous"]
            }
            if not menu:
                continue
            offered = sorted(set(menu.values()))
            reply = yield from issue(
                "relations",
                entity=read.label(item) or f"Q{item}",
                relations=offered,
                width=config.width,
            )
            if reply is None:
                return None
            selected = parse_relations(reply, offered)[: config.width]
            if not selected or len(re.findall(r"\(Score:", reply)) != len(selected):
                trace.append({"event": "invalid_relation", "entity": f"Q{item}", "reply": reply})
            for label, score in selected:
                signed = min((p for p in menu if menu[p] == label), key=lambda p: (p < 0, abs(p)))
                if score > 0:
                    relations.append({**node, "signed": signed, "label": label, "score": score})
        # Relation width is global in the paper, before the entity search.
        relations.sort(key=lambda n: (-n["score"], n["item"], abs(n["signed"]), n["signed"] < 0))
        candidates = []
        for node in relations[: config.width]:
            signed = node["signed"]
            for other in read.neighbours_of([node["item"]], abs(signed), reverse=signed < 0):
                candidates.append(
                    {
                        "entity": f"Q{other}",
                        "item": other,
                        "previous": signed,
                        "from": f"Q{node['item']}",
                        "score": node["score"],
                        "start_label": node["start_label"],
                        "relations": [*node["relations"], signed],
                        "relation_labels": [
                            *node["relation_labels"],
                            ("^" if signed < 0 else "") + node["label"],
                        ],
                        "label": read.label(other) or f"Q{other}",
                    }
                )
        if not candidates:
            return (
                yield from issue(
                    "answer" if layers else "fallback",
                    relation_chains=layers[-1] if layers else [],
                )
            )
        layers.append(candidates)
        reply = yield from issue("reasoning", relation_chains=candidates)
        if reply is None:
            return None
        stop, answer = parse_reasoning(reply)
        if stop:
            return (
                "{" + answer + "}"
                if answer
                else (
                    yield from issue(
                        "answer",
                        relation_chains=candidates,
                    )
                )
            )
        # Candidate entities belong to paths: repeated endpoints must retain their
        # different relation chains, as in the paper's two Australia candidates.
        kept = rng.sample(candidates, min(config.width, len(candidates)))
        trace.append(
            {
                "event": "random_prune",
                "candidates": len(candidates),
                "kept": [row["entity"] for row in kept],
            }
        )
        frontier = kept
    trace.append({"event": "depth_exhausted", "fallback": "released_cot"})
    return (yield from issue("fallback"))
