"""Iterative retrieval: multi-hop retrieval over the KG store, one entity at a time.

A multi-hop retrieval baseline on the same KG as path mode, in the style of IRCoT and
self-ask: one-hop retrieval (xgraph.rag) cannot list a multi-hop chain, so the model may
ask for more. Round 0 lists the one-hop facts of the request's topic entities (`topics`,
QIDs: the entities its gold query names, or with `--linked`, those linked in its text,
xgraph.linking), outgoing then incoming, each ordered by relation label and capped at
xgraph.rag.CAP, exactly as `python -m xgraph.rag --linked` lists them, before the
question. The model then either answers (`Answer: ⟦ label⟧`) or ends its reply with the
line `Next: ⟦ label⟧`, naming the entity whose facts it needs next; the reply is cut
after that line. The arm looks the label up in the KG store (`Facts.senses`: the items
whose canonical label equals it, NFC and case folded, else those with it as an alias;
best-connected first, at most SENSES of them), lists their one-hop facts the same way,
sense after sense, at most CAP in all, as `Facts from Wikidata about <label>:`, and
generates again. After max_lookups lookups (`--max-calls`, default 4) the last facts
message also asks for the final answer (LAST), and the reply after it is final. A reply
with neither line ends the request. Facts that would make the next prompt longer than
max_prompt_tokens are left out, last first; when not one fits, the request ends.

With a written start (`--written-start`), a short generation comes first: asked for the
entity the question starts from (start_messages: START_INSTRUCTIONS and the
demonstrations' first entities, at most START_TOKENS), the model writes `Start: ⟦
label⟧`, and round 0 lists the facts of that label's senses (as a lookup lists them),
then the topic entities' facts not yet listed, at most CAP in all; a reply naming no
label leaves the topics' facts alone. The start round counts as a round, its tokens
towards the record's, and the record keeps the label (`start`, null where none was
written).

A request naming a KG version (`kg_version`) reads every fact through it
(toolkg.version_stores, as the walk does): each edited (subject, relation) pair's objects
are the version's. Records keep the replies, the labels looked up, the rounds (generation
calls), the prompt and generated tokens summed over rounds, and whether the lookups ran
out (`budget_spent`: the last lookup was made and LAST asked for the answer). Raw
turns and per-call usage, including the written start before reply clipping, support
controlled serving replay (xgraph.serve_cost).

The demonstrations (DEMOS) are path mode's examples (prompts.PATH_EXAMPLES): the same
questions with as many hops, each hop a round, after the facts of xgraph.rag.DEMO_FACTS;
neither they nor the instructions mention escapes, as path mode's prompt does not.

    python -m xgraph.iterative requests --test target/mintqa/questions.json \\
        --kg data/external/mintqa-full/kg --linked \\
        --out target/mintqa/requests-iterative-linked.jsonl
    XG_TOOLS_DB=data/external/mintqa-full/kg.sqlite ./scripts/generate.sh <model> \\
        target/mintqa/requests-iterative-linked.jsonl iterative <out.jsonl> --max-calls 4
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

from xgraph.prompts import PATH_EXAMPLES
from xgraph.rag import CAP, DEMO_FACTS, facts_block
from xgraph.score import normalize
from xgraph.toolkg import KGStore

SENSES = 3
OVERHEAD = 16
"""Prompt tokens allowed per added message for the chat template's markup."""

INSTRUCTIONS = """\
Answer the question from facts of Wikidata with a short rationale, then the answer on a \
last line of its own.

Write every entity as a span with its English Wikidata label, like ⟦ Marie Curie⟧, \
and every fact you use as a span of subject, Wikidata property, and object, like \
⟦ Marie Curie · employer · University of Paris⟧. Use the exact labels Wikidata uses. \
When the question takes several steps, take them one at a time: if the facts listed do \
not answer the question yet, end with the line "Next: ⟦ label⟧", naming the entity whose \
facts you need next, and its facts follow.

End with the line "Answer: ⟦ label⟧", naming one entity."""

LAST = (
    'That was the last lookup. Give your final answer now, ending with the line "Answer: ⟦ label⟧".'
)

DEMOS = [
    {
        "question": PATH_EXAMPLES[0]["question"],
        "steps": [
            ("Steve Jobs", "⟦ Steve Jobs · educated at · Reed College⟧.\nAnswer: ⟦ Reed College⟧"),
        ],
    },
    {
        "question": PATH_EXAMPLES[1]["question"],
        "steps": [
            ("Mount Kilimanjaro", "⟦ Mount Kilimanjaro · country · Tanzania⟧.\nNext: ⟦ Tanzania⟧"),
            ("Tanzania", "⟦ Tanzania · continent · Africa⟧.\nAnswer: ⟦ Africa⟧"),
        ],
    },
    {
        "question": PATH_EXAMPLES[2]["question"],
        "steps": [
            (
                "The Hobbit",
                "⟦ The Hobbit · author · J. R. R. Tolkien⟧.\nNext: ⟦ J. R. R. Tolkien⟧",
            ),
            (
                "J. R. R. Tolkien",
                "⟦ J. R. R. Tolkien · place of birth · Bloemfontein⟧.\nNext: ⟦ Bloemfontein⟧",
            ),
            ("Bloemfontein", "⟦ Bloemfontein · country · South Africa⟧.\nAnswer: ⟦ South Africa⟧"),
        ],
    },
    {
        "question": PATH_EXAMPLES[3]["question"],
        "steps": [
            (
                "Harry Potter and the Philosopher's Stone",
                "⟦ Harry Potter and the Philosopher's Stone · author · J. K. Rowling⟧.\n"
                "Next: ⟦ J. K. Rowling⟧",
            ),
            (
                "J. K. Rowling",
                "⟦ J. K. Rowling · country of citizenship · United Kingdom⟧.\n"
                "Next: ⟦ United Kingdom⟧",
            ),
            ("United Kingdom", "⟦ United Kingdom · capital · London⟧.\nAnswer: ⟦ London⟧"),
        ],
    },
]
"""Path mode's examples, a round per hop: the entity whose facts each round lists (round
0's as the question's one-hop facts), and the reply."""

START_INSTRUCTIONS = """\
Do not answer the question. Name the entity the question starts from, whose facts are \
needed first, as a span with its English Wikidata label, on one line: "Start: ⟦ label⟧". \
Use the exact label Wikidata uses."""

START_TOKENS = 32
"""The most tokens the start round generates."""

_LINE = re.compile(r"^(Answer|Next):", re.MULTILINE)
_SPAN = re.compile(r"⟦([^⟧]*)⟧")

Triple = tuple[str, str, str]


def facts_about(label: str, triples: list[Triple]) -> str:
    lines = "\n".join(f"- {s} · {r} · {o}" for s, r, o in triples) or "(none)"
    return f"Facts from Wikidata about {label}:\n{lines}"


def iterative_messages(question: str, triples: list[Triple]) -> list[dict]:
    """The instructions, the demonstrations round by round, then round 0: the facts and
    the question."""
    messages = [{"role": "system", "content": INSTRUCTIONS}]
    for demo in DEMOS:
        for k, (entity, reply) in enumerate(demo["steps"]):
            shown = (
                f"{facts_block(DEMO_FACTS[entity])}\n\nQuestion: {demo['question']}"
                if k == 0
                else facts_about(entity, DEMO_FACTS[entity])
            )
            messages.append({"role": "user", "content": shown})
            messages.append({"role": "assistant", "content": reply})
    messages.append({"role": "user", "content": f"{facts_block(triples)}\n\nQuestion: {question}"})
    return messages


def start_messages(question: str) -> list[dict]:
    """The start round's prompt: the instructions, each demonstration's question with the
    entity its first round lists, then the question."""
    messages = [{"role": "system", "content": START_INSTRUCTIONS}]
    for demo in DEMOS:
        messages.append({"role": "user", "content": demo["question"]})
        messages.append({"role": "assistant", "content": f"Start: ⟦ {demo['steps'][0][0]}⟧"})
    messages.append({"role": "user", "content": question})
    return messages


def start_label(text: str) -> str | None:
    """The label a start reply names: its first span's, before any " · " (a path's
    start); None where it writes none, or an escaped one."""
    found = _SPAN.search(text)
    if found is None:
        return None
    label = found.group(1).split("·", 1)[0].strip()
    return label if label and not label.startswith("⁇") else None


def parse(text: str) -> tuple[str | None, str]:
    """What a reply asks for, by its first Answer or Next line ("answer", "next", or
    None), and the reply kept: cut after a Next line, else whole."""
    found = _LINE.search(text)
    if found is None:
        return None, text
    if found.group(1) == "Answer":
        return "answer", text
    end = text.find("\n", found.end())
    return "next", text if end < 0 else text[:end]


def next_label(kept: str) -> str:
    line = kept.rsplit("\n", 1)[-1]
    return line.split(":", 1)[1].strip().removeprefix("⟦").removesuffix("⟧").strip()


class Facts:
    """One-hop facts from a KG store, read as it is or through a KG version (a
    toolkg.VersionedStore over the same store)."""

    def __init__(self, db: str | Path, version: Any = None) -> None:
        self.read = version if version is not None else KGStore(db)
        edited = version.edited if version is not None else {}
        self._con = self.read.store._con if version is not None else self.read._con
        self._relations = dict(self._con.execute("SELECT prop, label FROM relations"))
        self._edited = {(int(s), int(p)): [int(o) for o in objs] for (s, p), objs in edited.items()}
        self._labels: dict[int, str | None] = {}

    def _label(self, item: int) -> str | None:
        if item not in self._labels:
            self._labels[item] = self.read.label(item)
        return self._labels[item]

    def senses(self, name: str, limit: int = SENSES) -> list[int]:
        """The items a name looks up: those whose canonical label it is, else those with
        it as an alias; best-connected first."""
        named = self.read.items_named(name)
        labelled = [i for i in named if normalize(self._label(i) or "") == normalize(name)]
        return sorted(labelled or named, key=lambda i: (-self.read.degree(i), i))[:limit]

    def lookup(self, name: str, cap: int = CAP) -> list[Triple]:
        """The one-hop facts of the name's senses, sense after sense, each once."""
        listed: dict[Triple, None] = {}
        for item in self.senses(name):
            listed.update(dict.fromkeys(self.one_hop([item], cap)))
        return list(listed)[:cap]

    def one_hop(self, items: list[int], cap: int = CAP) -> list[Triple]:
        """As xgraph.rag.one_hop lists the items' facts: outgoing, then incoming from
        other subjects, each by (relation, subject, object) label, at most cap."""
        mine = sorted({int(i) for i in items})
        outgoing: set[Triple] = set()
        for s in mine:
            rows = self._con.execute("SELECT p, o FROM triples WHERE s = ?", (s,)).fetchall()
            rows = [(p, o) for p, o in rows if (s, p) not in self._edited]
            rows += [(p, o) for (s_, p), objs in self._edited.items() if s_ == s for o in objs]
            for p, o in rows:
                triple = (self._label(s), self._relations.get(p), self._label(o))
                if all(triple):
                    outgoing.add(triple)  # type: ignore[arg-type]
        listed = sorted(outgoing, key=lambda t: (t[1], t[0], t[2]))[:cap]
        if len(listed) < cap:
            listed += self._incoming(mine, cap - len(listed))
        return listed

    def _incoming(self, mine: list[int], room: int) -> list[Triple]:
        """The first `room` incoming facts in label order; SQL sorts by label (its
        binary collation orders UTF-8 as Python orders strings), so that a hub's many
        incoming facts need not all be read."""
        marks = ",".join("?" * len(mine))
        props = {
            p
            for (p,) in self._con.execute(
                f"SELECT DISTINCT p FROM triples WHERE o IN ({marks})", mine
            ).fetchall()
        }
        added = [
            (s, p, o)
            for (s, p), objs in self._edited.items()
            for o in objs
            if o in mine and s not in mine
        ]
        props |= {p for _, p, _ in added}
        touched = {p for _, p, _ in added} | {
            p
            for s, p in self._edited
            if p in props
            and s not in mine
            and self._con.execute(
                f"SELECT 1 FROM triples WHERE s = ? AND p = ? AND o IN ({marks}) LIMIT 1",
                (s, p, *mine),
            ).fetchone()
        }  # relations whose incoming facts the version changes
        groups: dict[str, list[int]] = {}
        for p in props:
            if (label := self._relations.get(p)) is not None:
                groups.setdefault(label, []).append(p)
        listed: list[Triple] = []
        for relation in sorted(groups):
            if len(listed) >= room:
                break
            group = groups[relation]
            pmarks = ",".join("?" * len(group))
            where = (
                f"FROM triples t JOIN labels ls ON ls.item = t.s JOIN labels lo ON lo.item = t.o "
                f"WHERE t.o IN ({marks}) AND t.p IN ({pmarks}) AND t.s NOT IN ({marks})"
            )
            if touched.isdisjoint(group):
                rows = self._con.execute(
                    f"SELECT DISTINCT ls.label, lo.label {where} ORDER BY 1, 2 LIMIT ?",
                    (*mine, *group, *mine, room - len(listed)),
                ).fetchall()
                listed += [(s, relation, o) for s, o in rows]
                continue
            rows = self._con.execute(
                f"SELECT t.s, t.p, ls.label, lo.label {where}", (*mine, *group, *mine)
            ).fetchall()
            found = {(sl, ol) for s, p, sl, ol in rows if (s, p) not in self._edited}
            for s, p, o in added:
                if p in group and self._label(s) and self._label(o):
                    found.add((self._label(s), self._label(o)))  # type: ignore[arg-type]
            listed += [(s, relation, o) for s, o in sorted(found)][: room - len(listed)]
        return listed


def iterate(
    llm: Any,
    requests: list[dict],
    db: str | Path,
    params: Any,
    max_lookups: int = 4,
    versions: dict | None = None,
    cap: int = CAP,
    max_prompt_tokens: int | None = None,
    written_start: bool = False,
    start_params: Any = None,
    *,
    facts_factory: Any = Facts,
) -> list[dict]:
    """One record per request, all requests of a round to the model at once (see the
    module docstring). versions maps KG version names to toolkg.version_stores' stores.
    With written_start, a start round (with start_params, else params) comes first."""
    sources: dict[str | None, Facts] = {}
    from xgraph.prompts import SET_FORMAT, SET_INSTRUCTION, set_answer_fields

    states = []
    for request in requests:
        name = request.get("kg_version")
        if name is not None and name not in (versions or {}):
            raise ValueError(f"request {request['id']!r} reads KG version {name!r}, not given")
        if name not in sources:
            sources[name] = facts_factory(db, None if name is None else versions[name])  # type: ignore[index]
        states.append({"id": request["id"], "request": request, "facts": sources[name],
                       "lookups": [], "cap_events": [],
                       "replies": [], "turns": [], "rounds": 0, "prompt": 0, "generated": 0,
                       "done": False, "finish_reason": None, "start": None})  # fmt: skip
    if written_start:
        outputs = llm.chat(
            [start_messages(r["messages"][-1]["content"]) for r in requests],
            params if start_params is None else start_params,
            chat_template_kwargs={"enable_thinking": False},
            use_tqdm=False,
        )
        for state, output in zip(states, outputs, strict=True):
            completion = output.outputs[0]
            state["rounds"] += 1
            state["prompt"] += len(output.prompt_token_ids)
            state["generated"] += len(completion.token_ids)
            state["turns"].append(
                {
                    "stage": "start",
                    "text": completion.text,
                    "prompt_tokens": len(output.prompt_token_ids),
                    "generated_tokens": len(completion.token_ids),
                    "finish_reason": completion.finish_reason,
                }
            )
            state["start"] = start_label(completion.text)
    for state, request in zip(states, requests, strict=True):
        *before, asked = request["messages"]
        if request.get("answer_mode") == "set":
            before = [
                {
                    **m,
                    "content": m["content"].replace(
                        'End with the line "Answer: ⟦ label⟧", naming one entity.', ""
                    )
                    + ("\n" + SET_INSTRUCTION + "\n" + SET_FORMAT if m["role"] == "system" else ""),
                }
                for m in before
            ]
            if not any(m["role"] == "system" for m in before):
                before.insert(0, {"role": "system", "content": SET_INSTRUCTION + "\n" + SET_FORMAT})
        topics = [int(t[1:]) for t in request.get("topics") or []]
        listed: dict[Triple, None] = {}
        probe_cap = cap + int(request.get("answer_mode") == "set")
        if state["start"]:
            listed.update(dict.fromkeys(state["facts"].lookup(state["start"], probe_cap)))
        listed.update(dict.fromkeys(state["facts"].one_hop(topics, probe_cap)))
        if len(listed) > cap:
            state["cap_events"].append(
                {"stage": "initial_facts", "observed_at_least": len(listed), "kept": cap}
            )
        first = f"{facts_block(list(listed)[:cap])}\n\nQuestion: {asked['content']}"
        state["messages"] = [*before, {"role": "user", "content": first}]
    encode = None
    if max_prompt_tokens is not None:
        tokenizer = llm.get_tokenizer()

        def encode(text: str) -> int:
            return len(tokenizer.encode(text, add_special_tokens=False))

    while pending := [s for s in states if not s["done"]]:
        outputs = llm.chat(
            [s["messages"] for s in pending],
            params,
            chat_template_kwargs={"enable_thinking": False},
            use_tqdm=False,
        )
        for state, output in zip(pending, outputs, strict=True):
            completion = output.outputs[0]
            state["rounds"] += 1
            state["prompt"] += len(output.prompt_token_ids)
            state["generated"] += len(completion.token_ids)
            state["finish_reason"] = completion.finish_reason
            state["turns"].append(
                {
                    "stage": "retrieve",
                    "text": completion.text,
                    "prompt_tokens": len(output.prompt_token_ids),
                    "generated_tokens": len(completion.token_ids),
                    "finish_reason": completion.finish_reason,
                }
            )
            kind, kept = parse(completion.text)
            state["replies"].append(kept)
            label = next_label(kept) if kind == "next" else ""
            if not label or len(state["lookups"]) >= max_lookups:
                state["done"] = True
                continue
            state["lookups"].append(label)
            probe_cap = cap + int(state["request"].get("answer_mode") == "set")
            facts = state["facts"].lookup(label, probe_cap)
            if len(facts) > cap:
                state["cap_events"].append(
                    {
                        "stage": "lookup_facts",
                        "observed_at_least": len(facts),
                        "kept": cap,
                        "lookup": label,
                    }
                )
            lines = [f"- {s} · {r} · {o}" for s, r, o in facts[:cap]]
            header = f"Facts from Wikidata about {label}:"
            final = (
                SET_INSTRUCTION + "\n" + SET_FORMAT
                if state["request"].get("answer_mode") == "set"
                else LAST
            )
            tail = f"\n\n{final}" if len(state["lookups"]) == max_lookups else ""
            if encode is not None and lines:
                room = max_prompt_tokens - (  # type: ignore[operator]
                    len(output.prompt_token_ids) + len(completion.token_ids) + 2 * OVERHEAD
                    + encode(header) + encode(tail)
                )  # fmt: skip
                fitting = []
                for line in lines:
                    room -= encode("\n" + line)
                    if room < 0:
                        break
                    fitting.append(line)
                if len(fitting) < len(lines):
                    state["cap_events"].append(
                        {
                            "stage": "prompt_facts",
                            "before": len(lines),
                            "kept": len(fitting),
                            "lookup": label,
                        }
                    )
                if not fitting:
                    state["done"], state["finish_reason"] = True, "length"
                    continue
                lines = fitting
            content = header + "\n" + ("\n".join(lines) or "(none)") + tail
            state["messages"] = [
                *state["messages"],
                {"role": "assistant", "content": kept},
                {"role": "user", "content": content},
            ]
    return [
        {
            "id": s["id"],
            "arm": "iterative",
            "text": "\n".join(s["replies"]),
            **set_answer_fields(s["request"], s["replies"][-1]),
            **({"cap_events": s["cap_events"]} if s["request"].get("answer_mode") == "set" else {}),
            "replies": s["replies"],
            "turns": s["turns"],
            "lookups": s["lookups"],
            "budget_spent": len(s["lookups"]) >= max_lookups,
            **({"start": s["start"]} if written_start else {}),
            "rounds": s["rounds"],
            "token_ids": [],
            "prompt_tokens": s["prompt"],
            "generated_tokens": s["generated"],
            "finish_reason": s["finish_reason"],
        }
        for s in states
    ]


def build_requests(questions: list, topics: dict[str, list[int]]) -> list[dict]:
    """Chat requests ending with the bare question, which round 0 completes with the
    facts, each naming its topic entities."""
    return [
        {
            "id": q.id,
            "topics": [f"Q{n}" for n in topics.get(q.id, [])],
            "messages": [
                *iterative_messages(q.utterance, [])[:-1],
                {"role": "user", "content": q.utterance},
            ],
        }
        for q in questions
    ]


def main(argv: list[str] | None = None) -> None:
    from xgraph.wwq import NAMED, load_wwq

    parser = argparse.ArgumentParser(prog="python -m xgraph.iterative", description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    asked = commands.add_parser("requests", help="iterative retrieval's requests")
    asked.add_argument("--test", type=Path, required=True, help="WWQ-format questions (.json)")
    asked.add_argument("--kg", type=Path, required=True, help="the KG directory to link against")
    asked.add_argument("--out", type=Path, required=True, help="requests (JSONL)")
    asked.add_argument(
        "--linked",
        action="store_true",
        help="topic entities linked from the question text (xgraph.linking), not the gold query's",
    )
    args = parser.parse_args(argv)
    questions = load_wwq(args.test)
    if args.linked:
        from xgraph.linking import link

        topics = link({q.id: q.utterance for q in questions}, args.kg)
    else:
        topics = {q.id: sorted({int(n) for n in NAMED.findall(q.sparql)}) for q in questions}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    requests = build_requests(questions, topics)
    with args.out.open("w", encoding="utf-8") as out:
        for request in requests:
            out.write(json.dumps(request, ensure_ascii=False) + "\n")
    print(f"{len(requests)} requests, {sum(bool(r['topics']) for r in requests)} with topics")


if __name__ == "__main__":
    main()
