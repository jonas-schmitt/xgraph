"""Few-shot prompts that teach the span syntax, and reading the final answer back.

The quality gate's WWQ answer format (`docs/quality-gate.md`) is a short rationale
with fact spans, then the final answer as an entity span on a line of its own:

    ⟦ Steve Jobs · educated at · Reed College⟧.
    Answer: ⟦ Reed College⟧

Both arms get the same messages: instructions, worked examples from WWQ's train
split, then the question. Labels are Wikidata's English labels, so that a written
span can resolve in the KG.

The observed path variant (wwq_messages(..., path_spans=True, observe_paths=k), or
observed_requests selected by generation) retains PATH_EXAMPLES' questions, paths
and final answers, showing a frontier observation after each start and relation.
These are static demonstrations of the language; the served engine supplies the
actual, versioned and capped frontier. Custom demonstrations must use that language.

    python -m xgraph.prompts wwq --questions data/wwq/dev.json --limit 40 \\
        --out target/pilot/requests.jsonl
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from xgraph.syntax import SpanSyntax, preset

SET_INSTRUCTION = (
    "List every answer to the question; do not choose only one. "
    "Keep each entity name intact, including commas."
)
SET_FORMAT = (
    "End with Answer: followed by a JSON array of all answer labels, or one answer per line. "
    "Use an empty JSON array [] when no answer is supported. Write no text after the answer set."
)


def parse_answer_set(text: str, *, bare: bool = False) -> list[str]:
    """Read a final answer block without splitting names at commas or reading paths.

    Bare arrays/objects are explicit sets. Bare lines are accepted only at a known
    final-reader seam; rationale otherwise needs an Answer: marker.
    """
    text = text.strip()
    markers = list(re.finditer(r"^\s*(?:#\s*)?Answers?:[ \t]*", text, re.M | re.I))
    if markers:
        text = text[markers[-1].end() :].strip()
    elif not bare and not text.startswith(("[", "{", "```")):
        return []
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text).strip()
    if text.startswith(("[", "{")):
        try:
            value = json.loads(text)
        except json.JSONDecodeError:
            return []
        if isinstance(value, dict):
            value = value.get("answers", value.get("answer", value.get("label", [])))
        values = [value] if isinstance(value, str) else value
        if not isinstance(values, list) or any(not isinstance(v, str) for v in values):
            return []
    else:
        values = []
        for line in text.splitlines():
            line = line.strip()
            if line.startswith(("Action:", "Next:", "Path:", "Rationale:")):
                break
            spans = re.findall(r"⟦\s*([^⟧]*)⟧", line)
            values.extend(spans or [re.sub(r"^(?:[-*]\s+|\d+[.)]\s+)", "", line)])
    return list(
        dict.fromkeys(
            v.strip()
            for v in values
            if v.strip() and v.strip().casefold() != "unknown" and not v.strip().startswith("⁇")
        )
    )


def set_answer_fields(request: dict, text: str) -> dict:
    """Opt-in complete-set delivery; existing singleton protocols stay unchanged."""
    return (
        {"answer_mode": "set", "answers": parse_answer_set(text, bare=True)}
        if request.get("answer_mode") == "set"
        else {}
    )


INSTRUCTIONS = """\
Answer the question with a short rationale, then the answer on a last line of its own.

Write every entity as a span with its English Wikidata label, like ⟦ Marie Curie⟧, \
and every fact you use as a span of subject, Wikidata property, and object, like \
⟦ Marie Curie · employer · University of Paris⟧. Use the exact labels Wikidata uses. \
If you are not sure of a label or a fact, escape it inside the span, like \
⟦ ⁇ an unlisted laboratory⟧, rather than guessing.

End with the line "Answer: ⟦ label⟧", naming one entity."""

WWQ_EXAMPLES = [
    {
        "id": "WebQTrn-2879",
        "question": "what college did steve jobs attend?",
        "response": "⟦ Steve Jobs · educated at · Reed College⟧.\nAnswer: ⟦ Reed College⟧",
        "answer": "Reed College",
    },
    {
        "id": "WebQTrn-1259",
        "question": "where did brad paisley graduate from college?",
        "response": (
            "⟦ Brad Paisley · educated at · Belmont University⟧.\nAnswer: ⟦ Belmont University⟧"
        ),
        "answer": "Belmont University",
    },
    {
        "id": "WebQTrn-1240",
        "question": "where is mallorca?",
        "response": (
            "⟦ Mallorca⟧ is one of the ⟦ Balearic Islands⟧: "
            "⟦ Mallorca · country · Spain⟧.\nAnswer: ⟦ Spain⟧"
        ),
        "answer": "Spain",
    },
    {
        "id": "WebQTrn-2501",
        "question": "who plays lois in family guy?",
        "response": (
            "⟦ Family Guy · characters · Lois Griffin⟧ and "
            "⟦ Lois Griffin · voice actor · Alex Borstein⟧, a role she has had "
            "⟦ ⁇ since the first episode⟧.\nAnswer: ⟦ Alex Borstein⟧"
        ),
        "answer": "Alex Borstein",
    },
]

CHAIN_INSTRUCTIONS = INSTRUCTIONS.replace(
    "\n\nEnd with",
    " When the question takes several steps, write each step as a fact span whose subject "
    "is the object of the step before, and answer with the object of the last step. The "
    "facts in your spans come from the knowledge graph: where one differs from what you "
    "remember, follow the span.\n\nEnd with",
)
"""The chain prompt: multi-hop answers written as a chain of fact spans."""

CHAIN_EXAMPLES = [
    WWQ_EXAMPLES[0],
    {
        "id": "chain-1",
        "question": "On which continent is the country where Mount Kilimanjaro is?",
        "response": (
            "⟦ Mount Kilimanjaro · country · Tanzania⟧, and "
            "⟦ Tanzania · continent · Africa⟧.\nAnswer: ⟦ Africa⟧"
        ),
        "answer": "Africa",
    },
    {
        "id": "chain-2",
        "question": "In which country was the author of The Hobbit born?",
        "response": (
            "⟦ The Hobbit · author · J. R. R. Tolkien⟧, "
            "⟦ J. R. R. Tolkien · place of birth · Bloemfontein⟧, and "
            "⟦ Bloemfontein · country · South Africa⟧.\nAnswer: ⟦ South Africa⟧"
        ),
        "answer": "South Africa",
    },
    {
        "id": "chain-3",
        "question": (
            "What is the capital of the country whose citizen wrote "
            "Harry Potter and the Philosopher's Stone?"
        ),
        "response": (
            "⟦ Harry Potter and the Philosopher's Stone · author · J. K. Rowling⟧, "
            "⟦ J. K. Rowling · country of citizenship · United Kingdom⟧, and "
            "⟦ United Kingdom · capital · London⟧.\nAnswer: ⟦ London⟧"
        ),
        "answer": "London",
    },
]

PATH_INSTRUCTIONS = """\
Answer the question by the path through Wikidata that leads to its answer, then the \
answer on a last line of its own.

Write the path as one span: the entity the question starts from, with its English \
Wikidata label, then the Wikidata properties to follow from it, in order, relations only \
and no entities in between, like ⟦ The Hobbit · author · place of birth⟧. The knowledge \
graph follows the path for you.

End with the line "Answer: ⟦ label⟧", naming the entity the path reaches."""
"""The path-span prompt (path mode): the KG walks the path the model writes."""

PATH_EXAMPLES = [
    {
        "id": "path-0",
        "question": "what college did steve jobs attend?",
        "response": "⟦ Steve Jobs · educated at⟧.\nAnswer: ⟦ Reed College⟧",
        "answer": "Reed College",
    },
    {
        "id": "path-1",
        "question": "On which continent is the country where Mount Kilimanjaro is?",
        "response": "⟦ Mount Kilimanjaro · country · continent⟧.\nAnswer: ⟦ Africa⟧",
        "answer": "Africa",
    },
    {
        "id": "path-2",
        "question": "In which country was the author of The Hobbit born?",
        "response": ("⟦ The Hobbit · author · place of birth · country⟧.\nAnswer: ⟦ South Africa⟧"),
        "answer": "South Africa",
    },
    {
        "id": "path-3",
        "question": (
            "What is the capital of the country whose citizen wrote "
            "Harry Potter and the Philosopher's Stone?"
        ),
        "response": (
            "⟦ Harry Potter and the Philosopher's Stone · author · country of citizenship · "
            "capital⟧.\nAnswer: ⟦ London⟧"
        ),
        "answer": "London",
    },
]

MOUNTAIN_EXAMPLE = {
    "question": "What mountain range contains the highest point of Germany?",
    "response": "⟦ Germany · highest point · mountain range⟧.\nAnswer: ⟦ Wetterstein⟧",
    "answer": "Wetterstein",
}


def path_demo_requests(requests, demos: int, syntax: SpanSyntax):
    """Choose four or five maintained path questions; custom examples stay caller-owned."""
    if demos not in (4, 5):
        raise ValueError("path_demos must be 4 or 5")
    converted = []
    for request in requests:
        messages = list(request["messages"])
        last_user = max(
            (i for i, m in enumerate(messages) if m["role"] == "user"), default=len(messages)
        )
        at = next(
            (
                i
                for i, m in enumerate(messages[:last_user])
                if m["role"] == "user" and m["content"] == MOUNTAIN_EXAMPLE["question"]
            ),
            None,
        )
        if demos == 4 and at is not None and at + 1 < len(messages):
            if messages[at + 1]["role"] == "assistant":
                del messages[at : at + 2]
        elif demos == 5 and at is None:
            questions = {m["content"] for m in messages if m["role"] == "user"}
            if all(e["question"] in questions for e in PATH_EXAMPLES):
                at = max(i for i, m in enumerate(messages) if m["role"] == "user")
                messages[at:at] = [
                    {"role": "user", "content": MOUNTAIN_EXAMPLE["question"]},
                    {"role": "assistant", "content": restyle(MOUNTAIN_EXAMPLE["response"], syntax)},
                ]
        converted.append({**request, "messages": messages})
    return converted


OBSERVATION_INSTRUCTIONS = """The knowledge graph writes an observation after the start and
each relation: " ·→" followed by up to {k} reached entity labels, joined by " ;".
"(+N more)" counts additional entities in the capped frontier; "∅" means empty.
Read these observations to decide which relation the question still needs, or
whether to close the path and answer. All reached entities remain available even
when only some labels are shown. The answer is still an entity span."""


def audit_requests(requests, syntax: SpanSyntax, *, entities=20, menus=8):
    """Keep the maintained questions, teaching delayed observation and one decision.

    These are teaching examples, like the observed demonstrations; their displayed
    menus do not claim to enumerate a particular served KG version.
    """
    instruction = (
        f"Write a blind relation path as usual, with no observations inside it. After its closer, "
        f"the graph writes Executed with up to {entities} endpoint labels, including singletons. "
        "(+N more) counts omitted entities of the capped frontier. "
        "Every reached label remains selectable. "
        + (
            f"Relations (terminal) and Relations (penultimate) show up to {menus} "
            "available relations "
            "from the endpoints and the frontier before the last relation; menus are hints, and "
            "every executable relation remains available. "
            if menus
            else ""
        )
        + f"After Decision: {syntax.opener}, write exactly one of KEEP{syntax.closer}, "
        f"SELECT {syntax.opener}={syntax.label_prefix}label{syntax.closer}, "
        f"APPEND{syntax.separator}{syntax.label_prefix}relation{syntax.closer}, "
        f"REPLACE{syntax.separator}{syntax.label_prefix}relation{syntax.closer}, "
        f"or DROP{syntax.closer}. "
        "KEEP preserves the path. SELECT chooses a terminal label, including an undisplayed one. "
        "APPEND adds one terminal relation; REPLACE replaces only the last relation; DROP removes "
        "the last relation only when at least two were written. "
        "At the hop cap APPEND is unavailable. "
        "The graph executes that decision and displays edited endpoints, "
        "then writes the Answer prefix. "
        "Answer once inside its bound entity span. There is no further planning or second decision."
    )

    def span(body):
        return syntax.opener + syntax.label_prefix + body + syntax.closer

    def path(start, relations):
        return span(start + "".join(syntax.separator + syntax.label_prefix + r for r in relations))

    def display(text, limit):
        labels = text.split(" ; ")
        body = " ;".join(syntax.label_prefix + label for label in labels[:limit])
        if len(labels) > limit:
            body += f" (+{len(labels) - limit} more)"
        return syntax.opener + body + syntax.closer

    def example(start, relations, endpoints, terminal, penultimate, decision, answer, edited=None):
        text = path(start, relations) + "\nExecuted: " + display(endpoints, entities)
        if menus:
            text += "\nRelations (terminal): " + display(terminal, menus)
            text += "\nRelations (penultimate): " + display(penultimate, menus)
        text += "\nDecision: " + syntax.opener + decision + syntax.closer
        if edited is not None:
            text += "\nExecuted: " + display(edited, entities)
        return text + ".\nAnswer: " + span(answer)

    sep = syntax.separator + syntax.label_prefix
    examples = [
        example(
            "Steve Jobs",
            ["educated at"],
            "Reed College ; Homestead High School",
            "∅",
            "educated at",
            "SELECT " + syntax.opener + "=" + syntax.label_prefix + "Reed College",
            "Reed College",
        ),
        example(
            "Mount Kilimanjaro",
            ["country"],
            "Tanzania",
            "continent ; capital",
            "country",
            "APPEND" + sep + "continent",
            "Africa",
            "Africa",
        ),
        example(
            "The Hobbit",
            ["author", "place of birth", "continent"],
            "Africa",
            "∅",
            "country ; continent",
            "REPLACE" + sep + "country",
            "South Africa",
            "South Africa",
        ),
        example(
            "Harry Potter and the Philosopher's Stone",
            ["author", "country of citizenship", "capital"],
            "London",
            "country",
            "capital",
            "KEEP",
            "London",
        ),
    ]
    menu_question = "What mountain range contains the highest point of Germany?"
    menu_example = example(
        "Germany",
        ["highest point", "located in the administrative territorial entity"],
        "Garmisch-Partenkirchen",
        "country",
        "mountain range ; country",
        "REPLACE" + sep + "mountain range",
        "Wetterstein",
        "Wetterstein",
    )
    replacements = {
        restyle(old["response"], syntax): new
        for old_examples in (PATH_EXAMPLES, OBSERVED_PATH_EXAMPLES)
        for old, new in zip(old_examples, examples, strict=True)
    }
    converted = []
    for request in requests:
        messages, replaced = [], 0
        for n, message in enumerate(request["messages"]):
            content = message["content"]
            if message["role"] == "system":
                if instruction not in content:
                    content += "\n\n" + instruction
            elif (
                message["role"] == "assistant"
                and not (request.get("prefill_replay") and n == len(request["messages"]) - 1)
                and content in replacements
            ):
                content, replaced = replacements[content], replaced + 1
            messages.append({**message, "content": content})
        if not any(m["role"] == "system" for m in messages):
            messages.insert(0, {"role": "system", "content": instruction})
        if replaced == 4:
            at = max(i for i, m in enumerate(messages) if m["role"] == "user")
            messages[at:at] = [
                {"role": "user", "content": menu_question},
                {"role": "assistant", "content": menu_example},
            ]
        converted.append({**request, "messages": messages})
    return converted


# Same questions, starts, relations and answers as PATH_EXAMPLES. These are teaching
# examples, not claimed enumerations of any particular version of a served KG.
def menu_requests(
    requests, k: int, syntax: SpanSyntax, *, observe=0, select=False, demos=None, width=0
):
    """Short suggestions in the maintained demonstrations; custom demos stay caller-owned."""
    import re

    from xgraph.menus import menu_marker
    from xgraph.observations import observation_marker, selection_marker

    marker, selection = observation_marker(syntax), selection_marker(syntax)
    instruction = (
        f'The graph writes a relation menu "{menu_marker(syntax)}" after each start, '
        f'relation and selection, ending with "{syntax.closer}". '
        f"It suggests up to {k} available relations ranked for the question; (+N more) counts "
        "omitted relations and ∅ means none. Choose the relation the question needs, "
        "including an undisplayed one, or close the path and answer."
    )
    hints = [
        [["educated at", "occupation"], ["country"]],
        [["country", "mountain range"], ["continent", "capital"], []],
        [["author"], ["place of birth"], ["country"], []],
        [["author"], ["country of citizenship"], ["capital"], ["country"]],
    ]

    def display(labels):
        body = " ;".join(syntax.label_prefix + label for label in labels[:k])
        if not labels:
            body = syntax.label_prefix + "∅"
        if len(labels) > k:
            body += f" (+{len(labels) - k} more)"
        return menu_marker(syntax) + body + syntax.closer

    def insert(response, menus):
        body, _, rest = response.partition(syntax.closer)
        parts = re.split(
            "(" + "|".join(re.escape(d) for d in (marker, selection, syntax.separator)) + ")",
            body,
        )
        out, hop = "", 0
        for n in range(0, len(parts), 2):
            out += parts[n]
            following = parts[n + 1] if n + 1 < len(parts) else None
            if following != marker:
                out += display(menus[hop])
            if following is not None:
                out += following
                hop += following == syntax.separator
        return out + syntax.closer + rest

    plain = [
        {"messages": [{"role": "assistant", "content": restyle(e["response"], syntax)}]}
        for e in PATH_EXAMPLES
    ]
    variants = [plain]
    for choice in (False, True):
        variants.append(observed_requests(plain, observe or 3, syntax, select=choice, width=width))
    replacements = {}
    for variant in variants:
        for request, menus in zip(variant, hints, strict=True):
            response = next(m["content"] for m in request["messages"] if m["role"] == "assistant")
            replacements[response] = insert(response, menus)
    mountain_plain = restyle(MOUNTAIN_EXAMPLE["response"], syntax)
    mountain_observed = restyle(
        "⟦ Germany ·→ Germany · highest point ·→ Zugspitze · mountain range ·→ Wetterstein⟧.\n"
        "Answer: ⟦ Wetterstein⟧",
        syntax,
    )
    mountain = (
        "⟦ Germany ·→ Germany · highest point ·→ Zugspitze · mountain range ·→ Wetterstein⟧.\n"
        "Answer: ⟦ Wetterstein⟧"
        if observe
        else "⟦ Germany · highest point · mountain range⟧.\nAnswer: ⟦ Wetterstein⟧"
    )
    mountain = insert(
        restyle(mountain, syntax), [["highest point"], ["mountain range", "country"], ["country"]]
    )
    if demos == 5:
        replacements[mountain_plain] = replacements[mountain_observed] = mountain
    if demos not in (None, 4, 5):
        raise ValueError("path_demos must be 4 or 5")
    if demos == 4:
        requests = path_demo_requests(requests, demos, syntax)
    converted = []
    for request in requests:
        messages, replaced = [], 0
        for n, message in enumerate(request["messages"]):
            content = message["content"]
            if message["role"] == "system" and instruction not in content:
                content += "\n\n" + instruction
            elif (
                message["role"] == "assistant"
                and not (request.get("prefill_replay") and n == len(request["messages"]) - 1)
                and content in replacements
            ):
                content, replaced = replacements[content], replaced + 1
            messages.append({**message, "content": content})
        if not any(m["role"] == "system" for m in messages):
            messages.insert(0, {"role": "system", "content": instruction})
        if replaced == 4 and demos != 4:
            at = max(i for i, m in enumerate(messages) if m["role"] == "user")
            messages[at:at] = [
                {
                    "role": "user",
                    "content": "What mountain range contains the highest point of Germany?",
                },
                {"role": "assistant", "content": mountain},
            ]
        converted.append({**request, "messages": messages})
    return converted


OBSERVED_PATH_EXAMPLES = [
    {**example, "response": response}
    for example, response in zip(
        PATH_EXAMPLES,
        (
            "⟦ Steve Jobs ·→ Steve Jobs · educated at ·→ Reed College⟧.\nAnswer: ⟦ Reed College⟧",
            "⟦ Mount Kilimanjaro ·→ Mount Kilimanjaro · country ·→ Tanzania"
            " · continent ·→ Africa⟧.\nAnswer: ⟦ Africa⟧",
            "⟦ The Hobbit ·→ The Hobbit · author ·→ J. R. R. Tolkien"
            " · place of birth ·→ Bloemfontein · country ·→ South Africa⟧.\n"
            "Answer: ⟦ South Africa⟧",
            "⟦ Harry Potter and the Philosopher's Stone ·→ Harry Potter and the Philosopher's Stone"
            " · author ·→ J. K. Rowling · country of citizenship ·→ United Kingdom"
            " · capital ·→ London⟧.\nAnswer: ⟦ London⟧",
        ),
        strict=True,
    )
]


def observed_requests(
    requests: list[dict],
    k: int,
    syntax: SpanSyntax,
    *,
    select: bool = False,
    width: int = 0,
    demos: int | None = None,
) -> list[dict]:
    """Select the observed variant of the existing path demonstrations automatically.

    Preserve questions, request metadata and replayed prefills. Custom demonstrations
    are the caller's responsibility; the four maintained path examples are replaced.
    A request with fixed_path_demonstrations keeps its supplied assistant examples,
    allowing a development comparison to hold demonstration bytes constant.
    """
    examples = OBSERVED_PATH_EXAMPLES
    if select:
        display = (
            "Reed College (+1 more)"
            if k == 1 and width < 2
            else "Reed College ; Homestead High School"
        )
        examples = [
            {
                **examples[0],
                "response": (
                    "⟦ Steve Jobs ·→ Steve Jobs · educated at ·→ "
                    + display
                    + " ⟦= Reed College ·→ Reed College⟧.\nAnswer: ⟦ Reed College⟧"
                ),
            },
            *examples[1:],
        ]
    replacements = {
        restyle(a["response"], syntax): restyle(b["response"], syntax)
        for a, b in zip(PATH_EXAMPLES, examples, strict=True)
    }
    if demos == 5:
        replacements[restyle(MOUNTAIN_EXAMPLE["response"], syntax)] = restyle(
            "⟦ Germany ·→ Germany · highest point ·→ Zugspitze · mountain range ·→ Wetterstein⟧.\n"
            "Answer: ⟦ Wetterstein⟧",
            syntax,
        )
    if select:
        replacements.update(
            {
                restyle(a["response"], syntax): restyle(b["response"], syntax)
                for a, b in zip(OBSERVED_PATH_EXAMPLES, examples, strict=True)
            }
        )
    from xgraph.observations import observation_marker

    instruction = OBSERVATION_INSTRUCTIONS.format(k=k).replace(
        _SYNTAX.separator + "→", observation_marker(syntax)
    )
    instructions = [instruction]
    if width:
        instructions.append(
            f"When the capped frontier has at most {width} entities, "
            "the observation shows all of them."
        )
    if select:
        from xgraph.observations import selection_marker

        instructions.append(
            f'After an observation, write "{selection_marker(syntax)}" and one reached '
            "entity label to narrow the frontier to every entity with that label, including "
            "labels not displayed. "
            "The graph writes a new observation; then continue with a relation or close the path."
        )
    converted = []
    for request in requests:
        messages = []
        for n, message in enumerate(request["messages"]):
            content = message["content"]
            if message["role"] == "system":
                for instruction in instructions:
                    if instruction not in content:
                        content += "\n\n" + instruction
            elif (
                message["role"] == "assistant"
                and not request.get("fixed_path_demonstrations")
                and not (request.get("prefill_replay") and n == len(request["messages"]) - 1)
            ):
                content = replacements.get(content, content)
            messages.append({**message, "content": content})
        if not any(m["role"] == "system" for m in messages):
            messages.insert(0, {"role": "system", "content": "\n\n".join(instructions)})
        converted.append({**request, "messages": messages})
    return converted


def decision_requests(requests, syntax: SpanSyntax, *, close_next=False, defer=0):
    """Restyle only maintained demonstrations; custom examples and prefills are caller-owned."""
    if not (close_next or defer):
        return requests
    from xgraph.menus import menu_marker, menu_ranges, semantic_text
    from xgraph.observations import (
        decision_marker,
        observation_marker,
        path_parts,
        selection_marker,
    )

    decision, selection = decision_marker(syntax), selection_marker(syntax)
    instructions = []
    if close_next:
        instructions.append(
            f'After each post-start observation and any required selection, write "{decision}" '
            f'and either "CLOSE{syntax.closer}" to answer from the current frontier or '
            f'"NEXT{syntax.closer}" to request a relation menu and continue. The graph writes '
            "the path closer and bound answer prefix after CLOSE, and the relation separator "
            "after NEXT's menu. The start menu is supplied automatically. Select before this "
            "decision when required; no menu is shown before that selection."
        )
    if defer:
        instructions.append(
            "At a required selection, instead of an entity label you may write "
            f'"DEFER{syntax.closer}" '
            f'after "{selection}", at most {defer} times per path. DEFER retains every frontier '
            "entity and consumes this boundary's selection. At the answer, choose a label from "
            "the final frontier."
        )
    converted = []
    questions = {e["question"] for e in [*PATH_EXAMPLES, MOUNTAIN_EXAMPLE]}
    for request in requests:
        messages = []
        for n, message in enumerate(request["messages"]):
            content = message["content"]
            if message["role"] == "system":
                for instruction in instructions:
                    if instruction not in content:
                        content += "\n\n" + instruction
            elif (
                message["role"] == "assistant"
                and n
                and request["messages"][n - 1]["content"] in questions
                and not (request.get("prefill_replay") and n == len(request["messages"]) - 1)
            ):
                if close_next and decision not in content:
                    for lo, hi in reversed(list(menu_ranges(content, syntax))):
                        body = semantic_text(content[:lo], syntax).split(syntax.opener, 1)[-1]
                        hop = (
                            len(path_parts(body, syntax.separator, syntax.opener, syntax.escape)[0])
                            - 1
                        )
                        following = content[hi:]
                        if following.startswith(selection):
                            replacement = ""
                        elif hop:
                            command = "NEXT" if following.startswith(syntax.separator) else "CLOSE"
                            replacement = decision + command + syntax.closer
                            if command == "NEXT":
                                replacement += content[lo:hi]
                        else:
                            replacement = content[lo:hi]
                        content = content[:lo] + replacement + content[hi:]
                if defer:
                    chosen = (
                        selection
                        + syntax.label_prefix
                        + "Reed College"
                        + observation_marker(syntax)
                        + syntax.label_prefix
                        + "Reed College"
                    )
                    if chosen in content:
                        lo = content.index(chosen)
                        hi = lo + len(chosen)
                        if content[hi:].startswith(menu_marker(syntax)):
                            hi = content.index(syntax.closer, hi) + len(syntax.closer)
                        content = content[:lo] + selection + "DEFER" + syntax.closer + content[hi:]
            messages.append({**message, "content": content})
        if not any(m["role"] == "system" for m in messages):
            messages.insert(0, {"role": "system", "content": "\n\n".join(instructions)})
        converted.append({**request, "messages": messages})
    return converted


EXPRESSION_PATH_INSTRUCTIONS = PATH_INSTRUCTIONS.replace(
    '\n\nEnd with the line "Answer: ⟦ label⟧", naming the entity the path reaches.',
    """

To compare two entities by a date, write "earlier:" or "later:", the two entities with \
their English Wikidata labels separated by " |", then the properties that lead from each \
of them to the date compared, like ⟦ earlier: Jaws | Star Wars · publication date⟧. The \
knowledge graph compares the dates for you. Born first or older means an earlier date of \
birth, younger a later one.

To ask whether two entities share a value, write "same:", the two entities, then the \
properties that lead from each of them to the values compared, like \
⟦ same: Jaws | Alien · country of origin⟧. The knowledge graph answers yes or no.

End with the line "Answer: ⟦ label⟧", naming the entity or the date the path reaches, \
the entity the comparison gives, or its yes or no.""",
)
"""The path-span prompt with expressions (path mode with XGRAPH_EXPRESSIONS=1): paths, which
may reach dates, date comparisons, whose result the answer names, and value comparisons,
answered yes or no."""

EXPRESSION_ANSWERS = {
    "When was the director of film Psycho born?": "13 August 1899",
    "Which film came out first, Jaws or Star Wars?": "Jaws",
    "Which film has the director who was born later, Alien or Vertigo?": "Alien",
    "Who died first, Charles Darwin or Abraham Lincoln?": "Abraham Lincoln",
    "Do both directors of films Psycho and Alien have the same nationality?": "yes",
}
"""The answers of xgraph.expressions' worked examples that path mode can answer: a date path,
date comparisons, and a value comparison of 2WikiMultihopQA's bridge-comparison form. No
entity answer, and no compared entity, is a 2WikiMultihopQA dev answer; the date is written
day first, as the dataset writes most dates."""

VALUE_EXAMPLES = [
    {
        "question": "Were Charles Darwin and Abraham Lincoln of the same nationality?",
        "response": "Program: ⟦ same: Charles Darwin | Abraham Lincoln · country of citizenship⟧",
        "answer": "no",
    },
]
"""A value comparison answered no, beside the reused one answered yes."""


def _expression_examples() -> list[dict]:
    from xgraph.expressions import EXPRESSION_EXAMPLES

    examples = []
    answers = {**EXPRESSION_ANSWERS, **{e["question"]: e["answer"] for e in VALUE_EXAMPLES}}
    for example in [*EXPRESSION_EXAMPLES, *VALUE_EXAMPLES]:
        answer = answers.get(example["question"])
        if answer is not None:
            program = example["response"].removeprefix("Program: ")
            examples.append(
                {
                    "id": f"expression-{len(examples)}",
                    "question": example["question"],
                    "response": f"{program}.\nAnswer: ⟦ {answer}⟧",
                    "answer": answer,
                }
            )
    return examples


BIO_INSTRUCTIONS = """\
Write a short biography of the person asked about.

Write every entity as a span with its English Wikidata label, like ⟦ Marie Curie⟧, \
and every fact you state that Wikidata could hold as a span of subject, Wikidata \
property, and object, like ⟦ Marie Curie · employer · University of Paris⟧. Use the \
exact labels Wikidata uses. If you are not sure of a label or a fact, escape it inside \
the span, like ⟦ ⁇ an unlisted laboratory⟧, rather than guessing."""

BIO_EXAMPLE = {
    "topic": "Alan Turing",
    "response": (
        "⟦ Alan Turing⟧ was an English mathematician and computer scientist, born on "
        "23 June 1912 (⟦ Alan Turing · place of birth · Maida Vale⟧). He studied at "
        "⟦ Alan Turing · educated at · King's College, Cambridge⟧ and earned his doctorate "
        "at ⟦ Alan Turing · educated at · Princeton University⟧ "
        "(⟦ Alan Turing · doctoral advisor · Alonzo Church⟧). His paper on computable "
        "numbers introduced the ⟦ Alan Turing · notable work · Turing machine⟧, a founding "
        "idea of ⟦ Alan Turing · field of work · computer science⟧. During the Second World "
        "War he broke German ciphers at ⟦ Bletchley Park⟧ for the ⟦ Government Code and "
        "Cypher School⟧. After the war he worked at "
        "⟦ Alan Turing · employer · Victoria University of Manchester⟧. He died on 7 June "
        "1954 (⟦ Alan Turing · place of death · Wilmslow⟧)."
    ),
}

_SYNTAX = SpanSyntax()
_CANONICAL_SPAN = re.compile(re.escape(_SYNTAX.opener) + "(.*?)" + re.escape(_SYNTAX.closer))

DISAMBIGUATE = (
    "If several entities share a label, write the entity's English Wikipedia article title "
    "instead, like ⟦ Andrew Young (skier)⟧, so that the span names exactly one entity."
)
"""Added with titles=True, for a KG index whose labels include Wikipedia titles."""

ANSWER_ONLY_INSTRUCTIONS = """\
Answer the question with a short rationale, then the answer on a last line of its own.

End with the line "Answer: ⟦ label⟧", naming one entity with its exact English \
Wikidata label."""
"""For an ablation: spans on the answer line only, so that a constraint can fix the
answer's form without grounding the rationale in facts."""

TOOL_INSTRUCTIONS = """\
Answer the question with a short rationale, then the answer on a last line of its own.

Use the tools to look up the facts you need in Wikidata before you answer.

End with the line "Answer: ⟦ label⟧", naming one entity with its exact English \
Wikidata label."""
"""For the KG lookup tool baseline (xgraph.toolkg); zero-shot, as tool calls take turns."""

PLAIN_INSTRUCTIONS = """\
Answer the question with a short rationale, then the answer on a last line of its own.

End with the line "Answer: name", naming one entity."""


def restyle(text: str, syntax: SpanSyntax) -> str:
    """The text with its spans, written in the default syntax, rewritten in another."""
    if syntax == _SYNTAX:
        return text

    def segment(part: str) -> str:
        if part.startswith("→"):
            return "→" + syntax.label_prefix + part[1 + len(_SYNTAX.label_prefix) :]
        if part.startswith(_SYNTAX.escape):  # an escape right after the opener
            return syntax.escape + part[len(_SYNTAX.escape) :]
        body = part[len(_SYNTAX.label_prefix) :]
        return syntax.label_prefix + body.replace(_SYNTAX.escape, syntax.escape)

    def span(match: re.Match[str]) -> str:
        parts = match.group(1).split(_SYNTAX.separator)
        return syntax.opener + syntax.separator.join(map(segment, parts)) + syntax.closer

    return _CANONICAL_SPAN.sub(span, text)


def wwq_messages(
    question: str,
    syntax: SpanSyntax | None = _SYNTAX,
    *,
    titles: bool = False,
    shots: bool = True,
    answer_only: bool = False,
    tools: bool = False,
    chain: bool = False,
    path_spans: bool = False,
    expressions: bool = False,
    observe_paths: int = 0,
) -> list[dict]:
    """Instructions, worked examples, and the question; syntax None asks for no spans.

    shots=False leaves the examples out, as the fine-tuning examples do (xgraph.sft).
    answer_only=True asks for a span on the answer line only; tools=True asks for KG
    lookups through function calls (xgraph.toolkg), without examples; chain=True asks
    for multi-hop answers as a chain of fact spans (CHAIN_INSTRUCTIONS, CHAIN_EXAMPLES).
    expressions=True (with path_spans) adds date paths and date comparisons to the path
    examples (EXPRESSION_PATH_INSTRUCTIONS).
    """
    if expressions and not path_spans:
        raise ValueError("expressions extend the path-span prompt: they need path_spans=True")
    if tools:
        return [
            {"role": "system", "content": TOOL_INSTRUCTIONS},
            {"role": "user", "content": question},
        ]
    if answer_only:
        instructions = ANSWER_ONLY_INSTRUCTIONS

        def render(text: str) -> str:
            *rationale, answer = text.splitlines()
            return "\n".join([strip_spans("\n".join(rationale)), answer])

    elif syntax is None:
        instructions, render = PLAIN_INSTRUCTIONS, strip_spans
    else:
        base = PATH_INSTRUCTIONS if path_spans else CHAIN_INSTRUCTIONS if chain else INSTRUCTIONS
        base = EXPRESSION_PATH_INSTRUCTIONS if expressions else base
        if titles:
            base = base.replace("\n\nEnd with", f" {DISAMBIGUATE}\n\nEnd with")
        instructions = restyle(base, syntax)
        render = lambda text: restyle(text, syntax)  # noqa: E731
    messages = [{"role": "system", "content": instructions}]
    examples = PATH_EXAMPLES if path_spans else CHAIN_EXAMPLES if chain else WWQ_EXAMPLES
    if expressions:
        examples = [*PATH_EXAMPLES, *_expression_examples()]
    for example in examples if shots else ():
        messages.append({"role": "user", "content": example["question"]})
        messages.append({"role": "assistant", "content": render(example["response"])})
    messages.append({"role": "user", "content": question})
    if observe_paths:
        if not path_spans or syntax is None or expressions:
            raise ValueError("observed prompts need path spans with expressions off")
        return observed_requests([{"messages": messages}], observe_paths, syntax)[0]["messages"]
    return messages


def bio_messages(topic: str) -> list[dict]:
    """FActScore's question, after the instructions and one worked biography."""
    return [
        {"role": "system", "content": BIO_INSTRUCTIONS},
        {"role": "user", "content": f"Tell me a bio of {BIO_EXAMPLE['topic']}."},
        {"role": "assistant", "content": BIO_EXAMPLE["response"]},
        {"role": "user", "content": f"Tell me a bio of {topic}."},
    ]


def strip_spans(text: str, syntax: SpanSyntax = _SYNTAX) -> str:
    """The text as the judge reads it: span texts kept, span syntax removed."""
    from xgraph.audit import semantic_text

    text = semantic_text(text, syntax)
    for delimiter in (syntax.opener, syntax.closer, syntax.separator, syntax.escape):
        text = text.replace(delimiter.strip(), " ")
    return re.sub(r" +([.,;:!?)])", r"\1", re.sub(r" {2,}", " ", text)).strip()


def final_answer(text: str, syntax: SpanSyntax | None = _SYNTAX) -> str | None:
    """The label in the last "Answer: ⟦ label⟧" line; None if missing or escaped.

    With syntax None (prompts without spans) the answer is the rest of the line.
    """
    if syntax is None:
        matches = re.findall(r"^Answer:\s*(.+?)\s*$", text, re.MULTILINE)
        return matches[-1] if matches else None
    pattern = (
        r"^Answer:\s*"
        + re.escape(syntax.opener + syntax.label_prefix)
        + "(.*?)"
        + re.escape(syntax.closer)
    )
    matches = re.findall(pattern, text, re.MULTILINE)
    if not matches:
        return None
    label = matches[-1]
    return None if syntax.escape.strip() in label else label


# The ReFactX protocol (Pozzi et al., ISWC 2025; github.com/rpo19/ReFactX, branch ISWC2025,
# prompt_base.py, Apache-2.0) for comparison: its system prompt as its own
# get_prompt_template renders it with the defaults, minus the paragraph on entity
# descriptions, which the gate KG does not hold; and its `no_descr` few-shot style, in
# examples rewritten to entity answers and to facts the gate KG holds, written as fact
# spans after its "Fact:" command. The answer line takes xgraph's form, so that every
# arm is scored alike; its constraint runs without escapes, as ReFactX has none.
REFACTX_SYSTEM = (
    "You are a helpful question-answering assistant that bases its answers on facts from "
    "a knowledge base and always respects the prompt.\n\nThe process to answer questions:"
    "\n\n    You receive an input question.\n\n    You determine the reasoning path needed "
    "to answer the question based on the information available.\n\n    You determine the "
    "kind of answer you are asked. It can be a yes/no, a single entity, or a list of "
    "entities. Pay attention to the questions whose answer is a list of entities (e.g. "
    "Which countries share a border with Spain?): you need to find all the answer entities "
    "and include them all in the final answer.\n\n    You get relevant facts with the "
    '"Fact:" command. You can rely on these facts and use them a proof for your answer.\n'
    "    While getting facts you continue the reasoning explaining it step by step.\n\n"
    "    You conclude with a concise answer that depending on the question can be a yes/no, "
    "a single entity, or a list of entities. Pay attention to the questions whose answer is "
    'a list of entities.\n\n    The answer MUST be based on the proofs you found with "Fact:".'
    '\n\nIf you didn\'t find proofs with "Fact:" that support an answer you stop and you '
    'reply: "I don\'t know.".\n\nIf the question requires to find proof that an event '
    "happen and you didn't find any proof, you can assume that event didn't happen.\n\nIn "
    "case you are taking too long for answering (e.g., you already generated ten facts that "
    "are not useful for the question), you stop and you answer based on the proofs you "
    "acquired to that point.\n\nYou must always follow these instructions precisely and "
    "ensure your responses adhere strictly to this prompt."
)
REFACTX_EXAMPLES = [
    (
        "Where was the director of Slumdog Millionaire born?",
        "Reasoning: To answer this question, I need to find who is the director of Slumdog "
        "Millionaire and then his place of birth. The final answer will be his place of birth.\n"
        "Fact: ⟦ Slumdog Millionaire · director · Danny Boyle⟧\n"
        "Now I need the place of birth of Danny Boyle.\n"
        "Fact: ⟦ Danny Boyle · place of birth · Radcliffe⟧\n"
        "I found proof that the director of Slumdog Millionaire is Danny Boyle, who was born "
        "in Radcliffe, so this is the answer to the question.\n"
        "Answer: ⟦ Radcliffe⟧",
    ),
    (
        "Which countries share a border with Spain?",
        "Reasoning: To answer this question, I need to find all the countries that have a "
        "land border with Spain. They may be several. The answer will be the list of all "
        "these countries.\n"
        "Fact: ⟦ Spain · shares border with · Portugal⟧\n"
        "Fact: ⟦ Spain · shares border with · France⟧\n"
        "Fact: ⟦ Spain · shares border with · Andorra⟧\n"
        "I found Portugal, France, and Andorra. But let me see if it there are more countries "
        "that share a border with Spain.\n"
        "Fact: ⟦ Spain · shares border with · Gibraltar⟧\n"
        "Fact: ⟦ Spain · shares border with · Morocco⟧\n"
        "Fact: ⟦ Spain · diplomatic relation · Italy⟧\n"
        "This fact is about diplomatic relations and not borders.\n"
        "Fact: ⟦ Spain · diplomatic relation · Greece⟧\n"
        "Also this fact is not about borders. I'm not able to find other countries bordering "
        "with Spain.\n"
        "I found proof that Spain shares borders with Portugal, France, Andorra, Gibraltar, "
        "and Morocco.\n"
        "Answer: ⟦ Portugal⟧, ⟦ France⟧, ⟦ Andorra⟧, ⟦ Gibraltar⟧, ⟦ Morocco⟧",
    ),
]


def refactx_messages(question: str) -> list[dict]:
    messages = [{"role": "system", "content": REFACTX_SYSTEM}]
    for asked, answered in REFACTX_EXAMPLES:
        messages += [{"role": "user", "content": asked}, {"role": "assistant", "content": answered}]
    return [*messages, {"role": "user", "content": question}]


def refactx_requests(path: str | Path, limit: int | None = None) -> list[dict]:
    questions = json.loads(Path(path).read_text(encoding="utf-8"))[:limit]
    return [{"id": q["id"], "messages": refactx_messages(q["utterance"])} for q in questions]


def wwq_requests(
    path: str | Path,
    limit: int | None = None,
    syntax: SpanSyntax | None = _SYNTAX,
    titles: bool = False,
    shots: bool = True,
    answer_only: bool = False,
    tools: bool = False,
    chain: bool = False,
    anchor: bool = False,
    path_spans: bool = False,
    linked: dict[str, list[int]] | None = None,
    expressions: bool = False,
) -> list[dict]:
    """Requests for a WWQ-format split; with anchor, each names its question's entities
    (from the sparql field, or given linked, the items a linker found in its text) for
    derivation and path mode to start from."""
    questions = json.loads(Path(path).read_text(encoding="utf-8"))[:limit]

    def anchors(q: dict) -> list[str]:
        if linked is not None:
            return [f"Q{item}" for item in linked.get(q["id"], [])]
        return re.findall(r"wd:(Q\d+)", q.get("sparql", ""))

    return [
        {
            "id": q["id"],
            **({"anchor": anchors(q)} if anchor else {}),
            "messages": wwq_messages(
                q["utterance"],
                syntax,
                titles=titles,
                shots=shots,
                answer_only=answer_only,
                tools=tools,
                chain=chain,
                path_spans=path_spans,
                expressions=expressions,
            ),
        }
        for q in questions
    ]


def bio_requests(archive: str | Path, part: str = "unlabeled") -> list[dict]:
    """One request per FActScore prompt entity, the entity as its id: the unlabeled
    entities (the gate's test set) or the labeled ones (development data)."""
    import zipfile

    with zipfile.ZipFile(archive) as z:
        names = z.read(f"data/{part}/prompt_entities.txt").decode("utf-8").splitlines()
    return [{"id": n, "messages": bio_messages(n)} for n in names if n.strip()]


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="python -m xgraph.prompts", description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    wwq = commands.add_parser("wwq", help="chat requests for WWQ questions")
    wwq.add_argument("--questions", type=Path, required=True, help="a WWQ split (.json)")
    wwq.add_argument("--limit", type=int, help="only the first N questions")
    wwq.add_argument("--out", type=Path, required=True, help="requests (JSONL)")
    wwq.add_argument(
        "--syntax", default="brackets", help="span syntax preset, or plain for no spans"
    )
    wwq.add_argument(
        "--titles", action="store_true", help="ask for Wikipedia titles where labels are shared"
    )
    wwq.add_argument(
        "--zero-shot", action="store_true", help="no worked examples, as for a fine-tuned model"
    )
    wwq.add_argument(
        "--refactx", action="store_true", help="the ReFactX protocol's prompt, for comparison"
    )
    wwq.add_argument(
        "--answer-only", action="store_true", help="spans on the answer line only (an ablation)"
    )
    wwq.add_argument("--tools", action="store_true", help="for the KG lookup tool baseline")
    wwq.add_argument(
        "--chain", action="store_true", help="multi-hop answers as a chain of fact spans"
    )
    wwq.add_argument(
        "--anchor", action="store_true", help="name each question's entities for derivation"
    )
    wwq.add_argument(
        "--path-spans", action="store_true", help="answers as a path span of relations"
    )
    wwq.add_argument(
        "--link-kg",
        type=Path,
        help="with --anchor: anchors linked in the question text against this KG directory",
    )
    bio = commands.add_parser("bio", help="chat requests for FActScore's biographies")
    bio.add_argument("--archive", type=Path, required=True, help="FActScore data.zip")
    bio.add_argument(
        "--part", choices=["unlabeled", "labeled"], default="unlabeled", help="FActScore entities"
    )
    bio.add_argument("--out", type=Path, required=True, help="requests (JSONL)")
    args = parser.parse_args(argv)
    if args.command == "bio":
        requests = bio_requests(args.archive, args.part)
    elif args.refactx:
        requests = refactx_requests(args.questions, args.limit)
    else:
        syntax = None if args.syntax == "plain" else preset(args.syntax)
        linked = None
        if args.link_kg:
            from xgraph.linking import link

            asked = json.loads(args.questions.read_text(encoding="utf-8"))[: args.limit]
            linked = link({q["id"]: q["utterance"] for q in asked}, args.link_kg)
        requests = wwq_requests(
            args.questions,
            args.limit,
            syntax,
            args.titles,
            shots=not args.zero_shot,
            answer_only=args.answer_only,
            tools=args.tools,
            chain=args.chain,
            anchor=args.anchor,
            path_spans=args.path_spans,
            linked=linked,
        )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as out:
        for request in requests:
            out.write(json.dumps(request, ensure_ascii=False) + "\n")
    print(f"{len(requests)} requests")


if __name__ == "__main__":
    main()
