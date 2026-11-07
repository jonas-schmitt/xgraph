"""The walk baseline: the model picks, round by round, which KG relation to follow.

A strong baseline for answering through the KG (in the spirit of GWalk and other
iterative graph walks): asked for the entity the question starts from, the model names
it as a span; its best-connected entity with that label starts the walk. Each round
lists every relation the current entities have (the menu), numbered in label order, and
the model answers with a number (the relation to follow) or 0 (the current entities
answer the question); max_options caps the menu at its first relations, and each record
counts the relations offered (options_offered) and left out (options_clipped) each round.
With min_hops (default zero), stopping is refused until that many relations have been
followed. Refused stops consume rounds within the same max_hops + 1 decision budget;
an unmet minimum delivers no answer. The default prompts and records are unchanged.
After at most max_hops relations the walk stops, and its answer is the best-connected
entity reached. The model never names an intermediate entity, so an edited fact changes
the walk wherever its relation is followed. A request naming a KG version (kg_version)
walks the KG store read through it (toolkg.VersionedStore), so counterfactual KGs need
no store of their own. All questions of a round go to vLLM at once.

With a matched executor (--matched-paths), the walk reads the KG index and served
tokenizer instead: exact unquoted starts, sitelink rank where selected, all start
senses alive until final committed delivery, capped frontiers and every writable
relation label and alias. This mode needs the labels observation. It records plan
and reached for separate engine delivery; without the option legacy records stay
unchanged.

Each round's prompt also shows the walk observation, what the model is told about the
current entities (the start, then what each chosen relation reached), one of
OBSERVATIONS, in a fixed format:

- `labels`: their labels, at most MAX_SHOWN (the walk as it always was);
- `types`: their dominant class and their number, `⟨city⟩, 1 entity` or
  `⟨human⟩, 3 entities` (frontier_class; `⟨unknown⟩` when none of them has a class);
- `type-only`: their dominant class alone, `⟨city⟩`;
- `count`: only their number, `1 entity`;
- `none`: nothing, the observation line left out, so that only the path so far shows;
- `placebo-type`: a wrong class with the true number, `⟨country⟩, 1 entity`, by a seeded
  rule: the current entities' relations on the menu, in an order drawn from the seed, the
  request id and the round, are followed one by one (at most PLACEBO_TRIES), and the
  first dominant class they reach whose label differs from the true one's and names no
  Wikimedia project class (`Wikimedia category`) is shown; with none, a class of
  FALLBACK_CLASSES other than the true one, drawn the same way
  (counted in the record's placebo_fallbacks);
- `shuffled-type`: the true number with the true class of another request's current
  entities in the same round, `⟨country⟩, 1 entity`: drawn, by the seed, the request id
  and the round, from the classes the types observation would show the round's other
  requests (`unknown` among them), one draw per request, leaving out this request's own
  class; a real class, unrelated to these entities. With none (a request alone in its
  round, or every other request's class its own), a class of FALLBACK_CLASSES other than
  the true one, counted in placebo_fallbacks;
- `filler`: the true number behind neutral padding, `⟨····⟩, 1 entity`, the dots chosen
  so that it has as many tokens of the served tokenizer as the types observation would
  have (the nearest count where none is equal), so that `types` against `filler` differs
  in the class alone. (Before, filler was the padding without the number.)
- `previews`: walk previews, what each relation on the menu would reach from the current
  entities, in the types observation's format, `3. place of birth → ⟨city⟩, 1 entity`
  (the relation followed as the walk follows it, through the request's KG version), and
  a stop option `0. stop → ⟨human⟩, 1 entity` that holds the types observation instead
  of the observation line; each record keeps the previews shown each round (previews);
- `shuffled-previews`: the same previews and stop option, the previews permuted among the
  menu's relations by the seed, the request id and the round, so that the tokens and the
  set of previews stay and only their tie to the relations breaks.

The four SLOTTED previews hold the menu, its order, its arrows, the stop option and every
true count fixed, and differ only in the candidates' classes. Each shows coarse classes
(with or without coarse_types) in the type slot: `⟨human⟩` padded inside its brackets with
FILLER to as many tokens of the served tokenizer as the longest of SLOT_CLASSES has on the
menu, so that a slot's length tells nothing of its class. Their stop option shows the true
coarse class of the current entities in the slot, `0. stop → ⟨human···⟩, 1 entity`:

- `slot-previews`: each relation's true coarse class, the treatment the others control;
- `neutral-previews`: no class, the slot all padding (NEUTRAL), `⟨·····⟩, 3 entities`;
- `count-shuffled-previews`: the true classes permuted among the menu's relations whose
  counts fall in one count stratum (STRATA: one entity, two to five, six to 64, more),
  first among those whose classes are of one length in tokens, then swapping each relation
  still showing its own class with one of its stratum where both then show another's; by
  observe_seed, the request id and the round, so that seeds 0, 1 and 2 give three
  placebos, and every count stays its relation's own. Each record counts, each round, the
  relations whose class changed (types_changed), the treatment's coverage;
- `static-previews`: each relation's static class (relation_classes), the coarse class
  most of its triples reach over the whole KG, computed once from the KG store as it is,
  so that it holds what a static relation-to-class prior knows and nothing of the
  current entities or the request's KG version.

The previews and the controls are per request: what one request is shown never depends on
the others asked with it.

A dominant class can be as specific as an entity (a frontier's instance of may name
`Japan` or `1996 Summer Olympics`), so it can give away which entities were reached. With
coarse_types, every observation that shows a class (all but CLASSLESS) shows its coarse
class instead (coarse_class): the first of a fixed list, COARSE_CLASSES (`human`,
`country`, `human settlement`, `film`, ...), among the class itself, else among its
labelled superclasses one subclass of step up, then two, up to COARSE_DEPTH steps, and
`unknown` when none is; placebo-type and shuffled-type then draw coarse classes, and
their fallback is a coarse class.

Only the observation differs between them: the menu, the KG walked and the answer, always
the best-connected entity the KG reached, are the same, so that a difference in plans is
the observation's. Classes come from the store the request walks, read through its KG
version.
"""

from __future__ import annotations

import bisect
import random
import re

START_PROMPT = (
    "Which entity does this question start from? Answer with one line, "
    '"Start: ⟦ label⟧", using its English Wikidata label.\n\nQuestion: {question}'
)
STEP_PROMPT = (
    "Question: {question}\n\nPath so far: {path}\n{observed}\n"
    "Which relation leads from the current entities toward the answer? Reply with its "
    "number only, or 0 if the current entities answer the question.\n{options}"
)
OBSERVED = "Current entities: {observation}\n"
STOP = "stop"
"""The stop option's name in a menu with previews, `0. stop → ⟨human⟩, 1 entity`."""
MAX_SHOWN = 8
SLOTTED = ("slot-previews", "neutral-previews", "count-shuffled-previews", "static-previews")
"""The previews that show coarse classes in the type slot, with or without --coarse-types:
the true ones and the controls that hold all of them but the candidates' classes fixed."""
PREVIEWED = ("previews", "shuffled-previews", *SLOTTED)
OBSERVATIONS = (
    "labels", "types", "type-only", "count", "none", "placebo-type", "shuffled-type", "filler",
    *PREVIEWED,
)  # fmt: skip
CLASSLESS = ("labels", "count", "none")
"""The observations that show no class, which coarse types do not apply to."""
CLASS_RELATIONS = (31, 279)
"""instance of, then subclass of: the relations an entity's classes are read from."""
MAX_TYPED = 64
"""The most entities of a frontier whose classes are read, best-connected first."""
PLACEBO_TRIES = 8
UNKNOWN = "unknown"
FALLBACK_CLASSES = ("human", "country", "city", "film", "organization", "taxon")
FILLER = "·"
COARSE_CLASSES = {
    5: "human", 16521: "taxon", 6256: "country", 486972: "human settlement",
    56061: "administrative territorial entity", 82794: "geographic region",
    618123: "geographical feature", 43229: "organization", 11424: "film", 482994: "album",
    2188189: "musical work", 7725634: "literary work", 13442814: "scholarly article",
    17537576: "creative work", 1656682: "event", 618779: "award", 4164871: "position",
    28640: "profession", 34770: "language", 101352: "family name", 202444: "given name",
    41176: "building", 811979: "architectural structure", 7187: "gene", 8054: "protein",
    11173: "chemical compound", 4167836: "Wikimedia category",
}  # fmt: skip
"""The coarse classes, by QID, in the order that breaks a tie between two at one depth:
the more specific first (country before geographic region, film before creative work)."""
COARSE_DEPTH = 6
"""The most subclass of steps from a class to its coarse class."""
SLOT_CLASSES = (*COARSE_CLASSES.values(), UNKNOWN)
"""What a type slot can name; the slot's length is that of the longest of them."""
NEUTRAL = ""
"""What neutral-previews' type slot names: nothing, the slot all padding."""
STRATA = (1, 5, 64)
"""The count strata's upper bounds: one entity, two to five, six to 64, and more."""

_PROJECT = re.compile(r"(Wikimedia|Wikipedia|WikiProject)\b")
_START = re.compile(r"Start:\s*⟦\s*(.+?)\s*⟧")
_NUMBER = re.compile(r"\d+")


def parse_start(text: str) -> str | None:
    match = _START.search(text)
    return match.group(1) if match else None


def parse_choice(text: str, options: int) -> int | None:
    """The chosen option (0 to stop), or None if the reply names none of them."""
    match = _NUMBER.search(text)
    if not match:
        return None
    choice = int(match.group())
    return choice if 0 <= choice <= options else None


def count_stratum(n: int) -> int:
    """The count stratum of n entities (STRATA): 0 for one, 1 for two to five, 2 for six to
    64, 3 for more."""
    return bisect.bisect_left(STRATA, n)


def _count(items: list) -> str:
    return f"{len(items)} {'entity' if len(items) == 1 else 'entities'}"


def _best(store, items: list[int]) -> list[int]:
    return sorted(items, key=lambda i: (-store.degree(i), i))


def coarse_class(store, cls: int, cache: dict | None = None) -> str | None:
    """The class's coarse class: the first of COARSE_CLASSES among the class itself, else
    among its labelled superclasses one subclass of step up, then two, up to COARSE_DEPTH
    steps; None when none is. cache keeps each class's coarse class."""
    key = ("coarse", id(store), cls)
    if cache is not None and key in cache:
        return cache[key]
    found, level, seen = None, [cls], {cls}
    for depth in range(COARSE_DEPTH + 1):
        listed = [c for c in COARSE_CLASSES if c in level]
        if listed:
            found = COARSE_CLASSES[listed[0]]
            break
        if depth == COARSE_DEPTH:
            break
        level = sorted({o for c in level for o in store.objects_of([c], 279)} - seen)
        seen.update(level)
    if cache is not None:
        cache[key] = found
    return found


def relation_classes(store, cache: dict | None = None) -> dict[int, str]:
    """Each relation's static class: of the coarse classes its triples reach over the whole
    KG (each object's dominant class, as frontier_class gives it for that object alone,
    coarsened, `unknown` when it has none), the one most triples reach, the earlier in
    SLOT_CLASSES on a tie. A relation without a labelled object has none. Read from the KG
    store as it is, never through a KG version."""
    first, then = (store.lowest_objects(prop) for prop in CLASS_RELATIONS)
    votes: dict[int, dict[str, int]] = {}
    for prop, item, n in store.object_counts():
        cls = first.get(item, then.get(item))
        coarse = (coarse_class(store, cls, cache) if cls is not None else None) or UNKNOWN
        tally = votes.setdefault(prop, {})
        tally[coarse] = tally.get(coarse, 0) + n
    rank = {c: k for k, c in enumerate(SLOT_CLASSES)}
    return {p: min(t, key=lambda c: (-t[c], rank[c])) for p, t in votes.items()}


def _derange(shown: list[str], rows: list[int], rng: random.Random) -> None:
    """Permutes shown's values at rows so that as many rows as can change do: the rows in an
    order drawn from rng, sorted by how often their value occurs (ties in an order drawn
    too), then each value moved on by the most any value occurs."""
    rows = list(rows)
    rng.shuffle(rows)
    values = [shown[i] for i in rows]
    occurs = {v: values.count(v) for v in values}
    rank = {v: rng.random() for v in sorted(occurs)}
    order = sorted(range(len(rows)), key=lambda k: (-occurs[values[k]], rank[values[k]]))
    step = max(occurs.values(), default=0)
    for k, at in enumerate(order):
        shown[rows[order[(k + step) % len(rows)]]] = values[at]


def frontier_class(
    store, items: list[int], cache: dict | None = None, coarse: bool = False
) -> str | None:
    """The dominant class of the entities (the first MAX_TYPED): of each one's labelled
    classes under instance of, or else subclass of, the one most of them have, then the
    lowest QID; None when none has a class. With coarse, that class's coarse class
    (coarse_class). cache keeps each entity's classes."""
    votes: dict[int, int] = {}
    names: dict[int, str] = {}
    for item in items[:MAX_TYPED]:
        key = (id(store), item)
        if cache is None or key not in cache:
            found: dict[int, str] = {}
            for prop in CLASS_RELATIONS:
                found = store.objects_of([item], prop)
                if found:
                    break
            if cache is not None:
                cache[key] = found
        else:
            found = cache[key]
        for cls, label in found.items():
            votes[cls] = votes.get(cls, 0) + 1
            names[cls] = label
    if not votes:
        return None
    dominant = min(votes, key=lambda c: (-votes[c], c))
    return coarse_class(store, dominant, cache) if coarse else names[dominant]


class _Observer:
    """The walk observation of a state's current entities (see the module docstring)."""

    def __init__(
        self,
        mode: str,
        llm,
        seed: int,
        coarse: bool = False,
        store=None,
        observe_seed: int = 0,
    ) -> None:
        if mode not in OBSERVATIONS:
            raise ValueError(f"unknown walk observation {mode!r}; expected one of {OBSERVATIONS}")
        if coarse and mode in CLASSLESS:
            raise ValueError(f"coarse types need an observation with a class, not {mode!r}")
        self.mode, self.seed, self.observe_seed = mode, seed, observe_seed
        self.coarse = coarse or mode in SLOTTED
        self.fallbacks = tuple(COARSE_CLASSES.values()) if coarse else FALLBACK_CLASSES
        self.cache: dict = {}
        self.tokenizer = llm.get_tokenizer() if mode == "filler" or mode in SLOTTED else None
        self.fillers: dict[int, str] = {}
        self.degrees: dict = {}
        self.store = store  # the KG store as it is: static-previews' classes
        self.static: dict[int, str] | None = None
        self.slots: dict[str, str] = {}
        self.lengths: dict[str, int] = {}

    def round(self, states: list[dict], step: int) -> list[str | None]:
        """The observations of one round's states, in order."""
        if self.mode != "shuffled-type":
            return [self(state, step) for state in states]
        classes = [
            frontier_class(s["store"], s["current"], self.cache, self.coarse) or UNKNOWN
            for s in states
        ]
        ids = [s["request"]["id"] for s in states]
        pool = sorted(zip(ids, classes, strict=True))
        return [
            f"⟨{self._shuffled(state, true, pool, step)}⟩, {_count(state['current'])}"
            for state, true in zip(states, classes, strict=True)
        ]

    def __call__(self, state: dict, step: int) -> str | None:
        read, items = state["store"], state["current"]
        if self.mode == "labels":
            return ", ".join(read.label(i) or "" for i in items[:MAX_SHOWN])
        if self.mode == "none":
            return None
        count = _count(items)
        if self.mode == "count":
            return count
        true = frontier_class(read, items, self.cache, self.coarse)
        if self.mode == "placebo-type":
            return f"⟨{self._placebo(state, true, step)}⟩, {count}"
        if self.mode == "type-only":
            return f"⟨{true or UNKNOWN}⟩"
        if self.mode in SLOTTED:
            return f"{self._slot(true or UNKNOWN)}, {count}"
        typed = f"⟨{true or UNKNOWN}⟩, {count}"
        return self._filler(typed, count) if self.mode == "filler" else typed

    def previews(self, state: dict, step: int) -> list[str] | None:
        """What each relation on the menu would reach, in menu order, in the types
        observation's format; shuffled among the relations by the seed, the request id and
        the round in shuffled-previews; with each class in the type slot, replaced as the
        control says (see the module docstring), in SLOTTED. None without previews."""
        if self.mode not in PREVIEWED:
            return None
        read, current = state["store"], state["current"]
        classes, sizes = [], []
        for prop, _ in state["options"]:
            reached = list(read.objects_of(current, prop))
            sizes.append(len(reached))
            if self.mode == "neutral-previews":
                classes.append(NEUTRAL)
            elif self.mode == "static-previews":
                if self.static is None:
                    self.static = relation_classes(self.store, self.cache)
                classes.append(self.static.get(prop, UNKNOWN))
            else:
                if len(reached) > MAX_TYPED:  # the walk's order, where it picks the typed ones
                    reached = sorted(reached, key=lambda i: (-self._degree(read, i), i))
                classes.append(frontier_class(read, reached, self.cache, self.coarse) or UNKNOWN)
        if self.mode == "count-shuffled-previews":
            shown = self._count_shuffled(classes, sizes, state, step)
            state["types_changed"].append(sum(s != c for s, c in zip(shown, classes, strict=True)))
            classes = shown
        counts = [f"{n} {'entity' if n == 1 else 'entities'}" for n in sizes]
        if self.mode in SLOTTED:
            return [f"{self._slot(c)}, {n}" for c, n in zip(classes, counts, strict=True)]
        shown = [f"⟨{c}⟩, {n}" for c, n in zip(classes, counts, strict=True)]
        if self.mode == "shuffled-previews":
            random.Random(f"{self.seed}/{state['request']['id']}/{step}").shuffle(shown)
        return shown

    def _count_shuffled(self, classes: list[str], sizes: list[int], state: dict, step: int):
        """The classes permuted among the rows of one count stratum, by the observe seed, the
        request id and the round: first among the rows whose classes are of one length in
        tokens (_derange), then each row still showing its own class swapped with a row of
        its stratum where both then show another's, the rows in an order drawn too."""
        rng = random.Random(f"{self.observe_seed}/{state['request']['id']}/{step}")
        strata = [count_stratum(n) for n in sizes]
        groups: dict[tuple[int, int], list[int]] = {}
        for i, cls in enumerate(classes):
            groups.setdefault((strata[i], self._length(cls)), []).append(i)
        shown = list(classes)
        for group in sorted(groups):
            _derange(shown, groups[group], rng)
        rows = list(range(len(classes)))
        rng.shuffle(rows)
        for i in rows:
            if shown[i] != classes[i]:
                continue
            for j in rows:
                if strata[j] == strata[i] and shown[j] != classes[i] and shown[i] != classes[j]:
                    shown[i], shown[j] = shown[j], shown[i]
                    break
        return shown

    def _slot_tokens(self, text: str) -> int:
        return self._tokens(f" → {text},")  # as it stands on the menu

    def _length(self, name: str) -> int:
        """The class's length in tokens, unpadded, on the menu."""
        if name not in self.lengths:
            self.lengths[name] = self._slot_tokens(f"⟨{name}⟩")
        return self.lengths[name]

    def _slot(self, name: str) -> str:
        """The type slot naming the class: `⟨name⟩` padded with FILLER inside the brackets to
        the tokens of the longest of SLOT_CLASSES on the menu (the nearest count where none
        is equal), so that its length tells nothing of the class."""
        if name not in self.slots:
            budget = max(self._length(c) for c in SLOT_CLASSES)
            best: tuple[int, str] | None = None
            for k in range(4 * budget + 4):
                text = f"⟨{name}{FILLER * k}⟩"
                gap = abs(self._slot_tokens(text) - budget)
                if best is None or gap < best[0]:
                    best = (gap, text)
                if gap == 0:
                    break
            self.slots[name] = best[1]
        return self.slots[name]

    def _degree(self, read, item: int) -> int:
        key = (id(read), item)
        if key not in self.degrees:
            self.degrees[key] = read.degree(item)
        return self.degrees[key]

    def _shuffled(self, state: dict, true: str, pool: list[tuple], step: int) -> str:
        rng = random.Random(f"{self.seed}/{state['request']['id']}/{step}")
        others = [c for i, c in pool if i != state["request"]["id"] and c != true]
        if others:
            return rng.choice(others)
        state["placebo_fallbacks"] += 1
        return rng.choice([c for c in self.fallbacks if c != true and not _PROJECT.match(c)])

    def _placebo(self, state: dict, true: str | None, step: int) -> str:
        rng = random.Random(f"{self.seed}/{state['request']['id']}/{step}")
        props = sorted(p for p, _ in state["options"])
        rng.shuffle(props)
        read, sample = state["store"], state["current"][:MAX_TYPED]
        for prop in props[:PLACEBO_TRIES]:
            reached = sorted(read.objects_of(sample, prop))
            found = frontier_class(read, reached, self.cache, self.coarse)
            if found is not None and found != true and not _PROJECT.match(found):
                return found
        state["placebo_fallbacks"] += 1
        return rng.choice([c for c in self.fallbacks if c != true and not _PROJECT.match(c)])

    def _tokens(self, text: str) -> int:
        return len(self.tokenizer.encode(text, add_special_tokens=False))

    def _filler(self, typed: str, count: str) -> str:
        n = self._tokens(typed)
        if (n, count) not in self.fillers:
            best: tuple[int, str] | None = None
            for k in range(4 * n + 4):
                text = f"⟨{FILLER * k}⟩, {count}"
                gap = abs(self._tokens(text) - n)
                if best is None or gap < best[0]:
                    best = (gap, text)
                if gap == 0:
                    break
            self.fillers[n, count] = best[1]
        return self.fillers[n, count]


def walk(
    llm,
    requests: list[dict],
    store,
    params,
    max_hops: int = 4,
    versions: dict | None = None,
    observe: str = "labels",
    seed: int = 0,
    max_options: int | None = None,
    coarse_types: bool = False,
    observe_seed: int = 0,
    min_hops: int = 0,
    executor=None,
    stride: int = 0,
    max_prompt_tokens: int | None = None,
) -> list[dict]:
    """One record per request: the path walked, the answer text, the observations shown,
    and the tokens used. A request with a kg_version walks the store read through that
    KG version, versions[name] (toolkg.version_stores); observe is the walk observation,
    seed draws the placebo's classes, and observe_seed count-shuffled-previews'
    permutations."""
    if stride:
        from xgraph.stride import walk as stride_walk

        if observe != "labels" or max_options is not None or coarse_types:
            raise ValueError("stride needs labels and complete atomic menus")
        return stride_walk(
            llm,
            requests,
            store,
            params,
            executor=executor,
            stride=stride,
            max_hops=max_hops,
            versions=versions,
            max_prompt_tokens=max_prompt_tokens,
            min_hops=min_hops,
        )
    if not 0 <= min_hops <= max_hops:
        raise ValueError("min_hops must be between zero and max_hops")
    if executor is not None and observe != "labels":
        raise ValueError("matched walk needs the labels observation")
    observer = _Observer(observe, llm, seed, coarse_types, store, observe_seed)
    states = []
    for request in requests:
        name = request.get("kg_version")
        if name is not None and name not in (versions or {}):
            raise ValueError(f"request {request['id']!r} reads KG version {name!r}, not given")
        read = store if name is None else versions[name]
        states.append({"request": request, "store": read, "path": [], "current": [],
                       "done": False, "tokens": 0, "prompt_tokens": 0, "log": [],
                       "observations": [], "placebo_fallbacks": 0, "offered": [],
                       "clipped": [], "previews": [], "types_changed": []})  # fmt: skip

    def ask(active, prompts):
        from xgraph.prompts import SET_INSTRUCTION

        prompts = [
            SET_INSTRUCTION
            + "\nChoose a path whose complete reached set answers the question.\n"
            + p
            if s["request"].get("answer_mode") == "set"
            else p
            for s, p in zip(active, prompts, strict=True)
        ]
        outputs = llm.chat(
            [[{"role": "user", "content": p}] for p in prompts],
            params,
            chat_template_kwargs={"enable_thinking": False},
            use_tqdm=False,
        )
        for state, output in zip(active, outputs, strict=True):
            completion = output.outputs[0]
            state["tokens"] += len(completion.token_ids)
            state["prompt_tokens"] += len(output.prompt_token_ids)
            state["log"].append(completion.text)
        return [o.outputs[0].text for o in outputs]

    def frontier(state, path):
        request = state["request"]
        return [
            int(q[1:])
            for q in executor.frontier(
                path[0], path[1:], request.get("anchor"), request.get("kg_version")
            )
        ]

    for state in states:  # anchored: the walk starts from the question's entities
        anchor = [int(e[1:]) for e in state["request"].get("anchor") or []]
        if anchor:
            read = state["store"]
            state["current"] = (
                _best(read, anchor)[:1]
                if executor is None
                else frontier(state, [read.label(anchor[0]) or ""])
            )
            if not state["current"]:
                state["done"] = True
                continue
            state["path"] = [read.label(state["current"][0]) or ""]
    unanchored = [s for s in states if not s["current"] and not s["done"]]
    replies = (
        ask(
            unanchored, [START_PROMPT.format(question=s["request"]["question"]) for s in unanchored]
        )
        if unanchored
        else []
    )
    for state, reply in zip(unanchored, replies, strict=True):
        start = parse_start(reply)
        read = state["store"]
        items = (
            (frontier(state, [start]) if executor is not None else read.items_named(start))
            if start
            else []
        )
        if items:
            state["path"] = [start]
            state["current"] = items if executor is not None else _best(read, items)[:1]
        else:
            state["done"] = True
    for step in range(max_hops + 1):
        active = [s for s in states if not s["done"]]
        if not active:
            break
        for state in active:
            read = state["store"]
            if executor is None:
                relations = sorted(read.relations_of(state["current"]).items(), key=lambda r: r[1])
            else:
                request, path = state["request"], state["path"]
                relations = [
                    (label, label)
                    for label in executor.frontier_labels(
                        path[0], path[1:], request.get("anchor"), request.get("kg_version")
                    )
                ]
            state["options"] = relations[:max_options]
            state["offered"].append(len(state["options"]))
            state["clipped"].append(len(relations) - len(state["options"]))
        prompts = []
        for state, observation in zip(active, observer.round(active, step), strict=True):
            can_stop = len(state["path"]) - 1 >= min_hops
            state["observations"].append(observation)
            previews = observer.previews(state, step)
            if previews is None:
                lines = [f"{n}. {label}" for n, (_, label) in enumerate(state["options"], 1)]
            else:  # the observation moves to the stop option
                state["previews"].append(previews)
                menu = zip(state["options"], previews, strict=True)
                lines = ([f"0. {STOP} → {observation}"] if can_stop else []) + [
                    f"{n}. {label} → {shown}" for n, ((_, label), shown) in enumerate(menu, 1)
                ]
                observation = None
            template = (
                STEP_PROMPT
                if can_stop
                else STEP_PROMPT.replace(", or 0 if the current entities answer the question", "")
            )
            prompts.append(
                template.format(
                    question=state["request"]["question"],
                    path=" → ".join(state["path"]),
                    observed=""
                    if observation is None
                    else OBSERVED.format(observation=observation),
                    options="\n".join(lines),
                )
            )
        for state, reply in zip(active, ask(active, prompts), strict=True):
            choice = parse_choice(reply, len(state["options"]))
            if choice == 0 and len(state["path"]) - 1 < min_hops and state["options"]:
                continue
            if not choice or len(state["path"]) > max_hops:
                state["done"] = True
                continue
            prop, label = state["options"][choice - 1]
            reached = (
                frontier(state, [*state["path"], label])
                if executor is not None
                else state["store"].objects_of(state["current"], prop)
            )
            if not reached:
                state["done"] = True
                continue
            state["path"].append(label)
            state["current"] = (
                list(reached) if executor is not None else _best(state["store"], list(reached))
            )
    records = []
    for state in states:
        unmet = len(state["path"]) - 1 < min_hops
        answer = (
            state["store"].label(state["current"][0]) if state["current"] and not unmet else None
        )
        matched = None
        if executor is not None and state["current"] and not unmet:
            request, path = state["request"], state["path"]
            matched = executor.execute(
                path[0], path[1:], request.get("anchor"), request.get("kg_version")
            )
            answer = state["store"].label(int(matched.answer[1:])) if matched.answer else None
        text = "Path: " + " → ".join(state["path"])
        if answer:
            text += f"\nAnswer: ⟦ {answer}⟧"
        records.append(
            {
                "id": state["request"]["id"],
                "arm": "walk",
                "text": text,
                "path": state["path"],
                "replies": state["log"],
                "observe": observe,
                "observations": state["observations"],
                "options_offered": state["offered"],
                "options_clipped": state["clipped"],
                **(
                    {"placebo_fallbacks": state["placebo_fallbacks"]}
                    if observe in ("placebo-type", "shuffled-type")
                    else {}
                ),
                **({"previews": state["previews"]} if observe in PREVIEWED else {}),
                **(
                    {"types_changed": state["types_changed"]}
                    if observe == "count-shuffled-previews"
                    else {}
                ),
                "token_ids": [],
                "generated_tokens": state["tokens"],
                "prompt_tokens": state["prompt_tokens"],
                "finish_reason": "min_hops_unmet" if min_hops and unmet else "stop",
                **({"min_hops": min_hops} if min_hops else {}),
                **(
                    {
                        "matched_paths": True,
                        "plan": {"start": matched.plan[0], "relations": list(matched.plan[1:])}
                        if matched and matched.plan
                        else None,
                        "reached": [int(q[1:]) for q in matched.reached] if matched else [],
                        **(
                            {"kg_version": state["request"]["kg_version"]}
                            if state["request"].get("kg_version")
                            else {}
                        ),
                    }
                    if executor is not None
                    else {}
                ),
            }
        )
    from xgraph.programs import set_execution_fields

    for record, state in zip(records, states, strict=True):
        if state["request"].get("answer_mode") == "set" and executor is not None:
            record.update(set_execution_fields(record, state["request"], executor, state["store"]))
    return records
