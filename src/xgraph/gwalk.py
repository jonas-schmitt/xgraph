"""GWalk, the MQuAKE-Remastered authors' editing method, as a baseline arm (`gwalk`).

GWalk (Zhong et al., ICLR 2025) extracts a question's subject, has the model break the
question into relations, and answers hop by hop from the model's own memory; where its
bank of edited facts holds the hop (subject label and cloze template, matched with the
contriever), the edited object replaces the model's answer. The bank holds only edited
facts, and upstream's per-case mask hides another case's edit of a hop on the case's own
gold path. Every unedited hop comes from the model.

This module runs the authors' released loop, vendored in xgraph.gwalk_upstream (its
changes are listed there), against vLLM's OpenAI-compatible completions endpoint: the
raw few-shot text without a chat template, upstream's token limits and temperatures,
the sampling defaults of the model's generation config (what Hugging Face generate
would use), and stop rules with upstream's effective stops (stop_rules). Two changes
adapt it to a thinking chat model, which upstream's models are not: requests ban the
model's thinking tags (thinking_bias), and a hop's answer call ends after the line the
loop reads as its generated answer (answer_line_end), where upstream's models write
"Retrieved fact:" and a chat model goes on with commentary. Each case runs
sequentially, as upstream; a thread pool runs many cases at once and the server batches
their requests. Each request has its own seed (request_seed), so a run does not depend
on the order in which the server sees the requests.

The edit setting is the paper's all-edit one (3000-edit on CF-3k): every case of the
dataset is edited and its edits are in the bank, masked per case. With --worlds (a
variant), each case's bank holds its world's edits (python -m xgraph.mquake --worlds),
the edits path mode's world KG holds, and the mask, which reads the case's gold path, is
off: no edit left in a world rewrites a hop of a member's path.
Only the cases asked in --questions run (--limit: the first N of them by case id), with
all three questions, so that python -m xgraph.editing scores the records like any arm's.

    python -m xgraph.gwalk --data data/mquake/MQuAKE-Remastered-CF-3k.json \\
        --questions target/mqr-cf-3k-worlds/questions.json --url http://127.0.0.1:8766 \\
        --model /model --out target/mqr-cf-3k-worlds/gwalk.jsonl

scripts/gwalk.sh starts the server and runs this in the pinned vLLM container. Writes one
record per question, in case order (the walk's transcript as text, ending with
"Answer: ⟦ label⟧" when the walk reached an answer; the relations, hops, calls, tokens,
and the stage that failed, if any), and <out>.summary.json: calls, tokens, and finish
reasons per stage, parse failures, relation retrievals, bank facts used (and KG facts,
with --kg-db), seconds, and the paper's measure (a case is correct when upstream's
check_answer accepts any of its questions' answers).

With --repair-answers (a variant), a hop whose answer line the loop does not find, where
upstream's loop would fail the hop or read commentary as its answer, gets the first line
of its completion in the few-shot's form "<sentence> {}. <answer>" (repair_answer): a
chat model writes the answer below a bare "Generated answer:", in markdown bold, or above
its commentary. The repair only reads what the completion holds; a completion that leaves
the slot empty ("... is {}.") keeps failing its hop. Repaired hops are marked in the
records and counted in the summary.

With --complete-answers (a variant), a hop whose answer line is the model's template
sentence with the slot left empty, "Generated answer: <sentence> {}.", gets one more call
that continues from "Generated answer: <sentence> {}. ", the few-shot's form, to the end
of the line (complete_answer), and the loop reads the completed line. The object is the
model's; the calls and tokens count under their own stage, "complete", and completed
hops are marked in the records and counted in the summary. It composes with
--repair-answers, which runs first: a completion that holds an answer needs no call.

With --kg-db (a variant), GWalk reads a KG store as well as its bank, as path mode reads
the whole edited KG: a hop the bank has no fact for, whose relation template the bank
retrieved, is looked up in the store (KGLookup): the subject by name, the template by
the relation id the bank's id2rel gives it, and of several entities or objects the
best-connected one. A fact found replaces the model's answer as a bank fact does
("Retrieved fact contradicts ..."); a hop the store has no fact for keeps upstream's
behaviour. With --worlds, '{}' in the path stands for the world, so each case reads its
own world's KG store (data/wikidata/mqrw-kg-w{}.sqlite). Hops answered from the store
carry the triple's ids (kg_fact) and count in the summary's kg_facts, not bank_facts.

With --hop-order lookup-first (a variant; upstream's order, answer-first, is the
default), each hop is looked up in the bank, and with --kg-db in the KG store, before the
model is asked for its answer. A hop with a fact takes the fact's object without an
answer call, and its transcript reads "Retrieved fact: <fact>." and "Retrieved fact
answers the subquestion, so continue with this subject: <object>."; only a hop without
one gets upstream's answer call. In upstream's order a hop whose answer fails to parse
ends the walk before any lookup; here a fact the bank or the KG holds always decides its
hop. Such hops are marked looked_up_first and counted in the summary's
looked_up_first_hops.

With --conditions (a variant), GWalk asks the questions of an interventions build
(python -m xgraph.interventions, conditions.json) instead of MQuAKE's cases: each
condition is a case of its own with one question (condition_cases), reading the --kg-db
store through the condition's KG version from --kg-versions (condition_banks), and its
record's id is <question>@<condition>, so that python -m xgraph.interventions score
scores the records like any arm's, or the condition's record_id: python -m
xgraph.kgversions conditions writes MQuAKE's questions as conditions in their worlds'
versions whose records keep the questions' ids, which python -m xgraph.editing scores.
The bank holds no edited facts, since each
condition's KG version holds its edit; every fact comes from the store. A condition's
calls are seeded by its question, not the condition, so the conditions of a question
draw the same samples while their prompts agree. --limit takes the first N questions.
The relation catalogue, the templates the contriever snaps a breakdown's relation to, is
MQuAKE's (its 36 relations, from --data) unless --relations gives a KG's
relation_labels.jsonl: then it is the KG's own open inventory (open_bank), one template
"The <name> of {} is" per label and alias of each relation of the store's triples, with
the subquestion "What is the <name> of {}?", a name several relations share standing for
all of them (the first with a fact is read). With --relations-of-subject, a hop's
relation is retrieved only among the templates of the relations of the entities its
subject names in the condition's KG (subject_catalogue), or among all of them when the
subject names no entity. The summary has no paper measure for conditions.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import threading
import time
import urllib.error
import urllib.request
import zlib
from collections import Counter
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

UPSTREAM_COMMIT = "349dcc50460c251ee09a6804aab444508a016485"
CONTRIEVER = "facebook/contriever-msmarco"
CONTRIEVER_REVISION = "abe8c1493371369031bcb1e02acb754cf4e162fa"
SEED = 42
"""Upstream's seed (configs/global_setting.py)."""
PROMPTS = ("extract_entity", "fill_out_ga_w_blank2", "relation2subq_prompt2", "subq_breakdown")
STAGES = {
    ("done", 20): "extract",
    ("done", 50): "breakdown",
    ("end_block", 20): "rel2subq",
    ("facts", 50): "answer",
    ("line", 50): "complete",
}
"""Upstream's four call sites, told apart by stop rule and token limit: the subject
extraction, the relation breakdown, a subquestion from a relation the bank lacks, and a
hop's answer; and the completion of an answer's empty slot (--complete-answers)."""
HF_SAMPLING = {"top_k": 50, "top_p": 1.0, "min_p": None, "repetition_penalty": 1.0}
"""Hugging Face generate's defaults for what a generation config may leave out."""
RULES = {"facts": "facts", "done": "done", "end_block": "end_block", "line": "line"}
"""Upstream's stopping_criteria_dict, and `line` (--complete-answers); the names select a
rule of stop_rules."""
ANSWER_TOKENS, ANSWER_TEMPERATURE = 50, 1.0
"""The token limit and temperature of upstream's answer call (call_model's defaults),
which the completion of an empty slot takes over."""
LINE_RULES = frozenset({"facts"})
"""The rules whose calls also end after their generated-answer line (answer_line_end)."""
GENERATED = "Generated answer: "
THINKING = ("<think>", "</think>", "<|channel>")
"""A thinking model's tags, which no upstream model has and its prompts never show:
Qwen's thinking block, and the thinking channel Gemma 4 opens."""
BAN = -100.0
"""The strongest logit bias vLLM accepts."""
HOP_ORDERS = ("answer-first", "lookup-first")
"""Upstream's order (the model answers a hop, then a fact replaces the answer), and the
variant that looks a hop up before asking the model (--hop-order)."""


def load_prompts() -> dict[str, str]:
    """Upstream's four prompts, read in text mode as upstream reads them."""
    from importlib.resources import files

    base = files("xgraph.gwalk_upstream") / "prompts"
    return {name: (base / f"{name}.txt").read_text(encoding="utf-8") for name in PROMPTS}


def upstream_case(case: dict) -> dict:
    """A case of the MQuAKE-Remastered JSON release in the fields of its Hugging Face
    release, which the released code reads."""
    try:
        orig = case["orig"]
        return {
            "case_id": case["case_id"],
            "requested_rewrite": [
                {
                    "subject": e["subject"],
                    "prompt": e["prompt"],
                    "relation_id": e["relation_id"],
                    "target_new_str": e["target_new"]["str"],
                    "target_new_id": e["target_new"]["id"],
                    "target_true_str": e["target_true"]["str"],
                    "target_true_id": e["target_true"]["id"],
                    "question": e["question"],
                }
                for e in case["requested_rewrite"]
            ],
            "questions": case["questions"],
            "answer": case["answer"],
            "answer_alias": case["answer_alias"],
            "new_answer": case["new_answer"],
            "new_answer_alias": case["new_answer_alias"],
            "single_hops": case["single_hops"],
            "new_single_hops": case["new_single_hops"],
            "orig_triples": orig["triples"],
            "orig_triples_labeled": orig["triples_labeled"],
            "new_triples": orig["new_triples"],
            "new_triples_labeled": orig["new_triples_labeled"],
            "edit_triples": orig["edit_triples"],
        }
    except KeyError as error:
        raise ValueError(
            f"MQuAKE case {case.get('case_id', '?')}: missing field {error}"
        ) from error


def asked_case_ids(questions: list[dict]) -> list[int]:
    """The cases of question ids mquake-<case>-<k> (or mquake-<case>), by case id."""
    cases = set()
    for question in questions:
        qid = question["id"]
        parts = qid.split("-")
        if parts[0] != "mquake" or len(parts) not in (2, 3) or not parts[1].isdigit():
            raise ValueError(f"question id {qid!r} is not mquake-<case>-<k>")
        cases.add(int(parts[1]))
    return sorted(cases)


def stop_rules(dot_id: int, eos_ids: list[int]) -> dict[str, dict]:
    """Upstream's stopping criteria as vLLM stop rules with the same effective stops.

    Upstream's StoppingCriteriaSub compares only the last n-1 ids of its list (its loop
    starts at 1). With the Qwen2.5 entry of MODEL_to_SC, `facts` ("Retrieved fact:")
    fires on "rieved fact:", `done` ("Done.") on any lone "." token, so the subject
    extraction and the breakdown stop at their first "."; and `end_block` ("This ends the
    block:") never fires in these prompts, while other models' entries stop on " ends the
    block.", which ends the block the parser reads no sooner. The model's end-of-sequence
    ids stop every call, as they stop generate. `line`, no rule of upstream's, ends the
    completion of an answer's empty slot at the end of its line (--complete-answers).
    """
    eos = list(eos_ids)
    return {
        "facts": {"stop": ["Retrieved fact:"], "stop_token_ids": eos},
        "done": {"stop": [], "stop_token_ids": [dot_id, *eos]},
        "end_block": {"stop": [" ends the block."], "stop_token_ids": eos},
        "line": {"stop": ["\n"], "stop_token_ids": eos},
    }


def thinking_bias(token_to_id: Callable[[str], int | None]) -> dict[str, float]:
    """The logit bias that bans the served tokenizer's thinking tags, keyed by token id as
    the OpenAI API's logit_bias is; empty for a model without them."""
    ids = (token_to_id(tag) for tag in THINKING)
    return {str(i): BAN for i in ids if i is not None}


def answer_line_end(prompt: str, text: str) -> int | None:
    """Where in the completion `text` its first complete line that upstream's loop reads
    as a generated answer ends (after its newline), or None.

    The loop reads a hop's last line, less "Generated answer: ", and takes what follows
    the line's first ". " as the answer. Upstream's models end the line and write
    "Retrieved fact:", the stop, which the loop drops; a chat model writes commentary
    instead, whose last line the loop then reads. Ending the call after the first line
    that starts with "Generated answer: " and holds ". " gives the loop that line, as a
    stopping criterion would; a completion without one is left as it is.
    """
    full = prompt + text
    start = full.rfind("\n", 0, len(prompt)) + 1
    end = full.find("\n", len(prompt))
    while end >= 0:
        line = full[start:end]
        if reads_answer(line):
            return end + 1 - len(prompt)
        start, end = end + 1, full.find("\n", end + 1)
    return None


def reads_answer(line: str) -> bool:
    """Whether `line` is a generated answer the loop can read: it starts with "Generated
    answer: " and, after that, holds the ". " the loop splits the answer off at."""
    return line.startswith(GENERATED) and len(line[len(GENERATED) :].strip().split(". ")) >= 2


def repair_answer(asked: str, read: str) -> str | None:
    """The text the loop reads after a hop's answer call, ending with the answer line the
    completion holds, or None to leave the text as it is (--repair-answers, a variant).

    asked is the call's prompt; read the prompt and completion as the loop reads them
    (stripped, without upstream's stop, extra "Question: " blocks cut). The loop takes the
    text's last line as the hop's generated answer. When that line is not an answer
    line (reads_answer), a chat model has often written the answer elsewhere in the
    few-shot's form "<sentence> {}. <answer>": below a bare "Generated answer:", in
    markdown bold, or above commentary that ran to the token limit. The repair takes
    the completion's first line in that form (with "**" removed and a leading
    "Generated answer:" dropped, an answer after "{}. " that is not empty and holds no
    "{}"), and ends the text after it, as "Generated answer: " and that line. The search
    stops at a line holding "question:", where the model takes up another (sub)question.
    Nothing is guessed: a completion without such a line, or one that leaves the slot
    empty ("... is {}."), stays as it is, and its hop fails as upstream's would.
    """
    if reads_answer(read.strip().split("\n")[-1]):
        return None
    head = asked.strip()
    if not read.startswith(head):  # the loop cut into the prompt: no completion is left
        return None
    start = read.find("\n", len(head)) + 1
    while start > 0:
        end = read.find("\n", start)
        line = read[start:] if end < 0 else read[start:end]
        bare = line.replace("**", "").strip()
        if "question:" in bare.lower():
            return None
        bare = bare.removeprefix("Generated answer:").strip()
        answer = bare.partition("{}. ")[2].strip()
        if answer and "{}" not in answer:
            return read[:start] + GENERATED + bare
        start = end + 1
    return None


def empty_slot(asked: str, read: str) -> str | None:
    """The template sentence of a hop's answer line whose slot the model left empty, or
    None (--complete-answers, a variant).

    asked and read are repair_answer's. Where the few-shot writes "Generated answer:
    <sentence> {}. <answer>", a chat model often ends the sentence at the slot,
    "Generated answer: <sentence> {}.", and stops or goes on with commentary, so the
    loop finds no answer. The sentence counts when the line the loop reads is no answer
    line (reads_answer) and the completion's first line is "Generated answer: " and a
    sentence of the model's own that ends in " {}." and holds no other "{}". A bare
    "Generated answer:" is not completed: the sentence would not be the model's.
    """
    if reads_answer(read.strip().split("\n")[-1]):
        return None
    head = asked.strip()
    if not read.startswith(head + "\n"):  # no completion, or the loop cut into the prompt
        return None
    first = read[len(head) + 1 :].split("\n")[0]
    if not first.startswith(GENERATED):
        return None
    sentence = first[len(GENERATED) :].strip()
    if not sentence.endswith(" {}.") or sentence.count("{}") != 1:
        return None
    return sentence


def complete_answer(asked: str, read: str, model: Any) -> str | None:
    """The text the loop reads after a hop's answer call, with the answer line's empty
    slot completed by the model, or None to leave the text as it is (--complete-answers,
    a variant).

    When the answer line is a template sentence with an empty slot (empty_slot), one
    more call continues the call's prompt from "Generated answer: <sentence> {}. ", the
    few-shot's form, with the answer call's token limit and temperature, and ends at the
    end of the line (the `line` rule, stage "complete"). The completed line replaces the
    completion; the object is whatever the model writes, and a hop whose completion
    leaves the line without one still fails.
    """
    sentence = empty_slot(asked, read)
    if sentence is None:
        return None
    prompt = asked + GENERATED + sentence + " "
    return model.generate(prompt, "line", ANSWER_TOKENS, ANSWER_TEMPERATURE)


def sampling_defaults(generation_config: dict) -> dict:
    """The sampling settings generate takes from a generation config, with Hugging Face's
    defaults for those it leaves out (call_model passes only the temperature)."""
    settings = {key: generation_config.get(key, value) for key, value in HF_SAMPLING.items()}
    return {key: value for key, value in settings.items() if value is not None}


def request_seed(base: int, case_id: int, stage: str, k: int) -> int:
    """The seed of a case's k-th call of a stage."""
    return zlib.crc32(f"{base}-{case_id}-{stage}-{k}".encode())


def completion_text(prompt: str, choice: dict, stop_texts: dict[int, str]) -> str:
    """What upstream's call_model returns: the prompt and the completion, stripped, with
    the text of the stop that ended the completion, which generate keeps. vLLM leaves a
    stop's text out unless include_stop_str_in_output is set (it is), and versions differ
    in how they treat stop token ids, so a missing stop text is put back."""
    text = choice["text"]
    stop = choice.get("stop_reason")
    if isinstance(stop, str):
        tail = stop
    elif isinstance(stop, int) and not isinstance(stop, bool):
        tail = stop_texts.get(stop, "")  # end-of-sequence tokens have no text
    else:
        tail = ""
    if tail and not text.endswith(tail):
        text += tail
    return (prompt + text).strip()


def post_json(url: str, payload: dict, timeout: float) -> dict:
    request = urllib.request.Request(
        url, data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"}
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.load(response)
    except urllib.error.HTTPError as error:
        body = error.read().decode(errors="replace")[:1000]
        raise RuntimeError(f"POST {url}: HTTP {error.code}: {body}") from error


def get_json(url: str, timeout: float) -> dict:
    with urllib.request.urlopen(url, timeout=timeout) as response:
        return json.load(response)


class Completions:
    """Requests to vLLM's /v1/completions, shared by the threads."""

    def __init__(
        self,
        url: str,
        model: str,
        sampling: dict,
        rules: dict[str, dict],
        stop_texts: dict[int, str],
        timeout: float = 600.0,
        post: Callable[[str, dict, float], dict] = post_json,
        logit_bias: dict[str, float] | None = None,
    ):
        self.url, self.model, self.sampling = url.rstrip("/"), model, sampling
        self.rules, self.stop_texts, self.timeout, self.post = rules, stop_texts, timeout, post
        self.logit_bias = dict(logit_bias or {})

    def payload(
        self, prompt: str, rule: str, max_tokens: int, temperature: float, seed: int
    ) -> dict:
        payload = {
            "model": self.model,
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": temperature,
            **self.sampling,
            "seed": seed,
            **self.rules[rule],
            "include_stop_str_in_output": True,
        }
        if self.logit_bias:
            payload["logit_bias"] = self.logit_bias
        return payload

    def complete(
        self, prompt: str, rule: str, max_tokens: int, temperature: float, seed: int
    ) -> tuple[str, dict]:
        """call_model's text, and the call's tokens and finish reason ("answer_line" when
        the call ended after its generated-answer line; its generated tokens are still all
        the server generated)."""
        payload = self.payload(prompt, rule, max_tokens, temperature, seed)
        response = self.post(f"{self.url}/v1/completions", payload, self.timeout)
        choice = response["choices"][0]
        usage = {
            "prompt_tokens": response["usage"]["prompt_tokens"],
            "generated_tokens": response["usage"]["completion_tokens"],
            "finish_reason": choice.get("finish_reason"),
        }
        text = choice["text"]
        end = answer_line_end(prompt, text) if rule in LINE_RULES else None
        # nothing after the line but upstream's stop (or nothing at all): upstream's call
        if end is not None and text[end:] not in ("", choice.get("stop_reason")):
            usage["finish_reason"] = "answer_line"
            return (prompt + text[:end]).strip(), usage
        return completion_text(prompt, choice, self.stop_texts), usage


class CaseModel:
    """The `model` upstream's functions get for one case: call_model calls generate.

    A call's stage follows from its stop rule and token limit (STAGES); its seed from the
    case, the stage, and how many calls of that stage the case made before. Calls belong
    to the question gwalk_case set last, the breakdown's to its question, and the subject
    extraction's (made once per case) to the first question.
    """

    def __init__(self, client: Completions, case_id: int, seed: int, greedy: bool = False):
        self.client, self.case_id, self.seed, self.greedy = client, case_id, seed, greedy
        self.question: int | None = None
        self.calls: list[dict] = []
        self._made: Counter[str] = Counter()

    def generate(self, prompt: str, stop: str, generate_length: int, temperature: float) -> str:
        stage = STAGES[(stop, generate_length)]
        k = self._made[stage]
        self._made[stage] += 1
        text, usage = self.client.complete(
            prompt,
            stop,
            generate_length,
            0.0 if self.greedy else temperature,
            request_seed(self.seed, self.case_id, stage, k),
        )
        question = k if stage == "breakdown" else (self.question or 0)
        self.calls.append({"stage": stage, "question": question, **usage})
        return text


class Serialized:
    """A callable the threads share one call at a time (the contriever and its
    tokenizer)."""

    def __init__(self, inner: Any, lock: threading.Lock):
        self.inner, self.lock = inner, lock

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        with self.lock:
            return self.inner(*args, **kwargs)


@dataclass
class Bank:
    """GWalk's edit bank, built as upstream builds it: the label-keyed facts with the
    cases that made them, the relation templates and entities with their embeddings, and
    the dataset's aliases for fitting a subject onto a bank entity. masked: upstream's
    per-case mask, which hides another case's edit of a hop on the case's own gold path."""

    kg: dict
    rels: list[str]
    ents: list[str]
    id2rel: dict[str, str]
    ent2alias: dict[str, set[str]]
    rel_emb: Any = None
    ent_emb: Any = None
    masked: bool = True
    lookup: KGLookup | None = None
    """The KG store a hop the bank has no fact for is looked up in (--kg-db), or None."""
    rel2ids: dict[str, list[str]] | None = None
    """An open catalogue's relation ids of each template (open_bank); None: id2rel's."""
    subquestions: dict[str, str] | None = None
    """Each template's subquestion; None: upstream's REL2SUBQ."""
    relations_of_subject: bool = False
    """Retrieve a hop's template only among the subject's relations (subject_catalogue)."""


OPEN_TEMPLATE = "The {} of {{}} is"
"""An open catalogue's relation template, from a relation's label or alias."""
OPEN_SUBQUESTION = "What is the {} of {{}}?"
"""An open catalogue's subquestion, from the same name."""


def open_bank(relation_labels: str | Path, used: set[int] | None = None) -> Bank:
    """A bank whose relation catalogue is a KG's own (--relations, a variant), not
    MQuAKE's 36 relation templates: one template per label and alias of each relation in
    the KG's relation_labels.jsonl (OPEN_TEMPLATE, "The <name> of {} is"; of the
    relations in used only, when it is given: those of the KG store's triples), in the
    file's order, a name shared by relations giving one template of all of them, and a
    subquestion per template (OPEN_SUBQUESTION). It holds no edited facts and no entities:
    every fact is read from the KG store (--kg-db), which holds the edits."""
    rels: list[str] = []
    rel2ids: dict[str, list[str]] = {}
    subquestions: dict[str, str] = {}
    with Path(relation_labels).open(encoding="utf-8") as lines:
        for line in lines:
            prop, name, _ = json.loads(line)
            if used is not None and int(prop[1:]) not in used:
                continue
            escaped = name.replace("{", "{{").replace("}", "}}")
            template = OPEN_TEMPLATE.format(escaped)
            if template not in rel2ids:
                rels.append(template)
                rel2ids[template] = []
                subquestions[template] = OPEN_SUBQUESTION.format(escaped)
            if prop not in rel2ids[template]:
                rel2ids[template].append(prop)
    return Bank({}, rels, [], {}, {}, masked=False, rel2ids=rel2ids, subquestions=subquestions)


def catalogue_bank(dataset: list[dict]) -> Bank:
    """MQuAKE's relation catalogue (the all-edit bank's templates and id2rel) without its
    facts or entities, for questions whose edits a KG store holds (--conditions without
    --relations)."""
    full = build_bank(dataset, {case["case_id"] for case in dataset})
    return Bank({}, full.rels, [], full.id2rel, {}, masked=False)


def subject_catalogue(bank: Bank, subject: str) -> tuple[list[str], Any]:
    """The templates, and their embeddings, a hop's relation is retrieved among for its
    subject (--relations-of-subject): those of a relation of an entity the subject names
    in the bank's KG store (read as the bank's lookup reads it, through its KG version);
    the whole catalogue when the subject names no entity."""
    if bank.lookup is None:
        raise ValueError("narrowing the catalogue to the subject's relations needs a KG store")
    if not bank.lookup.names(subject):
        return bank.rels, bank.rel_emb
    props = {f"P{p}" for p in bank.lookup.relations(subject)}
    kept = [i for i, template in enumerate(bank.rels) if props & set(template_ids(bank, template))]
    return [bank.rels[i] for i in kept], bank.rel_emb[kept]


def template_ids(bank: Bank, template: str) -> list[str]:
    """The relation ids of a template: an open catalogue's, else those id2rel maps to it
    (process_kg gives each template one)."""
    if bank.rel2ids is not None:
        return bank.rel2ids.get(template, [])
    return sorted(p for p, t in bank.id2rel.items() if t == template)


def build_bank(dataset: list[dict], edited: set[int], masked: bool = True) -> Bank:
    """The bank of the edited cases' edits (process_kg) and the aliases of every case."""
    from xgraph.gwalk_upstream.mquake_dataset import process_kg
    from xgraph.gwalk_upstream.mquake_utils import get_ent_alias

    kg, rels, ents, id2rel = process_kg(dataset, edited)
    return Bank(kg, rels, ents, id2rel, get_ent_alias(dataset), masked=masked)


def world_banks(dataset: list[dict], worlds: dict[str, int]) -> dict[int, Bank]:
    """Per case of the worlds map (python -m xgraph.mquake --worlds: mquake-<case> to
    world), the bank of its world's edits as xgraph.mquake.worlds builds the world's KG:
    the members' edits first, then every other case's edits but those that rewrite a hop
    of a member's edited chain to another object. Unmasked: no edit left in a world
    rewrites a member's path, so the gold-path mask has nothing to hide."""
    members: dict[int, set[int]] = {}
    for key, world in worlds.items():
        members.setdefault(world, set()).add(int(key.split("-")[1]))
    banks = {}
    for cases in members.values():
        chain: dict[tuple[str, str], set[str]] = {}
        for case in dataset:
            if case["case_id"] in cases:
                for s, p, o in case["new_triples"]:
                    chain.setdefault((s, p), set()).add(o)
        ordered = [c for c in dataset if c["case_id"] in cases]
        for case in dataset:
            if case["case_id"] not in cases:
                kept = [
                    rewrite
                    for rewrite, (s, p, o) in zip(
                        case["requested_rewrite"], case["edit_triples"], strict=True
                    )
                    if not chain.get((s, p), set()) - {o}
                ]
                ordered.append({**case, "requested_rewrite": kept})
        bank = build_bank(ordered, {c["case_id"] for c in ordered}, masked=False)
        banks.update({case: bank for case in cases})
    return banks


class KGLookup:
    """A hop's fact from a KG store, for a hop the bank has no fact for (--kg-db, a
    variant): GWalk reads the edited KG, as path mode does, instead of only the edits.

    The hop's subject (the label the loop carries) names KG entities by label or alias,
    NFC and case folded (KGStore.items_named); its relation template is the relation id
    the bank's id2rel maps to it (MQuAKE's Wikidata property). Of the named entities with
    an object for that property, the best-connected one (most triples, KGStore.degree,
    then the smallest id) is the subject: a homonym without the fact is passed over; of
    its objects, the best-connected one likewise. The fact is the template filled with
    the subject and the object's canonical label, as a bank fact is. Lookups are cached
    and serialized: the threads share the store's one connection. store: the store at
    path read another way, such as through a KG version (toolkg.VersionedStore); lock: a
    lock shared with other lookups of the same connection (one per KG version).
    """

    def __init__(self, path: str | Path, store: Any = None, lock: Any = None):
        from xgraph.toolkg import KGStore

        self.path = Path(path)
        self.store = KGStore(self.path) if store is None else store
        self._lock = threading.Lock() if lock is None else lock
        self._found: dict[tuple[str, str], tuple[int, int, str] | None] = {}
        self._named: dict[str, list[int]] = {}
        self._relations: dict[str, set[int]] = {}

    def fact(
        self, subject: str, template: str, id2rel: dict[str, str]
    ) -> tuple[str, str, dict] | None:
        """The fact sentence, the object's label, and the triple's ids, or None when the
        template has no relation id or the KG holds no such fact."""
        props = sorted(p for p, t in id2rel.items() if t == template)
        return self.fact_for(subject, template, props[:1])  # process_kg: one id a template

    def fact_for(
        self, subject: str, template: str, props: list[str]
    ) -> tuple[str, str, dict] | None:
        """fact, with the template's relation ids given: the first of them, in order, the
        KG holds such a fact for (an open catalogue's name can stand for several)."""
        for prop in props:
            if not re.fullmatch(r"P\d+", prop):
                raise ValueError(f"relation id {prop!r} of {template!r} is not a Wikidata property")
            with self._lock:
                if (subject, prop) not in self._found:
                    self._found[(subject, prop)] = self._lookup(subject, int(prop[1:]))
                found = self._found[(subject, prop)]
            if found is not None:
                item, obj, label = found
                triple = {"subject": f"Q{item}", "relation": prop, "object": f"Q{obj}"}
                return f"{template.format(subject)} {label}", label, triple
        return None

    def names(self, subject: str) -> list[int]:
        """The entities the subject names (label or alias)."""
        with self._lock:
            if subject not in self._named:
                self._named[subject] = self.store.items_named(subject)
            return self._named[subject]

    def relations(self, subject: str) -> set[int]:
        """The relations of the entities the subject names, as the store reads them."""
        named = self.names(subject)
        with self._lock:
            if subject not in self._relations:
                self._relations[subject] = set(self.store.relations_of(named))
            return self._relations[subject]

    def _lookup(self, subject: str, prop: int) -> tuple[int, int, str] | None:
        with_fact = {}
        for item in self.store.items_named(subject):
            objects = self.store.objects_of([item], prop)
            if objects:
                with_fact[item] = objects
        if not with_fact:
            return None
        item = min(with_fact, key=lambda i: (-self.store.degree(i), i))
        objects = with_fact[item]
        obj = min(objects, key=lambda o: (-self.store.degree(o), o))
        return item, obj, objects[obj]


def kg_store_paths(kg_db: str, worlds: dict[str, int] | None) -> dict[int | None, Path]:
    """The KG store of each world (`{}` in kg_db stands for the world), or the one store
    (key None) without worlds; every store must exist."""
    if worlds is None:
        if "{}" in kg_db:
            raise SystemExit(f"--kg-db {kg_db}: '{{}}' stands for a world, which needs --worlds")
        paths: dict[int | None, Path] = {None: Path(kg_db)}
    else:
        if "{}" not in kg_db:
            raise SystemExit(
                f"--kg-db {kg_db}: with --worlds, the path holds '{{}}' for the world, so "
                "each case reads its own world's KG store"
            )
        paths = {world: Path(kg_db.format(world)) for world in sorted(set(worlds.values()))}
    for world, path in paths.items():
        if not path.is_file():
            whose = "the" if world is None else f"world {world}'s"
            raise SystemExit(f"--kg-db: {path}, {whose} KG store, is missing")
    return paths


def attach_kg(
    bank: Bank | dict[int, Bank], kg_db: str, worlds: dict[str, int] | None
) -> list[KGLookup]:
    """Give each case's bank its KG store (--kg-db): the one store, or with worlds the
    store of the case's world. Returns the stores' lookups."""
    paths = kg_store_paths(kg_db, worlds)
    lookups = {world: KGLookup(path) for world, path in paths.items()}
    if worlds is None:
        for one in bank.values() if isinstance(bank, dict) else [bank]:
            one.lookup = lookups[None]
        return list(lookups.values())
    if not isinstance(bank, dict):
        raise ValueError("with worlds, each case has its world's bank (world_banks)")
    for key, world in worlds.items():
        case = int(key.split("-")[1])
        if case not in bank:
            continue
        mine = bank[case]
        if mine.lookup is not None and mine.lookup is not lookups[world]:
            raise ValueError(f"case {case}'s bank is shared with another world's cases")
        mine.lookup = lookups[world]
    return list(lookups.values())


def condition_cases(conditions: dict, limit: int | None = None) -> list[dict]:
    """The questions of an interventions build (python -m xgraph.interventions,
    conditions.json) as GWalk cases (--conditions): one case per condition, with its
    question as the only question, its KG version, the record id the interventions score
    reads (<question>@<condition>, or the condition's own record_id, as python -m
    xgraph.kgversions conditions writes it for MQuAKE questions), and the question as its
    seed key, so that the conditions of a question draw the same samples where their
    prompts agree. limit: the first N questions only."""
    cases = []
    for question in conditions["questions"][:limit]:
        for condition in question["conditions"]:
            name = f"{question['id']}@{condition['name']}"
            cases.append(
                {
                    "case_id": name,
                    "questions": [question["utterance"]],
                    "kg_version": condition.get("version"),
                    "record_ids": [condition.get("record_id", name)],
                    "seed_key": question["id"],
                }
            )
    return cases


def condition_banks(
    bank: Bank,
    cases: list[dict],
    kg_db: str,
    kg_versions: str | Path | None,
    relations_of_subject: bool = False,
) -> dict[str, Bank]:
    """Each condition case's bank: the catalogue of bank (open_bank or catalogue_bank, its
    embeddings shared) with a lookup of the KG store kg_db read through the case's KG
    version (toolkg.VersionedStore; the store itself for a case without one), all lookups
    sharing the store's connection and one lock."""
    from xgraph.toolkg import KGStore, version_stores

    path = Path(kg_db)
    if not path.is_file():
        raise SystemExit(f"--kg-db: {kg_db}, the KG store, is missing")
    names = {case["kg_version"] for case in cases if case["kg_version"]}
    if names and kg_versions is None:
        raise SystemExit("the conditions read KG versions: give their version file, --kg-versions")
    base = KGStore(path)
    stores: dict[str | None, Any] = {None: base}
    if names:
        try:
            stores.update(version_stores(base, kg_versions, names))
        except ValueError as error:
            raise SystemExit(str(error)) from error
    lock = threading.Lock()
    lookups = {name: KGLookup(path, store, lock) for name, store in stores.items()}
    return {
        case["case_id"]: replace(
            bank, lookup=lookups[case["kg_version"]], relations_of_subject=relations_of_subject
        )
        for case in cases
    }


def store_identity(path: Path) -> dict:
    """A KG store's path, size, and modification time (UTC): too large to hash for every
    run, it is identified by these and rebuilt by python -m xgraph.toolkg build."""
    from datetime import UTC, datetime

    stat = path.stat()
    modified = datetime.fromtimestamp(stat.st_mtime, UTC).isoformat(timespec="seconds")
    return {"path": str(path), "bytes": stat.st_size, "modified": modified}


def case_records(case: dict, trace: dict, calls: list[dict], task_prompt: str) -> list[dict]:
    """One record per question, its id mquake-<case>-<k> or the case's record_ids
    (condition_cases): the walk's transcript (the answer prompt after the few-shot),
    ending with the answer line when the walk reached an answer."""
    records = []
    ids = case.get("record_ids") or [
        f"mquake-{case['case_id']}-{k}" for k in range(len(trace["questions"]))
    ]
    for k, walked in enumerate(trace["questions"]):
        prompt = walked["prompt"]
        if prompt.startswith(task_prompt):
            transcript = prompt[len(task_prompt) :].strip()
        else:  # not expected: the loop only appends to the prompt
            transcript = prompt.strip()
        answer = walked["answer"]
        mine = [c for c in calls if c["question"] == k]
        records.append(
            {
                "id": ids[k],
                "arm": "gwalk",
                "text": transcript + (f"\nAnswer: ⟦ {answer}⟧" if answer else ""),
                "answer": answer,
                "subject": trace["subject"],
                "relations": walked["relations"],
                "hops": walked["hops"],
                "calls": len(mine),
                "prompt_tokens": sum(c["prompt_tokens"] for c in mine),
                "generated_tokens": sum(c["generated_tokens"] for c in mine),
                "error": walked["error"],
            }
        )
    return records


def run_case(
    case: dict,
    edited: set[int],
    bank: Bank,
    prompts: dict[str, str],
    model: CaseModel,
    contriever: Any,
    tokenizer: Any,
    repair_answers: bool = False,
    complete_answers: bool = False,
    lookup_first: bool = False,
) -> tuple[dict, list[dict]]:
    """Upstream's loop on one case: its trace and its records. repair_answers and
    complete_answers: give the loop repair_answer (--repair-answers) and complete_answer
    (--complete-answers); a bank with a KG store (--kg-db) gives it the store's lookup;
    lookup_first: look each hop up before its answer call (--hop-order lookup-first). An
    open catalogue (open_bank) gives the loop its subquestions, and a bank narrowed to the
    subject's relations its subject_catalogue."""
    from xgraph.gwalk_upstream.eval_loop import gwalk_case

    kg_fact = None
    if bank.lookup is not None:
        lookup = bank.lookup

        def kg_fact(subject: str, template: str, _: dict) -> tuple[str, str, dict] | None:
            return lookup.fact_for(subject, template, template_ids(bank, template))

    trace = gwalk_case(
        case, case["case_id"] in edited, prompts, model, contriever, tokenizer, bank.kg,
        bank.rels, bank.ents, bank.id2rel, bank.rel_emb, bank.ent_emb, bank.ent2alias, RULES,
        mask=bank.masked, repair=repair_answer if repair_answers else None,
        complete=complete_answer if complete_answers else None,
        kg_fact=kg_fact, lookup_first=lookup_first, subquestions=bank.subquestions,
        catalogue=(lambda subject: subject_catalogue(bank, subject))
        if bank.relations_of_subject else None,
    )  # fmt: skip
    return trace, case_records(case, trace, model.calls, prompts["fill_out_ga_w_blank2"])


def run(
    cases: list[dict],
    edited: set[int],
    bank: Bank | dict[int, Bank],
    client: Completions,
    contriever: Any,
    tokenizer: Any,
    *,
    seed: int = SEED,
    greedy: bool = False,
    workers: int = 64,
    repair_answers: bool = False,
    complete_answers: bool = False,
    lookup_first: bool = False,
) -> tuple[list[dict], list[dict], list[dict]]:
    """Every case through upstream's loop, workers cases at a time: the records in case
    order, the traces, and every call. bank: one bank, or each case's (world_banks);
    repair_answers: --repair-answers; complete_answers: --complete-answers; lookup_first:
    --hop-order lookup-first."""
    prompts = load_prompts()
    models = [
        CaseModel(client, case.get("seed_key", case["case_id"]), seed, greedy) for case in cases
    ]
    done = [0]
    lock = threading.Lock()

    def one(i: int) -> tuple[dict, list[dict]]:
        mine = bank[cases[i]["case_id"]] if isinstance(bank, dict) else bank
        result = run_case(
            cases[i], edited, mine, prompts, models[i], contriever, tokenizer,
            repair_answers, complete_answers, lookup_first,
        )  # fmt: skip
        with lock:
            done[0] += 1
            if done[0] % 50 == 0 or done[0] == len(cases):
                print(f"gwalk: {done[0]}/{len(cases)} cases", file=sys.stderr, flush=True)
        return result

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(one, i) for i in range(len(cases))]
        try:
            results = [future.result() for future in futures]
        except BaseException:
            pool.shutdown(cancel_futures=True)
            raise
    traces = [trace for trace, _ in results]
    records = [record for _, rows in results for record in rows]
    calls = [call for model in models for call in model.calls]
    return records, traces, calls


def paper_measure(cases: list[dict], traces: list[dict], edited: set[int]) -> dict:
    """The paper's measure: a case is correct when upstream's check_answer (upper-cased
    exact match with the answer or an alias, the new answer for edited cases) accepts the
    answer of any of its questions."""
    from xgraph.gwalk_upstream.mquake_dataset import check_answer

    tally = {"edited": [0, 0], "unedited": [0, 0]}
    for case, trace in zip(cases, traces, strict=True):
        flag = case["case_id"] in edited
        right = any(check_answer(flag, case, q["answer"]) for q in trace["questions"])
        group = tally["edited" if flag else "unedited"]
        group[0] += right
        group[1] += 1
    result = {
        group: {"cases": n, "correct": right, "accuracy": right / n if n else None}
        for group, (right, n) in tally.items()
    }
    total = sum(n for _, n in tally.values())
    correct = sum(right for right, _ in tally.values())
    result["all"] = {
        "cases": total,
        "correct": correct,
        "accuracy": correct / total if total else None,
    }
    return result


def summarize(records: list[dict], traces: list[dict], calls: list[dict]) -> dict:
    """Calls, tokens, and finish reasons per stage, parse failures, relation retrievals,
    repaired and completed answers, bank facts, and KG facts (--kg-db)."""
    from xgraph.cost import distributions, request_cost

    hops = [hop for record in records for hop in record["hops"]]
    stages = sorted(set(STAGES.values()))
    return {
        "cases": len(traces),
        "questions": len(records),
        "distributions": distributions(map(request_cost, records)),
        "answered": sum(bool(r["answer"]) for r in records),
        "hops": len(hops),
        "calls": {
            **{s: sum(c["stage"] == s for c in calls) for s in stages},
            "total": len(calls),
        },
        "prompt_tokens": {
            **{s: sum(c["prompt_tokens"] for c in calls if c["stage"] == s) for s in stages},
            "total": sum(c["prompt_tokens"] for c in calls),
        },
        "generated_tokens": {
            **{s: sum(c["generated_tokens"] for c in calls if c["stage"] == s) for s in stages},
            "total": sum(c["generated_tokens"] for c in calls),
        },
        "finish_reasons": dict(sorted(Counter(str(c["finish_reason"]) for c in calls).items())),
        "finish_reasons_by_stage": {
            s: dict(sorted(Counter(str(c["finish_reason"]) for c in mine).items()))
            for s in stages
            if (mine := [c for c in calls if c["stage"] == s])
        },
        "parse_failures": {
            "breakdown_questions": sum(r["error"] == "breakdown" for r in records),
            "rel2subq_hops": sum(h.get("failed") == "rel2subq" for h in hops),
            "answer_hops": sum(h.get("failed") == "answer" for h in hops),
        },
        "relation_retrieval": {
            "hits": sum(h["template"] is not None for h in hops),
            "misses": sum(h["template"] is None for h in hops),
        },
        "repaired_hops": sum(bool(h.get("repaired")) for h in hops),
        "completed_hops": sum(bool(h.get("completed")) for h in hops),
        "looked_up_first_hops": sum(bool(h.get("looked_up_first")) for h in hops),
        "bank_facts": sum(bool(h.get("contradicted")) and "kg_fact" not in h for h in hops),
        "kg_facts": sum("kg_fact" in h for h in hops),
    }


def load_contriever(name: str, revision: str) -> tuple[Any, Any]:
    """The contriever and its tokenizer on the GPU (upstream's initialize_contriever, at
    a pinned revision)."""
    from transformers import AutoModel, AutoTokenizer

    contriever = AutoModel.from_pretrained(name, revision=revision).to("cuda")
    return contriever, AutoTokenizer.from_pretrained(name, revision=revision)


def contriever_weights(name: str, revision: str) -> Path:
    """The contriever's weights in the Hugging Face cache, for provenance."""
    from huggingface_hub import try_to_load_from_cache

    for filename in ("pytorch_model.bin", "model.safetensors"):
        path = try_to_load_from_cache(name, filename, revision=revision)
        if isinstance(path, str):
            return Path(path)
    raise SystemExit(
        f"{name} at revision {revision} is not in the Hugging Face cache; fetch it with "
        f"`hf download {name} --revision {revision}`"
    )


def check_args(args: argparse.Namespace) -> None:
    """The combinations of options a run accepts (SystemExit otherwise)."""
    if args.conditions is None:
        if args.data is None:
            raise SystemExit("--questions asks MQuAKE cases: give their dataset, --data")
        given = [
            f for f in ("relations", "relations_of_subject", "kg_versions") if getattr(args, f)
        ]
        if given:
            flags = ", ".join("--" + f.replace("_", "-") for f in given)
            raise SystemExit(f"{flags}: only with --conditions")
        return
    if args.worlds is not None:
        raise SystemExit("--conditions asks each condition in its KG version, not in --worlds")
    if args.kg_db is None or "{}" in args.kg_db:
        raise SystemExit("--conditions reads one KG store, --kg-db (without '{}')")
    if args.relations is None and args.data is None:
        raise SystemExit("--conditions needs a relation catalogue: --relations, or --data")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="python -m xgraph.gwalk", description=__doc__)
    parser.add_argument(
        "--data",
        type=Path,
        help="MQuAKE-Remastered JSON: the cases and their bank; with --conditions, only the "
        "relation catalogue, where --relations gives none",
    )
    asked = parser.add_mutually_exclusive_group(required=True)
    asked.add_argument(
        "--questions", type=Path, help="the asked questions (python -m xgraph.mquake)"
    )
    asked.add_argument(
        "--conditions",
        type=Path,
        help="an interventions build's conditions.json (python -m xgraph.interventions): each "
        "condition is asked as a case of its own, reading its KG version through --kg-db and "
        "--kg-versions, with records <question>@<condition> (condition_cases; a variant)",
    )
    parser.add_argument(
        "--relations",
        type=Path,
        help="with --conditions: a KG's relation_labels.jsonl; the relation catalogue is the "
        "KG's own, a template per name of each relation of the --kg-db store's triples "
        "(open_bank), not MQuAKE's (a variant)",
    )
    parser.add_argument(
        "--relations-of-subject",
        action="store_true",
        help="with --conditions: retrieve a hop's relation template only among the relations "
        "of the entities its subject names in the KG (subject_catalogue; a variant)",
    )
    parser.add_argument(
        "--kg-versions",
        type=Path,
        help="with --conditions: the build's KG version file; each condition reads the --kg-db "
        "store through its version",
    )
    parser.add_argument("--out", type=Path, required=True, help="output records (JSONL)")
    parser.add_argument("--model", required=True, help="the served model's directory and name")
    parser.add_argument("--url", default="http://127.0.0.1:8766", help="the vLLM server")
    parser.add_argument(
        "--limit",
        type=int,
        help="run only the first N asked cases (with --conditions, the first N questions)",
    )
    parser.add_argument("--workers", type=int, default=64, help="cases run at once")
    parser.add_argument("--seed", type=int, default=SEED, help="base of the request seeds")
    parser.add_argument(
        "--greedy", action="store_true", help="greedy decoding for every call (a variant)"
    )
    parser.add_argument(
        "--worlds",
        type=Path,
        help="worlds.json (python -m xgraph.mquake --worlds): each case's bank holds only its "
        "world's edits, without upstream's gold-path mask (a variant)",
    )
    parser.add_argument(
        "--repair-answers",
        action="store_true",
        help="when the line the loop reads after a hop's answer call is no answer line, read "
        "the completion's first line in the few-shot's form '... {}. <answer>' (repair_answer; "
        "a variant)",
    )
    parser.add_argument(
        "--complete-answers",
        action="store_true",
        help="when a hop's answer line is the model's template sentence with an empty slot "
        "('Generated answer: ... {}.'), have the model write the object after '{}. ' in one "
        "more call, stage 'complete' (complete_answer; a variant)",
    )
    parser.add_argument(
        "--kg-db",
        help="a KG store (python -m xgraph.toolkg build): a hop the bank has no fact for is "
        "looked up there, and its object replaces the model's answer as a bank fact does "
        "(KGLookup; a variant); with --worlds, '{}' in the path stands for the case's world",
    )
    parser.add_argument(
        "--hop-order",
        choices=HOP_ORDERS,
        default="answer-first",
        help="answer-first: upstream's order, the model answers each hop and a bank or KG fact "
        "then replaces its answer; lookup-first: each hop is looked up first, and the model "
        "is asked only where no fact is found (a variant)",
    )
    parser.add_argument("--contriever", default=CONTRIEVER)
    parser.add_argument("--contriever-revision", default=CONTRIEVER_REVISION)
    parser.add_argument("--timeout", type=float, default=600.0, help="seconds per request")
    parser.add_argument(
        "--provenance-only",
        action="store_true",
        help="write <out>.provenance.json on the host, without running",
    )
    args = parser.parse_args(argv)
    check_args(args)
    model_dir = Path(args.model)
    worlds = json.loads(args.worlds.read_text(encoding="utf-8")) if args.worlds else None
    if args.conditions:
        stores = {None: Path(args.kg_db)}
    else:
        stores = kg_store_paths(args.kg_db, worlds) if args.kg_db else {}
    inputs = [args.conditions or args.questions] + ([args.data] if args.data else [])
    inputs += [args.worlds] if args.worlds else []
    inputs += [args.relations] if args.relations else []
    if args.kg_versions:
        from xgraph.kgversions import version_files

        inputs += version_files(args.kg_versions)
    variant = {
        "conditions": str(args.conditions) if args.conditions else None,
        "relations": str(args.relations) if args.relations else None,
        "relations_of_subject": args.relations_of_subject,
        "kg_versions": str(args.kg_versions) if args.kg_versions else None,
    }

    if args.provenance_only:
        from xgraph.generate import model_revision
        from xgraph.provenance import provenance

        weights = contriever_weights(args.contriever, args.contriever_revision)
        record = {
            "artifact_kind": "xgraph.gwalk_provenance",
            "version": 1,
            "arm": "gwalk",
            "upstream": {
                "repository": "https://github.com/henryzhongsc/MQuAKE-Remastered",
                "commit": UPSTREAM_COMMIT,
                "license": "CC BY 4.0",
            },
            "model": {"path": str(model_dir), "revision": model_revision(model_dir)},
            "contriever": {"name": args.contriever, "revision": args.contriever_revision},
            "seed": args.seed,
            "greedy": args.greedy,
            "worlds": str(args.worlds) if args.worlds else None,
            "repair_answers": args.repair_answers,
            "complete_answers": args.complete_answers,
            "hop_order": args.hop_order,
            "kg_db": args.kg_db,
            "kg_stores": [store_identity(path) for path in stores.values()],
            **variant,
            "limit": args.limit,
            "workers": args.workers,
            "provenance": provenance([*inputs, model_dir / "generation_config.json", weights]),
        }
        args.out.parent.mkdir(parents=True, exist_ok=True)
        Path(f"{args.out}.provenance.json").write_text(json.dumps(record, indent=1) + "\n")
        return

    from tokenizers import Tokenizer

    dataset = []
    if args.data:
        data = json.loads(args.data.read_text(encoding="utf-8"))
        dataset = [upstream_case(case) for case in data]
    by_id = {case["case_id"]: case for case in dataset}
    if args.conditions:
        conditions = json.loads(args.conditions.read_text(encoding="utf-8"))
        cases = condition_cases(conditions, args.limit)
        edited: set = set()  # no bank edits: each condition's KG version holds its edit
    else:
        asked = asked_case_ids(json.loads(args.questions.read_text(encoding="utf-8")))
        missing = [c for c in asked if c not in by_id]
        if missing:
            raise SystemExit(f"{args.questions}: cases {missing[:5]} are not in {args.data}")
        cases = [by_id[c] for c in asked[: args.limit]]
        edited = set(by_id)  # all-edit: every case's edits are in the bank (rand_list_3k_all)

    generation_config = json.loads((model_dir / "generation_config.json").read_text())
    eos = generation_config.get("eos_token_id", [])
    eos = [eos] if isinstance(eos, int) else list(eos)
    served = Tokenizer.from_file(str(model_dir / "tokenizer.json"))
    dot = served.token_to_id(".")
    if dot is None:
        raise SystemExit(f"{model_dir}/tokenizer.json has no '.' token for the done stop rule")
    rules = stop_rules(dot, eos)
    sampling = sampling_defaults(generation_config)
    bias = thinking_bias(served.token_to_id)
    client = Completions(
        args.url, args.model, sampling, rules, {dot: "."}, args.timeout, logit_bias=bias
    )
    server = get_json(f"{client.url}/version", args.timeout)

    from xgraph.gwalk_upstream.mquake_utils import get_sent_embeddings

    if args.conditions:
        if args.relations:
            from xgraph.toolkg import KGStore

            base = open_bank(args.relations, KGStore(args.kg_db).used_relations())
        else:
            base = catalogue_bank(dataset)
        bank: Bank | dict = base
        unique = [base]
    elif worlds is not None:
        banks = world_banks(dataset, worlds)
        bank = banks
        unique = list({id(b): b for b in banks.values()}.values())
        missing = [c["case_id"] for c in cases if c["case_id"] not in banks]
        if missing:
            raise SystemExit(f"{args.worlds}: cases {missing[:5]} are in no world")
    else:
        bank = build_bank(dataset, edited)
        unique = [bank]
    if args.kg_db and not args.conditions:
        attach_kg(bank, args.kg_db, worlds)
    contriever, tokenizer = load_contriever(args.contriever, args.contriever_revision)
    for one in unique:
        one.rel_emb = get_sent_embeddings(one.rels, contriever, tokenizer)
        if one.ents:  # an open or catalogue-only bank has none to fit subjects onto
            one.ent_emb = get_sent_embeddings(one.ents, contriever, tokenizer)
    if args.conditions:  # after the embeddings, which the condition banks share
        bank = condition_banks(
            unique[0], cases, args.kg_db, args.kg_versions, args.relations_of_subject
        )
    lock = threading.Lock()
    shared = Serialized(contriever, lock), Serialized(tokenizer, lock)

    started = time.perf_counter()
    records, traces, calls = run(
        cases, edited, bank, client, *shared, seed=args.seed, greedy=args.greedy,
        workers=args.workers, repair_answers=args.repair_answers,
        complete_answers=args.complete_answers, lookup_first=args.hop_order == "lookup-first",
    )  # fmt: skip
    seconds = time.perf_counter() - started
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as out:
        for record in records:
            out.write(json.dumps(record, ensure_ascii=False) + "\n")
    summary = {
        "artifact_kind": "xgraph.gwalk",
        "version": 2,
        "arm": "gwalk",
        "upstream_commit": UPSTREAM_COMMIT,
        "edited_cases": len(edited),
        "banks": [
            {
                "pairs": sum(len(r) for r in one.kg.values()),
                "entities": len(one.ents),
                "relations": len(one.rels),
                "masked": one.masked,
                "kg_db": str(one.lookup.path) if one.lookup is not None else None,
            }
            for one in unique
        ],
        **summarize(records, traces, calls),
        "paper_measure": None if args.conditions else paper_measure(cases, traces, edited),
        **variant,
        "seconds": seconds,
        "timing_kind": "workload_wall",
        "cache_state": "unknown",  # an externally managed server
        "workers": args.workers,
        "seed": args.seed,
        "greedy": args.greedy,
        "worlds": str(args.worlds) if args.worlds else None,
        "repair_answers": args.repair_answers,
        "complete_answers": args.complete_answers,
        "hop_order": args.hop_order,
        "kg_db": args.kg_db,
        "kg_stores": [store_identity(path) for path in stores.values()],
        "sampling": sampling,
        "stop_rules": rules,
        "answer_line_rules": sorted(LINE_RULES),
        "logit_bias": bias,
        "model": args.model,
        "url": args.url,
        "server_version": server.get("version"),
        "contriever": {"name": args.contriever, "revision": args.contriever_revision},
        # the order of the bank's relations and entities (sets upstream) follows it
        "python_hash_seed": os.environ.get("PYTHONHASHSEED"),
    }
    Path(f"{args.out}.summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    print(json.dumps({k: summary[k] for k in ("cases", "questions", "answered", "seconds")}))


if __name__ == "__main__":
    main()
