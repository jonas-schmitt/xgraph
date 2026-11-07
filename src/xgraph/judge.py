"""The quality gate's judge: gpt-6.1-sol through `codex exec` (`docs/quality-gate.md`).

Every call pins the model and its reasoning effort, runs in an empty directory with
a read-only sandbox and no saved session, and asks for JSON that a schema fixes.
Results are cached by the hash of model, effort, schema, and prompt in a JSON-lines
file, so that a rerun of the gate reuses every decision and a decision can be read
back. A failed call or a malformed answer is retried, then raised.

Two tasks as in FActScore but batched per text, and one for answers:

- `atomic_facts`: split a text (delimiters removed) into short self-contained facts.
- `supported`: decide for each claim whether the given Wikipedia passages support it.
- `same_entity`: decide whether an answer names one of the gold answer entities, so
  that accuracy does not depend on the answer's surface form.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import tempfile
import threading
from collections.abc import Callable
from pathlib import Path

MODEL = "gpt-6.1-sol"
EFFORT = "medium"
ATTEMPTS = 3

SUPPORT_SCHEMA = {
    "type": "object",
    "properties": {"supported": {"type": "array", "items": {"type": "boolean"}}},
    "required": ["supported"],
    "additionalProperties": False,
}
FACTS_SCHEMA = {
    "type": "object",
    "properties": {"facts": {"type": "array", "items": {"type": "string"}}},
    "required": ["facts"],
    "additionalProperties": False,
}

SUPPORT_PROMPT = """\
Decide whether the context from Wikipedia supports each claim about {topic}.
A claim is supported only if the context states it or directly implies it; a claim the
context does not mention, or contradicts, is not supported. Use only the context, not
your own knowledge.

Context:
{context}

Claims:
{claims}

Answer with one decision per claim, in order."""

FACTS_PROMPT = """\
Break the following text about {topic} into atomic facts: short, self-contained
statements that each hold exactly one piece of information, name their subject
instead of using pronouns, and together cover every claim in the text. Skip
statements without factual content, such as opinions or hedges about what is known.

Text:
{text}"""


SAME_SCHEMA = {
    "type": "object",
    "properties": {"same": {"type": "boolean"}},
    "required": ["same"],
    "additionalProperties": False,
}

SAME_PROMPT = """\
Question: {question}

Correct answers (each line is one entity, by its names):
{gold}

Proposed answer: {answer}

Does the proposed answer name one of the correct answer entities? Different spellings,
abbreviations, or fuller or shorter names of the same entity count as the same entity;
a different entity does not, even a closely related one (a city is not its country, an
actor is not the character). Judge by the names given, not by whether you agree with
the correct answers."""


class JudgeError(RuntimeError):
    """The judge did not return a usable answer after its retries."""


class Judge:
    def __init__(
        self, cache: str | Path, runner: Callable[..., subprocess.CompletedProcess] = subprocess.run
    ) -> None:
        self._cache_path = Path(cache)
        self._runner = runner
        self._lock = threading.Lock()
        self._cache: dict[str, dict] = {}
        if self._cache_path.exists():
            with self._cache_path.open(encoding="utf-8") as lines:
                for line in lines:
                    record = json.loads(line)
                    self._cache[record["key"]] = record["result"]

    def supported(self, topic: str, claims: list[str], passages: list[str]) -> list[bool]:
        if not claims:
            return []
        prompt = SUPPORT_PROMPT.format(
            topic=topic,
            context="\n\n".join(passages) if passages else "(no context found)",
            claims="\n".join(f"{n}. {claim}" for n, claim in enumerate(claims, 1)),
        )

        def check(result: dict) -> list[bool]:
            decisions = result["supported"]
            if len(decisions) != len(claims):
                raise ValueError(f"{len(decisions)} decisions for {len(claims)} claims")
            return [bool(d) for d in decisions]

        return self._ask(prompt, SUPPORT_SCHEMA, check)

    def atomic_facts(self, topic: str, text: str) -> list[str]:
        prompt = FACTS_PROMPT.format(topic=topic, text=text)
        return self._ask(prompt, FACTS_SCHEMA, lambda r: [str(f) for f in r["facts"]])

    def same_entity(self, question: str, answer: str, gold: list[list[str]]) -> bool:
        """Whether the answer names one of the gold entities, each given by its names."""
        prompt = SAME_PROMPT.format(
            question=question, answer=answer, gold="\n".join("- " + " / ".join(n) for n in gold)
        )
        return self._ask(prompt, SAME_SCHEMA, lambda r: bool(r["same"]))

    def _ask(self, prompt: str, schema: dict, check: Callable[[dict], object]):
        key = hashlib.sha256(
            json.dumps([MODEL, EFFORT, schema, prompt], sort_keys=True).encode()
        ).hexdigest()
        with self._lock:
            if key in self._cache:
                return check(self._cache[key])
        problem = ""
        for _ in range(ATTEMPTS):
            try:
                result = self._call(prompt, schema)
                value = check(result)
            except (
                ValueError,
                KeyError,
                TypeError,
                json.JSONDecodeError,
                subprocess.TimeoutExpired,
            ) as error:
                problem = str(error)
                continue
            with self._lock:
                self._cache[key] = result
                self._cache_path.parent.mkdir(parents=True, exist_ok=True)
                with self._cache_path.open("a", encoding="utf-8") as out:
                    out.write(json.dumps({"key": key, "result": result}) + "\n")
            return value
        raise JudgeError(f"no usable answer after {ATTEMPTS} attempts: {problem}")

    def _call(self, prompt: str, schema: dict) -> dict:
        with tempfile.TemporaryDirectory(prefix="xgraph-judge-") as work:
            schema_path = Path(work) / "schema.json"
            schema_path.write_text(json.dumps(schema))
            out = Path(work) / "answer.json"
            command = [
                "codex", "exec", "-m", MODEL, "-c", f"model_reasoning_effort={EFFORT}",
                "--sandbox", "read-only", "--skip-git-repo-check", "--ephemeral",
                "--output-schema", str(schema_path), "-o", str(out), "-",
            ]  # fmt: skip
            done = self._runner(
                command, input=prompt, cwd=work, capture_output=True, text=True, timeout=600
            )
            if done.returncode != 0:
                raise ValueError(f"codex exited {done.returncode}: {done.stderr[-300:]}")
            return json.loads(out.read_text(encoding="utf-8"))
