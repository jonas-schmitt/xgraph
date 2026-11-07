"""Data-only SubgraphRAG retrieval and frozen-reader preparation."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import time
from pathlib import Path

import numpy as np

from xgraph.method_prompts import PromptSource
from xgraph.provenance import provenance, sha256_file

PROMPT_NAMES = ("icl_sys_prompt", "icl_user_prompt", "icl_ass_prompt", "icl_cot_prompt")
MODELS = {"qwen38": ("Qwen3.8-27B-FP8", 55, 8), "gemma31": ("gemma-4-31B-it-FP8-dynamic", 75, 16)}


def queue_plan(requests, prompts, out, models, lengths, *, max_tokens=256, seed=17):
    """Construct commands and explicit cost scenarios. This function cannot launch jobs."""
    if set(lengths) != set(MODELS) or type(max_tokens) is not int or max_tokens < 1:
        raise ValueError("pilot needs both frozen model families and a positive completion cap")
    requests, prompts, out, models = map(
        lambda p: Path(p).resolve(), (requests, prompts, out, models)
    )
    repo = Path(__file__).resolve().parents[2]
    families, low, high, counts = {}, 0.0, 0.0, set()
    for family, (name, headroom, kv) in MODELS.items():
        first_lengths, retry_lengths = lengths[family]["first"], lengths[family]["retry"]
        n = len(first_lengths)
        if (
            not n
            or len(retry_lengths) != n
            or any(
                type(v) is not int or v <= 0 or v + max_tokens > 16384
                for v in (*first_lengths, *retry_lengths)
            )
        ):
            raise ValueError(f"{family}: invalid prompt lengths or context overflow")
        counts.add(n)
        # Planning assumptions, not measurements or confidence bounds. The low
        # scenario uses one short call; the high uses both calls at their caps.
        low += sum(first_lengths) / 3000 + n * min(64, max_tokens) / 12 + 60
        high += (sum(first_lengths) + sum(retry_lengths)) / 1000 + 2 * n * max_tokens / 3 + 600
        directory = out / family
        first, retry, retry_out, merged = [
            directory / filename
            for filename in ("first.jsonl", "retry-requests.jsonl", "retry.jsonl", "answers.jsonl")
        ]

        def generate(
            request_file,
            output_file,
            *,
            name=name,
            headroom=headroom,
            kv=kv,
            ban_channel=family == "gemma31",
        ):
            command = [
                "env",
                f"XG_NEED_GIB={headroom}",
                "XG_GPU_IDLE=1",
                "XG_LORA=",
                "XG_PATHS=0",
                "XG_OPEN=0",
                "XG_AUDIT=0",
                "XG_MENU=0",
                "nice",
                "-n",
                "10",
                str(repo / "scripts/generate.sh"),
                str(models / name),
                str(request_file),
                "baseline",
                str(output_file),
                "--max-tokens",
                str(max_tokens),
                "--max-model-len",
                "16384",
                "--max-num-seqs",
                "8",
                "--kv-cache-gib",
                str(kv),
                "--seed",
                str(seed),
            ]
            return command + (["--ban-tokens", "<|channel>"] if ban_channel else [])

        cpu = ["nice", "-n", "10", "uv", "run", "--no-sync", "python", "-m", "xgraph.subgraphrag"]
        families[family] = {
            "first": generate(requests, first),
            "prepare_retry": [
                *cpu,
                "retry",
                "--requests",
                str(requests),
                "--first",
                str(first),
                "--prompts",
                str(prompts),
                "--out",
                str(retry),
            ],
            "retry": generate(retry, retry_out),
            "retry_condition": "nonempty_retry_requests",
            "merge": [
                *cpu,
                "merge",
                "--requests",
                str(requests),
                "--first",
                str(first),
                "--second",
                str(retry_out),
                "--out",
                str(merged),
            ],
        }
    if len(counts) != 1:
        raise ValueError("families must price the same question cohort")
    return {
        "status": "dry_run_only",
        "questions": counts.pop(),
        "seed": seed,
        "families": families,
        "max_tokens_per_call": max_tokens,
        "price": {
            "reader_gpu_hours_low": low / 3600,
            "reader_gpu_hours_high": high / 3600,
            "retriever_gpu_hours": None,
            "retriever_cost_status": "unknown_offline_release",
            "assumptions": {
                "prefill_tokens_per_second": [1000, 3000],
                "decode_tokens_per_second": [3, 12],
                "startup_seconds": [60, 300],
                "low_calls_per_question": 1,
                "high_calls_per_question": 2,
                "low_completion_tokens": min(64, max_tokens),
            },
            "within_10_reader_gpu_hours": high <= 36000,
            "scope": (
                "unmeasured reader-only scenarios; includes up to four model startups; "
                "excludes retrieval training, embedding and dedicated serving qualification"
            ),
        },
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare", help="CPU-only reader requests from published scores")
    prepare.add_argument("--scores", type=Path, required=True)
    prepare.add_argument("--panel", type=Path)
    prepare.add_argument(
        "--pilot", type=int, default=0, help="seeded cost-only subset; zero keeps all"
    )
    prepare.add_argument("--top-k", type=int, default=100)
    prepare.add_argument("--seed", type=int, default=17)
    prepare.add_argument("--prompts", type=Path, required=True)
    prepare.add_argument("--out", type=Path, required=True)
    retry = commands.add_parser("retry", help="CPU-only released clarification requests")
    merge = commands.add_parser("merge", help="CPU-only complete usage and answer delivery")
    for command in (retry, merge):
        command.add_argument("--requests", type=Path, required=True)
        command.add_argument("--first", type=Path, required=True)
        command.add_argument("--out", type=Path, required=True)
    retry.add_argument("--prompts", type=Path, required=True)
    merge.add_argument("--second", type=Path)
    score = commands.add_parser("score", help="CPU-only native scorer from cached GTE embeddings")
    for flag in ("checkpoint", "sample", "embeddings", "out"):
        score.add_argument("--" + flag, type=Path, required=True)
    score.add_argument("--top-k", type=int, default=500)
    score.add_argument("--batch-size", type=int, default=4096)
    plan = commands.add_parser("plan", help="preview only; no execution mode exists")
    for flag in ("requests", "prompts", "out", "models"):
        plan.add_argument("--" + flag, type=Path, required=True)
    plan.add_argument("--dry-run", action="store_true", required=True)
    plan.add_argument("--max-tokens", type=int, default=256)
    plan.add_argument("--seed", type=int, default=17)
    args = parser.parse_args(argv)
    inputs, extra, started = [], {}, time.monotonic()
    if args.command == "plan":
        from transformers import AutoTokenizer

        from xgraph.gcr import chat_token_ids
        from xgraph.generate import model_revision

        requests = _read_lines(args.requests)
        prompts = PromptSource.load(args.prompts, PROMPT_NAMES)
        retry = retry_requests(requests, [{"id": r["id"], "text": ""} for r in requests], prompts)
        lengths, revisions = {}, {}
        inputs = [args.requests, args.prompts]
        for family, (name, _, _) in MODELS.items():
            model = args.models / name
            tokenizer = AutoTokenizer.from_pretrained(
                model, local_files_only=True, trust_remote_code=False
            )
            lengths[family] = {
                stage: [len(chat_token_ids(tokenizer, r["messages"])) for r in rows]
                for stage, rows in (("first", requests), ("retry", retry))
            }
            revisions[family] = {"model": str(model.resolve()), "revision": model_revision(model)}
            inputs.extend(
                model / name for name in ("tokenizer.json", "tokenizer_config.json", "config.json")
            )
        report = queue_plan(
            args.requests,
            args.prompts,
            args.out,
            args.models,
            lengths,
            max_tokens=args.max_tokens,
            seed=args.seed,
        )
        report.update(models=revisions, prompt_token_lengths=lengths, provenance=provenance(inputs))
        print(json.dumps(report, indent=2))
        return
    summary = args.out.with_suffix(args.out.suffix + ".summary.json")
    if args.out.exists() or summary.exists():
        raise ValueError(f"output already exists: {args.out} or {summary}")
    if args.command == "prepare":
        inputs = [args.scores, args.prompts]
        if args.scores.suffix == ".jsonl":
            scored = _by_id(_read_lines(args.scores))
        else:
            scored = load_tensor_data(args.scores)
        prompts = PromptSource.load(args.prompts, PROMPT_NAMES)
        if args.panel:
            inputs.append(args.panel)
            panel = json.loads(args.panel.read_text())
        else:
            panel = [
                {"id": qid, "utterance": row["question"]} for qid, row in sorted(scored.items())
            ]
        if not 0 <= args.pilot <= len(panel):
            raise ValueError("pilot count must be nonnegative and fit the question panel")
        if args.pilot:
            panel = sorted(
                panel, key=lambda r: hashlib.sha256(f"{args.seed}:{r['id']}".encode()).digest()
            )[: args.pilot]
        rows = prepare_requests(scored, prompts, panel, top_k=args.top_k)
        extra = {
            "top_k": args.top_k,
            "seed": args.seed,
            "selected_ids": [r["id"] for r in rows],
            "selection": "cost_only_test_pilot" if args.pilot else "full_supplied_panel",
        }
        if any(r["subgraphrag"]["kind"] == "native_scorer_cached_embeddings" for r in rows):
            extra["retriever_cost_status"] = "cached_embeddings_full_preparation_unpriced"
    elif args.command == "score":
        inputs = [args.checkpoint, args.sample, args.embeddings]
        rows = [_score_sample(args)]
        extra = {
            "top_k": args.top_k,
            "batch_size": args.batch_size,
            "retriever_cost_status": "scoring_only_embeddings_unpriced",
        }
    else:
        inputs = [args.requests, args.first]
        requests, first = _read_lines(args.requests), _read_lines(args.first)
        if args.command == "retry":
            inputs.append(args.prompts)
            rows = retry_requests(requests, first, PromptSource.load(args.prompts, PROMPT_NAMES))
        else:
            if args.second and args.second.exists():
                inputs.append(args.second)
            second = _read_lines(args.second) if args.second and args.second.exists() else []
            rows = merge_reader_records(requests, first, second)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")
    summary.write_text(
        json.dumps(
            {
                "stage": args.command,
                "questions": len(rows),
                "seconds": time.monotonic() - started,
                "retriever_cost_status": "unknown_offline_release",
                **extra,
                "provenance": provenance(inputs),
            },
            indent=2,
        )
        + "\n"
    )


def _read_lines(path):
    with Path(path).open(encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def _score_sample(args):
    sample = json.loads(args.sample.read_text())
    scorer = TripleScorer.load(args.checkpoint)
    raw_triples = list(
        zip(sample["h_id_list"], sample["r_id_list"], sample["t_id_list"], strict=True)
    )
    triples = np.asarray(raw_triples) if raw_triples else np.empty((0, 3), dtype=np.int64)
    entities = sample["text_entity_list"] + sample["non_text_entity_list"]
    relations = sample["relation_list"]
    if args.top_k < 1:
        raise ValueError("top_k must be positive")
    with np.load(args.embeddings, allow_pickle=False) as embeddings:
        if len(embeddings["entity_embs"]) != len(sample["text_entity_list"]) or len(
            embeddings["relation_embs"]
        ) != len(relations):
            raise ValueError("embedding rows differ from the supplied entity or relation lists")
        logits = scorer.score(
            triples=triples,
            topics=sample["q_entity_id_list"],
            question=embeddings["q_emb"],
            entities=embeddings["entity_embs"],
            num_non_text=len(sample["non_text_entity_list"]),
            relations=embeddings["relation_embs"],
            batch_size=args.batch_size,
        )
    exponent = np.exp(-np.abs(logits))
    scores = np.where(logits >= 0, 1 / (1 + exponent), exponent / (1 + exponent))
    order = np.lexsort((np.arange(len(scores)), -scores))[: args.top_k]
    return {
        "id": sample["id"],
        "question": sample["question"],
        "scored_triples": [
            [
                entities[triples[i, 0]],
                relations[triples[i, 1]],
                entities[triples[i, 2]],
                float(scores[i]),
            ]
            for i in order
        ],
        "scorer": scorer.metadata(),
    }


def retry_requests(requests, first, prompts):
    """Released sys_icl_dc clarification, with no failed reply in the retry history."""
    originals, replies = _by_id(requests), _by_id(first)
    if originals.keys() != replies.keys():
        raise ValueError("SubgraphRAG first outputs must be complete and match requests")
    return [
        dict(
            request,
            messages=[
                *request["messages"],
                {"role": "user", "content": prompts.strings["icl_cot_prompt"]},
            ],
        )
        for qid, request in originals.items()
        if _needs_retry(replies[qid]["text"])
    ]


def merge_reader_records(requests, first, second):
    """Deliver answer lines while retaining discarded completions and all usage."""
    originals, replies, retries = _by_id(requests), _by_id(first), _by_id(second)
    if originals.keys() != replies.keys():
        raise ValueError("SubgraphRAG first outputs must be complete and match requests")
    expected = {qid for qid, row in replies.items() if _needs_retry(row["text"])}
    if retries.keys() != expected:
        raise ValueError("SubgraphRAG retry outputs must exactly match required retries")
    result = []
    for qid, request in originals.items():
        turns = [replies[qid]] + ([retries[qid]] if qid in retries else [])
        prompt_tokens = generated_tokens = 0
        for turn in turns:
            prompt, generated = (
                turn["prompt_tokens"],
                turn.get("generated_tokens", len(turn["token_ids"])),
            )
            if any(type(n) is not int or n < 0 for n in (prompt, generated)):
                raise ValueError(f"question {qid}: missing or invalid token usage")
            if turn.get("model_calls", 1) != 1:
                raise ValueError(f"question {qid}: each reader stage must be one model call")
            prompt_tokens += prompt
            generated_tokens += generated
        text = turns[-1]["text"]
        answers = list(
            dict.fromkeys(
                answer.strip()
                for answer in re.findall(r"^\s*ans:\s*([^\n]+)", text, re.I | re.M)
                if answer.strip().casefold() not in ("not available", "no information available")
            )
        )
        result.append(
            {
                **turns[-1],
                "arm": "subgraphrag-cached-embeddings"
                if request["subgraphrag"]["kind"] == "native_scorer_cached_embeddings"
                else "subgraphrag-replay",
                "answers": answers,
                "model_calls": len(turns),
                "prompt_tokens": prompt_tokens,
                "generated_tokens": generated_tokens,
                "turns": turns,
                "subgraphrag": request["subgraphrag"],
            }
        )
    return result


def _by_id(rows):
    result = {}
    for row in rows:
        qid = row["id"]
        if not isinstance(qid, str) or qid in result:
            raise ValueError(f"invalid or duplicate question id {qid!r}")
        result[qid] = row
    return result


def _needs_retry(text):
    lower = text.lower()
    return (
        "ans:" not in lower
        or "ans: not available" in lower
        or "ans: no information available" in lower
    )


def prepare_requests(scored, prompts, panel, *, top_k=100):
    """Released scored-K prompt; no gold fields, answer-present filter or graph expansion."""
    if type(top_k) is not int or top_k < 1:
        raise ValueError("top_k must be a positive integer")
    result, seen = [], set()
    strings = prompts.strings
    for row in panel:
        qid, question = row["id"], row["utterance"]
        if not isinstance(qid, str) or qid in seen:
            raise ValueError(f"invalid or duplicate panel id {qid!r}")
        if not isinstance(question, str) or not question.strip():
            raise ValueError(f"question {qid}: empty question")
        seen.add(qid)
        sample = scored.get(qid)
        if sample is not None and sample["question"].strip() != question.strip():
            raise ValueError(f"question {qid}: retrieval question differs from panel")
        triples, seen_triples = [], set()
        for triple in sample["scored_triples"] if sample is not None else []:
            if len(triple) != 4 or any(not isinstance(t, str) or not t for t in triple[:3]):
                raise ValueError(f"question {qid}: invalid scored triple")
            if (
                not isinstance(triple[3], (int, float))
                or not math.isfinite(triple[3])
                or not 0 <= triple[3] <= 1
            ):
                raise ValueError(f"question {qid}: invalid triple probability")
            value = tuple(triple[:3])
            if value not in seen_triples:
                triples.append(value)
                seen_triples.add(value)
                if len(triples) == top_k:
                    break
        text = "Triplets:\n" + "\n".join(f"({s},{r},{o})" for s, r, o in triples)
        text += "\n\nQuestion:\n" + question + ("" if question.endswith("?") else "?")
        result.append(
            {
                "id": qid,
                "question": question,
                "messages": [
                    {"role": "system", "content": strings["icl_sys_prompt"]},
                    {"role": "user", "content": strings["icl_user_prompt"]},
                    {"role": "assistant", "content": strings["icl_ass_prompt"]},
                    {"role": "user", "content": text},
                ],
                "subgraphrag": {
                    "kind": "native_scorer_cached_embeddings"
                    if sample is not None and "scorer" in sample
                    else "released_test_retrieval_replay",
                    "top_k": top_k,
                    "triples": len(triples),
                    "prompts": prompts.metadata(),
                    **(
                        {"scorer": sample["scorer"]}
                        if sample is not None and "scorer" in sample
                        else {}
                    ),
                    "status": "missing_retrieval"
                    if sample is None
                    else "ready"
                    if triples
                    else "empty_retrieval",
                },
            }
        )
    return result


def load_tensor_data(path):
    """Restricted CPU deserialization; never allowlist pickle globals or retry unsafely."""
    import pickle

    import torch

    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except (OSError, RuntimeError, ValueError, pickle.UnpicklingError) as error:
        raise ValueError(f"restricted tensor data {path}: {error}") from error


class TripleScorer:
    """CPU fp32 inference in the released feature order, without PyG or upstream code."""

    def __init__(self, state, config, path):
        self.state, self.config, self.path = state, config, Path(path)
        self.dimension = state["non_text_entity_emb.weight"].shape[1]

    @classmethod
    def load(cls, path):
        import torch

        checkpoint = load_tensor_data(path)
        try:
            config = checkpoint["config"]["retriever"]
            state = checkpoint["model_state_dict"]
            topic_pe = config["topic_pe"]
            forward = config["DDE_kwargs"]["num_rounds"]
            reverse = config["DDE_kwargs"]["num_reverse_rounds"]
            if type(topic_pe) is not bool or any(
                type(n) is not int or not 0 <= n <= 64 for n in (forward, reverse)
            ):
                raise ValueError("invalid DDE configuration")
            d = state["non_text_entity_emb.weight"].shape[1]
            expected = {
                "non_text_entity_emb.weight": (1, d),
                "pred.0.weight": (d, 4 * d + 4 * topic_pe + 4 * (forward + reverse)),
                "pred.0.bias": (d,),
                "pred.2.weight": (1, d),
                "pred.2.bias": (1,),
            }
            if set(state) != set(expected) or d <= 0:
                raise ValueError("unexpected weight keys or embedding dimension")
            for key, shape in expected.items():
                tensor = state[key]
                if (
                    type(tensor) is not torch.Tensor
                    or tensor.dtype != torch.float32
                    or tuple(tensor.shape) != shape
                    or not torch.isfinite(tensor).all()
                ):
                    raise ValueError(f"{key} must be finite fp32 with shape {shape}")
            return cls({key: tensor.numpy() for key, tensor in state.items()}, config, path)
        except (KeyError, IndexError, TypeError, AttributeError, ValueError) as error:
            raise ValueError(f"SubgraphRAG checkpoint {path}: {error}") from error

    def score(
        self, *, triples, topics, question, entities, num_non_text, relations, batch_size=4096
    ):
        d = self.dimension
        for name, array in (
            ("question", question),
            ("entities", entities),
            ("relations", relations),
        ):
            if (
                not isinstance(array, np.ndarray)
                or array.dtype != np.float32
                or array.ndim != 2
                or array.shape[1] != d
                or not np.isfinite(array).all()
            ):
                raise ValueError(f"{name} must be finite fp32 with shape (N, {d})")
        if (
            question.shape[0] != 1
            or type(num_non_text) is not int
            or num_non_text < 0
            or type(batch_size) is not int
            or batch_size < 1
        ):
            raise ValueError(
                "scoring needs one question, nonnegative non-text count and positive batch size"
            )
        triples = np.asarray(triples)
        counts = self.config["DDE_kwargs"]
        positional = directional_encoding(
            len(entities) + num_non_text,
            triples,
            topics,
            forward_rounds=counts["num_rounds"],
            reverse_rounds=counts["num_reverse_rounds"],
        )
        if not self.config["topic_pe"]:
            positional = positional[:, 2:]
        if np.any((triples[:, 1] < 0) | (triples[:, 1] >= len(relations))):
            raise ValueError("relation id outside the relation embedding table")
        node_text = np.concatenate(
            (entities, np.repeat(self.state["non_text_entity_emb.weight"], num_non_text, axis=0))
        )
        nodes = np.concatenate((node_text, positional), axis=1)
        logits = np.empty(len(triples), np.float32)
        for start in range(0, len(triples), batch_size):
            part = triples[start : start + batch_size]
            features = np.concatenate(
                (
                    np.repeat(question, len(part), axis=0),
                    nodes[part[:, 0]],
                    relations[part[:, 1]],
                    nodes[part[:, 2]],
                ),
                axis=1,
            )
            hidden = np.maximum(
                features @ self.state["pred.0.weight"].T + self.state["pred.0.bias"], 0
            )
            logits[start : start + len(part)] = (
                hidden @ self.state["pred.2.weight"].T + self.state["pred.2.bias"]
            ).ravel()
        return logits

    def metadata(self):
        return {
            "path": str(self.path),
            "sha256": sha256_file(self.path),
            "config": self.config,
            "implementation": "numpy_cpu_fp32",
            "embedding_dimension": self.dimension,
        }


def directional_encoding(num_entities, triples, topics, *, forward_rounds=2, reverse_rounds=2):
    """Topic one-hot followed by forward rounds and independent reverse rounds."""
    triples = np.asarray(triples)
    topics = np.asarray(topics)
    if (
        type(num_entities) is not int
        or num_entities < 0
        or triples.ndim != 2
        or triples.shape[1] != 3
        or triples.dtype.kind not in "iu"
        or topics.ndim != 1
        or (topics.size and topics.dtype.kind not in "iu")
        or any(type(n) is not int or n < 0 for n in (forward_rounds, reverse_rounds))
    ):
        raise ValueError("DDE needs nonnegative counts, integer (N, 3) triples and topic ids")
    if any(
        np.any((ids < 0) | (ids >= num_entities)) for ids in (triples[:, 0], triples[:, 2], topics)
    ):
        raise ValueError("DDE entity or topic id outside the entity table")
    triples, topics = triples.astype(np.int64, copy=False), topics.astype(np.int64, copy=False)
    seed = np.zeros((num_entities, 2), dtype=np.float32)
    seed[:, 0] = 1
    seed[topics] = [0, 1]
    columns = [seed]
    for source, target, rounds in (
        (triples[:, 0], triples[:, 2], forward_rounds),
        (triples[:, 2], triples[:, 0], reverse_rounds),
    ):
        state = seed
        degree = np.bincount(target, minlength=num_entities).clip(min=1)[:, None]
        for _ in range(rounds):
            updated = np.zeros_like(seed)
            np.add.at(updated, target, state[source])
            state = (updated / degree).astype(np.float32)
            columns.append(state)
    return np.concatenate(columns, axis=1)


if __name__ == "__main__":
    main()
