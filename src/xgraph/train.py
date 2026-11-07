"""LoRA fine-tuning on the span format, run inside the vLLM container (scripts/train-lora.sh).

    python -m xgraph.train --model /model --data train.jsonl --out /out/adapter

Trains a LoRA adapter on chat examples (xgraph.sft): the loss covers the reply only,
thinking is off in the chat template as at generation, and the adapter targets the
decoder layers' attention and MLP projections (which vLLM can serve as LoRA), not
the vision tower or the linear-attention projections, in an image-text model (Qwen3.8)
and in a text-only causal LM (Qwen2.5) alike. The bf16 base model is frozen
with gradient checkpointing; examples go one at a time with gradient accumulation,
in an order fixed by the seed. With --span-loss masked, the reply's loss is the
cross-entropy under the constraint's renormalized distribution, from the KG index of
the example's world (xgraph.policy). Path-writing examples (xgraph.pathdata: a path
span, then the answer) are masked in path mode, as XGRAPH_PATHS=1 serves them:
--paths, with --committed-paths for committed paths, and --anchor to start each reply
from its example's `anchor` (QIDs), as anchored requests do. --no-escapes masks without
the escape marker, as XGRAPH_ESCAPES=0 serves; the masks match the served ones only when
these settings match the evaluation's (XG_PATHS, XG_COMMITTED_PATHS, XG_ESCAPES), and
path-mode evaluations run with XG_ESCAPES=0. With --answer-loss
singleton, a path-writing example whose path reaches several entities (its
`endpoint_frontier`, recorded by xgraph.pathdata) carries no loss on its answer span's
label, which names one of them arbitrarily; its path span and the tokens after the label
still train, and the label's tokens still move the constraint's state. Writes the
adapter and train-summary.json (losses per logging interval, settings, the answers left
out, timings).
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path
from typing import Any

IGNORE = -100
TARGETS = (
    r"model\.(?:language_model\.)?layers\.\d+\."
    r"(?:self_attn\.(?:q_proj|k_proj|v_proj|o_proj)|mlp\.(?:gate_proj|up_proj|down_proj))"
)
"""The decoder layers' attention and MLP projections, matched in full by PEFT: under
model.language_model in an image-text model (Qwen3.8), whose vision tower, linear
attention, and MTP head stay out, and under model directly in a causal LM (Qwen2.5)."""


def _ids(tokens: Any) -> list[int]:
    """apply_chat_template returns a list, or a mapping with input_ids, by version."""
    if hasattr(tokens, "keys"):
        tokens = tokens["input_ids"]
    return list(tokens)


def encode(tokenizer: Any, messages: list[dict], max_len: int) -> tuple[list[int], list[int]]:
    """Token ids of a chat example, and labels that ignore everything before the reply.

    An example longer than max_len loses tokens from the front, keeping the reply.
    """
    full = _ids(tokenizer.apply_chat_template(messages, tokenize=True, enable_thinking=False))
    prompt = _ids(
        tokenizer.apply_chat_template(
            messages[:-1], tokenize=True, add_generation_prompt=True, enable_thinking=False
        )
    )
    if full[: len(prompt)] != prompt:
        raise ValueError(
            "the chat template does not render the prompt as a prefix of the conversation"
        )
    labels = [IGNORE] * len(prompt) + full[len(prompt) :]
    return full[-max_len:], labels[-max_len:]


def answer_positions(ids: list[int], labels: list[int], tokenizer: Any, syntax: Any) -> set[int]:
    """The positions of the reply tokens (labels not IGNORE) that write the answer span's
    label: the reply's last span, between its opener and its closer, the label prefix
    included. Found by the tokens' bytes (tokenizer: a ServedTokenizer), so that a token
    that writes part of the label with part of a delimiter counts as the label's; special
    tokens write nothing. Empty when the reply has no span."""
    positions = [p for p, label in enumerate(labels) if label != IGNORE]
    pieces = [tokenizer.token_bytes(ids[p]) or b"" for p in positions]
    text = b"".join(pieces)
    opener = text.rfind(syntax.opener.encode())
    if opener < 0:
        return set()
    start = opener + len(syntax.opener.encode())
    end = text.find(syntax.closer.encode(), start)
    end = len(text) if end < 0 else end
    found, at = set(), 0
    for position, piece in zip(positions, pieces, strict=True):
        if at < end and at + len(piece) > start:
            found.add(position)
        at += len(piece)
    return found


def excluded_positions(
    example: dict, ids: list[int], labels: list[int], tokenizer: Any, answer_loss: str
) -> set[int]:
    """The reply positions the loss leaves out: with answer_loss singleton, the answer
    span's label (answer_positions) when the example's endpoint_frontier (recorded by
    python -m xgraph.pathdata) holds more than one entity, since the reply's answer is
    then one of several the path reaches; nothing otherwise."""
    from xgraph.syntax import SpanSyntax

    if answer_loss == "all":
        return set()
    if "endpoint_frontier" not in example:
        raise ValueError(
            f"example {example.get('id')!r} has no endpoint_frontier (python -m xgraph.pathdata)"
        )
    if example["endpoint_frontier"] <= 1:
        return set()
    return answer_positions(ids, labels, tokenizer, SpanSyntax())


def engines(
    specs: list[str],
    tokenizer: Any,
    *,
    paths: bool = False,
    committed_paths: bool = False,
    escapes: bool = True,
) -> dict[str, Any]:
    """One engine per example world, from WORLD=DIR specs of saved KG indexes: in path
    mode with paths True (a path-writing reply), with committed paths as well when
    committed_paths is True, and without the escape marker when escapes is False. These
    must be the served engine's settings (XGRAPH_PATHS, XGRAPH_COMMITTED_PATHS,
    XGRAPH_ESCAPES), for the loss to normalize over the tokens the deployed policy allows."""
    from xgraph.engine import Engine
    from xgraph.index import load_kg_index

    if committed_paths and not paths:
        raise ValueError("committed paths need path mode (paths=True)")
    built = {}
    for spec in specs:
        world, separator, directory = spec.partition("=")
        if not separator or not world or not directory:
            raise ValueError(f"an index is given as WORLD=DIR, not {spec!r}")
        index = load_kg_index(directory)
        built[world] = Engine(
            index,
            tokenizer,
            index.span_syntax,
            max_sequences=1,
            paths=paths,
            committed_paths=committed_paths,
            escapes=escapes,
        )
    return built


def _masked_step(
    model,
    engine,
    ids: list[int],
    labels: list[int],
    torch,
    anchor: list[str] | None = None,
    excluded: set[int] = frozenset(),
) -> tuple[Any, bool]:
    """The reply's loss under the constraint's renormalized distribution (xgraph.policy),
    the reply bound to the anchor's entities when one is given, over the reply positions
    not excluded (excluded_positions), which still move the constraint's state.

    A reply the constraint would not allow (a target token outside its mask) falls
    back to the ordinary cross-entropy; the caller counts these.
    """
    from xgraph.policy import masked_loss

    logits = model(input_ids=torch.tensor([ids], device="cuda")).logits[0]
    positions = [p for p, label in enumerate(labels) if label != IGNORE and p > 0]
    rows = logits[[p - 1 for p in positions]]
    reply = [labels[p] for p in positions]
    keep = [p not in excluded for p in positions]
    try:
        return masked_loss(engine, rows, reply, anchor, keep=keep), True
    except ValueError:
        kept = [i for i, k in enumerate(keep) if k]
        targets = torch.tensor([reply[i] for i in kept], device=rows.device)
        return torch.nn.functional.cross_entropy(rows[kept].float(), targets), False


def optimizer_batches(examples: int, epochs: int, accum: int, rng) -> list[tuple[int, list[int]]]:
    """The examples of each optimizer step, as (epoch, indices): each epoch shuffled and
    cut into batches of accum, its last batch possibly shorter, so that no batch spans
    two epochs and no example is left out."""
    batches = []
    for epoch in range(epochs):
        order = list(range(examples))
        rng.shuffle(order)
        batches.extend((epoch, order[at : at + accum]) for at in range(0, examples, accum))
    return batches


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="python -m xgraph.train", description=__doc__)
    parser.add_argument("--model", required=True, help="bf16 model directory")
    parser.add_argument("--data", type=Path, required=True, help="chat examples (JSONL)")
    parser.add_argument("--out", type=Path, required=True, help="adapter directory")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--alpha", type=int, default=32)
    parser.add_argument("--accum", type=int, default=8)
    parser.add_argument("--max-len", type=int, default=768)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--targets", default=TARGETS, help="regex of the modules LoRA adapts")
    parser.add_argument(
        "--span-loss",
        choices=["plain", "masked"],
        default="plain",
        help="masked: cross-entropy over the tokens the constraint allows (xgraph.policy)",
    )
    parser.add_argument(
        "--index",
        action="append",
        default=[],
        metavar="WORLD=DIR",
        help="with --span-loss masked: each example world's KG index (default world: original)",
    )
    parser.add_argument(
        "--paths",
        action="store_true",
        help="with --span-loss masked: mask in path mode, for path-writing examples",
    )
    parser.add_argument(
        "--committed-paths", action="store_true", help="with --paths: committed paths"
    )
    parser.add_argument(
        "--no-escapes",
        action="store_true",
        help="with --span-loss masked: mask without the escape marker; must match the "
        "evaluation's XG_ESCAPES (path-mode runs use XG_ESCAPES=0)",
    )
    parser.add_argument(
        "--anchor",
        action="store_true",
        help="with --span-loss masked: start each reply from its example's anchor (QIDs)",
    )
    parser.add_argument(
        "--answer-loss",
        choices=["all", "singleton"],
        default="all",
        help="singleton: no loss on the answer span's label where the example's "
        "endpoint_frontier holds several entities (excluded_positions)",
    )
    args = parser.parse_args(argv)
    if args.span_loss != "masked" and (
        args.paths or args.committed_paths or args.anchor or args.no_escapes
    ):
        parser.error(
            "--paths, --committed-paths, --no-escapes, and --anchor need --span-loss masked"
        )
    if args.committed_paths and not args.paths:
        parser.error("--committed-paths needs --paths")
    examples = [json.loads(line) for line in args.data.read_text().splitlines() if line]
    if args.answer_loss != "all" and (
        unsized := sum("endpoint_frontier" not in e for e in examples)
    ):
        parser.error(
            f"--answer-loss {args.answer_loss}: {unsized} examples have no endpoint_frontier "
            "(python -m xgraph.pathdata frontiers)"
        )

    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoModelForImageTextToText, AutoTokenizer

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    try:  # Qwen3.8 is an image-text model; a text-only model loads as a causal LM
        model = AutoModelForImageTextToText.from_pretrained(
            args.model, dtype=torch.bfloat16, device_map="cuda"
        )
    except ValueError:
        model = AutoModelForCausalLM.from_pretrained(
            args.model, dtype=torch.bfloat16, device_map="cuda"
        )
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    model = get_peft_model(
        model,
        LoraConfig(
            r=args.rank,
            lora_alpha=args.alpha,
            lora_dropout=0.05,
            target_modules=args.targets,
            task_type="CAUSAL_LM",
        ),
    )
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    encoded = [encode(tokenizer, e["messages"], args.max_len) for e in examples]
    worlds = [e.get("world", "original") for e in examples]
    anchors = [e.get("anchor") if args.anchor else None for e in examples]
    if args.anchor and (unanchored := sum(not a for a in anchors)):
        raise SystemExit(f"--anchor: {unanchored} examples have no anchor")
    served = None
    if args.span_loss == "masked" or args.answer_loss != "all":
        from xgraph.tokenizer import ServedTokenizer

        served = ServedTokenizer.from_file(Path(args.model) / "tokenizer.json")
    excluded = [
        excluded_positions(e, ids, labels, served, args.answer_loss)
        for e, (ids, labels) in zip(examples, encoded, strict=True)
    ]
    masks = {}
    if args.span_loss == "masked":
        masks = engines(
            args.index,
            served,
            paths=args.paths,
            committed_paths=args.committed_paths,
            escapes=not args.no_escapes,
        )
        missing = sorted(set(worlds) - set(masks))
        if missing:
            raise SystemExit(f"no --index for example worlds {missing}")
    off_policy = 0
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=0.0
    )
    batches = optimizer_batches(len(encoded), args.epochs, args.accum, random)
    steps = len(batches)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda s: min(1.0, (s + 1) / 20) * max(0.0, 1 - s / max(steps, 1))
    )
    model.train()
    losses, window, step, started = [], [], 0, time.time()
    for epoch, batch in batches:
        for i in batch:
            ids, labels = encoded[i]
            if masks:
                loss, on_policy = _masked_step(
                    model, masks[worlds[i]], ids, labels, torch, anchors[i], excluded[i]
                )
                off_policy += not on_policy
            else:
                kept = [IGNORE if p in excluded[i] else t for p, t in enumerate(labels)]
                loss = model(
                    input_ids=torch.tensor([ids], device="cuda"),
                    labels=torch.tensor([kept], device="cuda"),
                ).loss
            (loss / len(batch)).backward()  # the batch's mean loss
            window.append(loss.item())
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        step += 1
        if step % 10 == 0:
            losses.append({"epoch": epoch, "step": step, "loss": sum(window) / len(window)})
            print(json.dumps(losses[-1]), flush=True)
            window = []
    args.out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(args.out)
    summary = {
        "artifact_kind": "xgraph.lora_training",
        "version": 1,
        "settings": {k: str(v) for k, v in vars(args).items()},
        "targets": args.targets,
        "examples": len(encoded),
        "trainable_parameters": trainable,
        "optimizer_steps": step,
        "span_loss": args.span_loss,
        "paths": args.paths,
        "committed_paths": args.committed_paths,
        "escapes": not args.no_escapes,
        "anchored": args.anchor,
        "answer_loss": args.answer_loss,
        "answers_left_out": sum(bool(e) for e in excluded),
        "worlds": {w: worlds.count(w) for w in sorted(set(worlds))},
        "off_policy_examples": off_policy,
        "seconds": time.time() - started,
        "losses": losses,
    }
    (args.out / "train-summary.json").write_text(json.dumps(summary, indent=1) + "\n")


if __name__ == "__main__":
    main()
