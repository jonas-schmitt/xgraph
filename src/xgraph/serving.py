"""Serving cost of a gate arm (`docs/quality-gate.md`, run setup).

Sends chat requests to a running vLLM OpenAI-compatible server at a fixed
concurrency, streaming each response to time its first token, and reports per
request the time to first token, the end-to-end latency, and the token counts the
server reports, plus percentiles and output throughput over the run; a response
without its token usage is an error. Decoding is the gate's: greedy, thinking off, and
with --opt-in (the constrained arm) the engine's per-request arguments xgraph.generate
sends (request_arguments): the opt-in, a request's `anchor` and `kg_version`, and with
`prefill_replay: true` its final assistant message as a replayed prefill (xgraph_prefill).
With --open, every request ends with the span opener of the KG index the server was
started with (--kg-index) as a final assistant message and starts inside a span
(xgraph_open), as `python -m xgraph.generate --open` does. A final assistant message is
continued. The server's own engine settings (XGRAPH_PATHS, XGRAPH_ESCAPES, ...) and KG
versions (XGRAPH_KG_VERSIONS) are set where it starts (xgraph.vllm), and must match the
evaluated run's for the cost to be that system's.

    python -m xgraph.serving --url http://127.0.0.1:8000 --model /model \\
        --requests target/serving/requests.jsonl --concurrency 16 --opt-in \\
        --out target/serving/constrained-16.json
"""

from __future__ import annotations

import argparse
import json
import time
import urllib.request
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from xgraph.cost import DEDICATED_GPU_SECONDS_ESTIMATOR
from xgraph.generate import (
    CONTINUED,
    load_requests,
    opened_requests,
    replayed_prefill,
    request_arguments,
)

SAMPLING_FIELDS = frozenset(
    (
        "temperature",
        "top_p",
        "repetition_penalty",
        "logit_bias",
        "stop",
        "stop_token_ids",
        "ignore_eos",
        "min_tokens",
        "skip_special_tokens",
        "spaces_between_special_tokens",
        "include_stop_str_in_output",
    )
)


def _one(
    url: str,
    model: str,
    request: dict,
    max_tokens: int,
    opt_in: bool,
    opener: str | None = None,
    *,
    timeout: float = 600,
    extra_args: dict | None = None,
    structured_outputs: dict | None = None,
    return_token_ids: bool = False,
) -> dict:
    started = time.perf_counter()
    arm = "constrained" if opt_in else "baseline"
    arguments = request_arguments(arm, request, opened=opener is not None and opt_in)
    prefill = replayed_prefill(request)
    previous = request["messages"][-1] if request["messages"] else {}
    display_prefix = previous.get("content", "") if previous.get("role") == "assistant" else ""
    if opener is not None:
        request = opened_requests([request], opener)[0]
    body: dict = {
        "model": model,
        "messages": request["messages"],
        "temperature": 0,
        "max_tokens": max_tokens,
        "stream": True,
        "stream_options": {"include_usage": True},
        "chat_template_kwargs": {"enable_thinking": False},
    }
    sampling = request.get("sampling", {})
    if not isinstance(sampling, dict) or sampling.keys() - SAMPLING_FIELDS:
        raise ValueError(f"request {request['id']!r}: unsupported sampling fields: {sampling!r}")
    body.update(sampling)
    if request["messages"] and request["messages"][-1]["role"] == "assistant":
        body.update(CONTINUED)  # an opener or a prefill is continued, as in xgraph.generate
    if extra_args is not None:
        arguments = {**(arguments or {}), **extra_args}
    if arguments:
        # OpenAI vllm_xargs values allow scalars and scalar lists, not objects.
        # Keep the offline LLM API arguments and caller's records unchanged.
        for field in ("xgraph_path_options", "gcr"):
            if isinstance(arguments.get(field), dict):
                arguments = {**arguments, field: json.dumps(arguments[field], allow_nan=False)}
        body["vllm_xargs"] = arguments
    if structured_outputs is not None:
        body["structured_outputs"] = structured_outputs
    if return_token_ids:
        body["return_token_ids"] = True
    if "seed" in request:
        body["seed"] = request["seed"]
    http = urllib.request.Request(
        url.rstrip("/") + "/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    first = None
    pieces: list[str] = []
    usage: dict = {}
    finish_reason = None
    token_ids, engine_metrics, request_id = [], None, None
    done = False
    with urllib.request.urlopen(http, timeout=timeout) as response:
        for raw in response:
            line = raw.decode("utf-8").strip()
            if line == "data: [DONE]":
                done = True
                break
            if not line.startswith("data: "):
                continue
            chunk = json.loads(line[len("data: ") :])
            request_id = chunk.get("id") or request_id
            engine_metrics = chunk.get("metrics") or engine_metrics
            if chunk.get("error"):
                raise ValueError(f"request {request['id']!r}: streaming error: {chunk['error']}")
            usage = chunk.get("usage") or usage
            for choice in chunk.get("choices") or []:
                token_ids.extend(choice.get("token_ids") or [])
                finish_reason = choice.get("finish_reason") or finish_reason
                content = (choice.get("delta") or {}).get("content")
                if content:
                    first = first if first is not None else time.perf_counter()
                    pieces.append(content)
    ended = time.perf_counter()
    if not done:
        raise ValueError(f"request {request['id']!r}: incomplete stream (no [DONE])")
    counts = {name: usage.get(name) for name in ("prompt_tokens", "completion_tokens")}
    if any(type(count) is not int or count < 0 for count in counts.values()):
        raise ValueError(
            f"request {request['id']!r}: the server reported no token usage ({usage or 'none'}); "
            "its throughput cannot be counted"
        )
    if return_token_ids and (
        len(token_ids) != counts["completion_tokens"]
        or any(type(t) is not int or t < 0 for t in token_ids)
    ):
        raise ValueError(f"request {request['id']!r}: exact generated token IDs are missing")
    return {
        "id": request["id"],
        "text": (opener or prefill or display_prefix) + "".join(pieces),
        "ttft": first - started if first is not None else None,
        "latency": ended - started,
        "finish_reason": finish_reason,
        "seed": request.get("seed"),
        "max_tokens": max_tokens,
        "request_id": request_id,
        "engine_metrics": engine_metrics,
        "cached_prompt_tokens": (usage.get("prompt_tokens_details") or {}).get("cached_tokens"),
        **({"token_ids": token_ids} if return_token_ids else {}),
        **counts,
        **({"opener": opener} if opener is not None else {}),
        **({"prefill": prefill} if prefill else {}),
    }


def measure(
    url: str,
    model: str,
    requests: list[dict],
    *,
    concurrency: int,
    max_tokens: int,
    opt_in: bool,
    opener: str | None = None,
) -> list[dict]:
    """One result per request, in request order, with concurrency requests in flight. Each
    request carries the engine's per-request arguments its fields give, as in
    xgraph.generate (request_arguments: anchor, KG version, replayed prefill), with opt_in;
    with an opener (the span opener), each starts inside a span (a forced opener)."""
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        return list(pool.map(lambda r: _one(url, model, r, max_tokens, opt_in, opener), requests))


def summarize(results: list[dict], wall_seconds: float) -> dict:
    """Percentiles and throughput; a result without its token counts is refused."""
    if missing := [r.get("id") for r in results if r["completion_tokens"] is None
                   or r["prompt_tokens"] is None]:  # fmt: skip
        raise ValueError(f"results {missing} have no token counts")
    ttft = np.array([r["ttft"] for r in results if r["ttft"] is not None])
    latency = np.array([r["latency"] for r in results])
    tokens = sum(r["completion_tokens"] for r in results)
    return {
        "requests": len(results),
        "wall_seconds": wall_seconds,
        "ttft_p50": float(np.percentile(ttft, 50)) if len(ttft) else None,
        "ttft_p90": float(np.percentile(ttft, 90)) if len(ttft) else None,
        "latency_p50": float(np.percentile(latency, 50)),
        "latency_p90": float(np.percentile(latency, 90)),
        "prompt_tokens": sum(r["prompt_tokens"] for r in results),
        "completion_tokens": tokens,
        "output_tokens_per_second": tokens / wall_seconds,
    }


def measure_workload(
    requests: list[dict],
    *,
    concurrency: int,
    question: Callable[[dict], dict],
    gpu_count: int = 1,
    progress: Callable[[int, int, dict], None] | None = None,
) -> tuple[list[dict], dict]:
    """Admit a finite panel, then serve at most concurrency whole questions at once.

    Latency and visible-content TTFT include client queueing. A question callback
    executes every model/tool call synchronously; independent questions progress
    asynchronously. No retries are hidden. Failed questions remain in the report.
    """
    if not requests or concurrency <= 0 or gpu_count <= 0:
        raise ValueError("a workload needs requests, positive concurrency and GPU count")
    started = time.perf_counter()
    cpu_started = time.process_time()

    def run(request, admitted):
        service_started = time.perf_counter()
        try:
            result = {**question(request), "status": "completed"}
        except Exception as error:
            result = {
                "id": request["id"],
                "status": "failed",
                "error": str(error),
                "ttft": None,
                "prompt_tokens": None,
                "completion_tokens": None,
                "model_calls": 0,
                "tool_calls": 0,
                **getattr(error, "result", {}),
            }
        ended = time.perf_counter()
        queue = service_started - admitted
        if result.get("ttft") is not None:
            result["ttft"] += queue
        result.update(
            admitted_seconds=admitted - started,
            client_queue_seconds=queue,
            service_seconds=ended - service_started,
            latency=ended - admitted,
        )
        return result

    pool = ThreadPoolExecutor(max_workers=concurrency)
    futures = []
    try:
        for request in requests:
            futures.append(pool.submit(run, request, time.perf_counter()))
        if progress is not None:
            for count, future in enumerate(as_completed(futures), 1):
                progress(count, len(requests), future.result())
        results = [future.result() for future in futures]
    except BaseException as error:
        pool.shutdown(wait=True, cancel_futures=True)
        error.results = [
            f.result() for f in futures if not f.cancelled() and f.done() and f.exception() is None
        ]
        raise
    else:
        pool.shutdown(wait=True)
    wall = time.perf_counter() - started
    service = sum(r["service_seconds"] for r in results)
    # Queueing is already charged to the question occupying the reserved GPU.
    # Serial dispatch gaps are shared equally. Batched costs are an allocation,
    # not independent GPU-busy measurements, and conserve the measured interval.
    for result in results:
        attributed = (
            result["service_seconds"] + (wall - service) / len(results)
            if concurrency == 1
            else wall * result["service_seconds"] / service
        )
        result["gpu_seconds"] = gpu_count * attributed
    complete = [r for r in results if r["status"] == "completed"]
    summary = {
        "requests": len(requests),
        "completed_questions": len(complete),
        "failures": len(results) - len(complete),
        "truncations": sum(
            r.get("truncated", r.get("finish_reason") == "length") for r in complete
        ),
        "wall_seconds": wall,
        "gpu_seconds": gpu_count * wall,
        "gpu_seconds_estimator": DEDICATED_GPU_SECONDS_ESTIMATOR,
        "gpu_attribution": "serial" if concurrency == 1 else "service_weighted",
        "client_cpu_seconds": time.process_time() - cpu_started,
        "cpu_scope": "load_generator_process_all_threads; excludes server/workers",
        "completed_questions_per_second": len(complete) / wall,
        "correct_answers_per_second": None,
        "ttft_boundary": "question admission to first visible content of first model call",
        "latency_boundary": "question admission through final model/tool/delivery completion",
        "retries": sum(r.get("retries", 0) for r in results),
    }
    for metric in ("latency", "ttft"):
        values = [r[metric] for r in complete if r.get(metric) is not None]
        summary.update(
            {
                f"{metric}_p{p}": float(np.percentile(values, p)) if values else None
                for p in (50, 95, 99)
            }
        )
    for key in ("prompt_tokens", "completion_tokens", "model_calls", "tool_calls"):
        values = [r.get(key) for r in results]
        known = [v for v in values if v is not None]
        summary[key] = sum(known) if len(known) == len(results) else None
        summary[f"{key}_observed_total"] = sum(known) if known else None
        summary[f"{key}_known_questions"] = len(known)
    summary["ttft_known_questions"] = sum(r.get("ttft") is not None for r in complete)
    summary["output_tokens_per_second"] = (
        summary["completion_tokens"] / wall if summary["completion_tokens"] is not None else None
    )
    return results, summary


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="python -m xgraph.serving", description=__doc__)
    parser.add_argument("--url", required=True, help="the server's base URL")
    parser.add_argument("--model", required=True, help="the served model name")
    parser.add_argument("--requests", type=Path, required=True, help="chat requests (JSONL)")
    parser.add_argument("--concurrency", type=int, required=True)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--opt-in", action="store_true", help="constrain with the engine")
    parser.add_argument(
        "--open",
        action="store_true",
        help="start every answer inside a span: the span opener as a prefilled assistant turn",
    )
    parser.add_argument(
        "--kg-index", type=Path, help="with --open: the KG index the server was started with"
    )
    parser.add_argument("--out", type=Path, required=True, help="results (JSON)")
    args = parser.parse_args(argv)
    opener = None
    if args.open:
        if not args.opt_in:
            parser.error("--open needs --opt-in: a forced opener needs the engine")
        if args.kg_index is None:
            parser.error("--open needs --kg-index, whose span syntax names the opener")
        from xgraph.index import saved_span_syntax

        opener = saved_span_syntax(args.kg_index).opener
    requests = load_requests(args.requests)
    started = time.perf_counter()
    results = measure(
        args.url,
        args.model,
        requests,
        concurrency=args.concurrency,
        max_tokens=args.max_tokens,
        opt_in=args.opt_in,
        opener=opener,
    )
    summary = summarize(results, time.perf_counter() - started)
    report = {
        "artifact_kind": "xgraph.serving_cost",
        "version": 1,
        "concurrency": args.concurrency,
        "opt_in": args.opt_in,
        "opener": opener,
        "kg_index": str(args.kg_index) if args.kg_index else None,
        "summary": summary,
        "results": results,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=1, ensure_ascii=False) + "\n")
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
