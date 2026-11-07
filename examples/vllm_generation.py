"""Generate against the toy index inside vLLM's GPU sampling loop.

Requires a CUDA build of xgraph and the repository's vLLM Model Runner V2 interface.
Build the toy index using this model's tokenizer.json first. See examples/README.md.
--dry-run prints the full configuration on CPU without importing vLLM or loading
weights. Tests also execute the generation call with a stand-in runtime.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--max-model-len", type=int, default=2048)
    parser.add_argument("--kv-cache-gib", type=int, default=8)
    args = parser.parse_args()
    config = {
        "environment": {
            "VLLM_USE_V2_MODEL_RUNNER": "1",
            "XGRAPH_KG_INDEX": str(args.index.resolve()),
            "XGRAPH_TOKENIZER": str(args.tokenizer.resolve()),
            "XGRAPH_PATHS": "1",
            "XGRAPH_COMMITTED_PATHS": "1",
            "XGRAPH_BOUND_ESCAPES": "0",
            "XGRAPH_END_AFTER_ANSWER": "1",
        },
        "llm": {
            "model": args.model,
            "logits_processors": ["xgraph.vllm:XGraphLogitsProcessor"],
            "max_model_len": args.max_model_len,
            "max_num_seqs": 1,
            "gpu_memory_utilization": 0.45,
            "kv_cache_memory_bytes": args.kv_cache_gib * 1024**3,
        },
    }
    if args.dry_run:
        print(json.dumps(config, indent=2))
        return
    generate(config)


def generate(config):
    """The real worker runs this call; CPU tests supply an interface stand-in."""
    os.environ.update(config["environment"])
    from vllm import LLM, SamplingParams

    llm = LLM(**config["llm"])
    outputs = llm.chat(
        [
            {
                "role": "system",
                "content": (
                    "Answer by writing a KG path and then an entity answer. "
                    "Format: ⟦ start · relation · relation⟧\nAnswer: ⟦ endpoint⟧. "
                    "Use the graph's exact labels. Relations include creator and born in. "
                    "If a path cannot continue, write ⁇ inside the span to escape."
                ),
            },
            {"role": "user", "content": "Where was the creator of Atlas born?"},
        ],
        sampling_params=SamplingParams(
            temperature=0,
            seed=0,
            max_tokens=128,
            extra_args={"xgraph": 1, "xgraph_anchor": ["item:atlas"]},
        ),
    )
    print(outputs[0].outputs[0].text)


if __name__ == "__main__":
    main()
