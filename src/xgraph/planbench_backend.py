"""Lazy vLLM backend for the opt-in planning pilot; no model loads at import time."""

from __future__ import annotations

import json
import os
from pathlib import Path

from xgraph.planbench_interfaces import Reply


def model_metadata_paths(model: Path) -> list[Path]:
    """Runtime tokenizer/template/config assets, excluding the large weight shards."""
    required = [
        model / name for name in ("config.json", "generation_config.json", "tokenizer.json")
    ]
    missing = [str(p) for p in required if not p.is_file()]
    if missing:
        raise ValueError(f"missing frozen model metadata: {missing}")
    return sorted(
        {
            p
            for pattern in ("*.json", "*.jinja", "*.txt", "*.model", "*.tiktoken")
            for p in model.glob(pattern)
            if p.is_file()
        }
    )


def engine_kwargs(model, kv_cache_gib, context_tokens, seed, memory_share):
    if kv_cache_gib <= 0 or context_tokens < 1 or not 0 < memory_share <= 1:
        raise ValueError("invalid planning model/cache configuration")
    return {
        "model": str(model),
        "generation_config": "vllm",
        "kv_cache_memory_bytes": int(kv_cache_gib * (1 << 30)),
        "gpu_memory_utilization": memory_share,
        "max_model_len": context_tokens,
        "max_num_seqs": 1,
        "enable_prefix_caching": False,
        "enforce_eager": True,
        "limit_mm_per_prompt": {"image": 0, "video": 0},
        "seed": seed,
        "logits_processors": ["xgraph.planbench_mask:PlanningLogitsProcessor"],
    }


class VLLMBackend:
    def __init__(
        self,
        model: Path,
        budget,
        kv_cache_gib=16,
        seed=16,
        memory_share=0.45,
        *,
        llm=None,
        sampling_class=None,
    ):
        self.seed = seed
        self.model = Path(model)
        os.environ["PLANBENCH_TOKENIZER"] = str(self.model / "tokenizer.json")
        if llm is None:
            import vllm

            sampling_class = vllm.SamplingParams
            llm = vllm.LLM(
                **engine_kwargs(model, kv_cache_gib, budget.context_tokens, seed, memory_share)
            )
        self.llm, self.sampling_class = llm, sampling_class
        self.tokenizer = llm.get_tokenizer()
        eos = []
        for name in ("config.json", "generation_config.json"):
            path = self.model / name
            if path.exists():
                value = json.loads(path.read_text()).get("eos_token_id", [])
                eos.extend([value] if isinstance(value, int) else value or [])
        value = self.tokenizer.eos_token_id
        eos.extend([value] if isinstance(value, int) else value or [])
        self.eos = sorted(set(eos))
        if not self.eos or any(type(t) is not int or t < 0 for t in self.eos):
            raise ValueError("model has no valid explicit EOS ids")

    def _tokens(self, messages):
        return self.tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True, enable_thinking=False
        )

    def count_prompt(self, messages):
        return len(self._tokens(messages))

    def complete(self, messages, max_tokens, *, mask_arguments=None):
        prompt = list(self._tokens(messages))
        params = self.sampling_class(
            temperature=0,
            top_p=1,
            seed=self.seed,
            max_tokens=max_tokens,
            stop_token_ids=self.eos,
            extra_args=mask_arguments,
            skip_special_tokens=True,
        )
        (output,) = self.llm.generate([{"prompt_token_ids": prompt}], params, use_tqdm=False)
        if list(output.prompt_token_ids) != prompt or len(output.outputs) != 1:
            raise ValueError("vLLM changed the admitted prompt or returned multiple completions")
        completion = output.outputs[0]
        return Reply(
            completion.text,
            len(prompt),
            len(completion.token_ids),
            completion.finish_reason,
            tuple(completion.token_ids),
        )
