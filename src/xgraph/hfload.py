"""Loading a bf16 model for a forward pass with bounded host memory.

transformers' from_pretrained on the GB10 holds the weights in host memory while it
fills the model on the GPU, and the safetensors shards fill the page cache; on
unified memory a 27B model then needs about twice its size and was killed (exit 137).
Where a checkpoint's tensor names are the model's parameter names (as for
Qwen3.8-27B, whose extra multi-token-prediction tensors are skipped), the model is
built on the GPU from its config and each shard is read straight to the GPU, one
tensor at a time, so host memory stays small. Otherwise from_pretrained is used.
Runs where the model is (the pinned vLLM container).
"""

from __future__ import annotations

import json
from pathlib import Path


def load_model(model_dir: str | Path):
    """The model in bf16 on the GPU, in eval mode."""
    import torch
    from safetensors import safe_open
    from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForImageTextToText

    model_dir = Path(model_dir)
    config = AutoConfig.from_pretrained(model_dir)
    try:
        auto = AutoModelForImageTextToText
        with torch.device("meta"):
            skeleton = auto.from_config(config, dtype=torch.bfloat16)
    except ValueError:
        auto = AutoModelForCausalLM
        with torch.device("meta"):
            skeleton = auto.from_config(config, dtype=torch.bfloat16)
    wanted = set(skeleton.state_dict())
    del skeleton
    index = model_dir / "model.safetensors.index.json"
    shards = (
        sorted(set(json.loads(index.read_text())["weight_map"].values()))
        if index.exists()
        else ["model.safetensors"]
    )
    names = {}
    for shard in shards:
        with safe_open(model_dir / shard, "pt", device="cpu") as f:
            names.update({name: shard for name in f.keys()})  # noqa: SIM118 (not a dict)
    tied = getattr(config, "tie_word_embeddings", False) or getattr(
        config.get_text_config(), "tie_word_embeddings", False
    )
    missing = wanted - set(names) - ({"lm_head.weight"} if tied else set())
    if missing:
        print(f"{model_dir}: {sorted(missing)[:3]} named apart from the checkpoint, "
              "loading through from_pretrained", flush=True)  # fmt: skip
        model = auto.from_pretrained(model_dir, dtype=torch.bfloat16, device_map="cuda")
        return model.eval()
    print(f"{model_dir}: loading {len(wanted)} tensors straight to the GPU", flush=True)
    with torch.device("cuda"):
        model = auto.from_config(config, dtype=torch.bfloat16)
    params = model.state_dict()
    with torch.no_grad():
        for shard in shards:
            with safe_open(model_dir / shard, "pt", device="cuda") as f:
                for name in f.keys():  # noqa: SIM118 (not a dict)
                    if name in wanted:
                        params[name].copy_(f.get_tensor(name))
    if tied:  # the output head is the input embedding
        model.tie_weights()
    return model.eval()
