"""Native CPU or CUDA inference for Stella's packed-QKV, post-norm GTE encoder.

Checkpoint configuration and tensors are data. No checkpoint architecture code is
loaded. The unused CLS pooler is retained so every checkpoint key is checked.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F


class StellaModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        supported = {
            "model_type": "new",
            "pack_qkv": True,
            "hidden_act": "gelu",
            "position_embedding_type": "rope",
            "layer_norm_type": "layer_norm",
        }
        for key, expected in supported.items():
            if config.get(key) != expected:
                raise ValueError(f"Stella requires {key}={expected!r}")
        scaling = config.get("rope_scaling")
        if scaling != {"type": "ntk", "factor": 2.0}:
            raise ValueError("Stella requires fixed NTK RoPE scaling factor 2")
        if config.get("logn_attention_scale", False):
            raise ValueError("Stella logn attention scaling is unsupported")
        hidden, heads = config["hidden_size"], config["num_attention_heads"]
        if hidden % heads or (hidden // heads) % 2:
            raise ValueError("Stella needs even attention head dimensions")
        self.config = config
        self.new = nn.ModuleDict(
            {
                "embeddings": nn.ModuleDict(
                    {
                        "word_embeddings": nn.Embedding(
                            config["vocab_size"], hidden, padding_idx=config["pad_token_id"]
                        ),
                        "token_type_embeddings": nn.Embedding(config["type_vocab_size"], hidden),
                        "LayerNorm": nn.LayerNorm(hidden, eps=config["layer_norm_eps"]),
                    }
                ),
                "encoder": nn.ModuleDict(
                    {
                        "layer": nn.ModuleList(
                            [StellaLayer(config) for _ in range(config["num_hidden_layers"])]
                        )
                    }
                ),
                "pooler": nn.ModuleDict({"dense": nn.Linear(hidden, hidden)}),
            }
        )

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)
        embeddings = self.new["embeddings"]
        hidden = embeddings["LayerNorm"](
            embeddings["word_embeddings"](input_ids)
            + embeddings["token_type_embeddings"](token_type_ids)
        )
        dim = self.config["hidden_size"] // self.config["num_attention_heads"]
        # The upstream cache is initialized at twice the original context length:
        # its fixed NTK frequencies apply even to short input sequences.
        powers = torch.arange(0, dim, 2, dtype=torch.float32, device=hidden.device) / dim
        inverse = (self.config["rope_theta"] * 2) ** (-powers) / 2 ** (2 / dim)
        angles = (
            torch.arange(input_ids.shape[1], dtype=torch.float32, device=hidden.device)[:, None]
            * inverse
        )
        angles = torch.cat((angles, angles), -1)[None, None]
        cosine, sine = angles.cos(), angles.sin()
        for layer in self.new["encoder"]["layer"]:
            hidden = layer(hidden, attention_mask, cosine, sine)
        return hidden


class StellaLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        hidden, intermediate = config["hidden_size"], config["intermediate_size"]
        self.heads = config["num_attention_heads"]
        self.attention = nn.ModuleDict(
            {
                "qkv_proj": nn.Linear(hidden, 3 * hidden),
                "o_proj": nn.Linear(hidden, hidden),
            }
        )
        self.attn_ln = nn.LayerNorm(hidden, eps=config["layer_norm_eps"])
        self.mlp = nn.ModuleDict(
            {
                "up_gate_proj": nn.Linear(hidden, 2 * intermediate, bias=False),
                "down_proj": nn.Linear(intermediate, hidden),
            }
        )
        self.mlp_ln = nn.LayerNorm(hidden, eps=config["layer_norm_eps"])

    def forward(self, hidden, mask, cosine, sine):
        batch, length, width = hidden.shape
        q, k, v = [
            part.reshape(batch, length, self.heads, width // self.heads).transpose(1, 2)
            for part in self.attention["qkv_proj"](hidden).chunk(3, -1)
        ]

        def rotate(value):
            first, second = value.chunk(2, -1)
            # Frequencies/rotation are evaluated in fp32, then SDPA receives a
            # consistent Q/K/V dtype, including when weights are bf16 or fp16.
            return (value.float() * cosine + torch.cat((-second, first), -1).float() * sine).to(
                value.dtype
            )

        context = (
            F.scaled_dot_product_attention(
                rotate(q), rotate(k), v, attn_mask=mask[:, None, None, :].bool(), dropout_p=0.0
            )
            .transpose(1, 2)
            .reshape(batch, length, width)
        )
        hidden = self.attn_ln(hidden + self.attention["o_proj"](context))
        up, gate = self.mlp["up_gate_proj"](hidden).chunk(2, -1)
        return self.mlp_ln(hidden + self.mlp["down_proj"](up * F.gelu(gate)))


class NativeStella:
    """Local tokenizer, masked mean pooling and the checkpoint's dense head."""

    def __init__(self, path, *, device="cpu", dtype="float32"):
        from safetensors.torch import load_file
        from transformers import AutoTokenizer

        self.device = torch.device(device)
        if self.device.type not in ("cpu", "cuda"):
            raise ValueError("Stella device must be cpu or cuda")
        if dtype not in ("float32", "bfloat16", "float16"):
            raise ValueError("Stella dtype must be float32, bfloat16 or float16")
        if self.device.type == "cpu" and dtype != "float32":
            raise ValueError("Stella CPU path requires float32")
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise ValueError("Stella CUDA device is unavailable")
        self.dtype = dtype
        path = Path(path)
        config = json.loads((path / "config.json").read_text())
        pooling = json.loads((path / "1_Pooling/config.json").read_text())
        if (
            pooling.get("word_embedding_dimension") != config["hidden_size"]
            or pooling.get("pooling_mode_mean_tokens") is not True
            or any(
                value
                for key, value in pooling.items()
                if key.startswith("pooling_mode_") and key != "pooling_mode_mean_tokens"
            )
        ):
            raise ValueError("Stella requires masked mean-token pooling only")
        dense = json.loads((path / "2_Dense_1024/config.json").read_text())
        if (
            dense["in_features"] != config["hidden_size"]
            or dense.get("activation_function") != "torch.nn.modules.linear.Identity"
        ):
            raise ValueError("Stella dense head needs matching input size and Identity activation")
        # Meta initialization avoids a second full-size random allocation. Safetensors
        # maps CPU tensors; strict loading also checks the unused CLS pooler.
        with torch.device("meta"):
            self.model = StellaModel(config)
            self.linear = nn.ModuleDict(
                {
                    "linear": nn.Linear(
                        dense["in_features"], dense["out_features"], bias=dense["bias"]
                    )
                }
            )
        for module, file in (
            (self.model, path / "model.safetensors"),
            (self.linear, path / "2_Dense_1024/model.safetensors"),
        ):
            state = load_file(str(file), device="cpu")
            if any(tensor.dtype != torch.float32 for tensor in state.values()):
                raise ValueError(f"Stella requires fp32 tensors: {file}")
            module.load_state_dict(state, strict=True, assign=True)
            module.eval().requires_grad_(False)
            # Keep the dense projection in fp32 after fp32 pooling. Transfer one
            # module at a time, without a duplicate fp32 model on the GPU.
            module.to(
                device=self.device,
                dtype=getattr(torch, dtype) if module is self.model else torch.float32,
            )
        self.tokenizer = AutoTokenizer.from_pretrained(
            path, local_files_only=True, trust_remote_code=False
        )
        self.dimension = dense["out_features"]

    def token_lengths(self, texts):
        return [
            len(ids) for ids in self.tokenizer(texts, truncation=True, max_length=512)["input_ids"]
        ]

    @torch.inference_mode()
    def encode(self, texts):
        tokens = self.tokenizer(
            texts, padding=True, truncation=True, max_length=512, return_tensors="pt"
        )
        tokens = {key: value.to(self.device) for key, value in tokens.items()}
        mask = tokens["attention_mask"]
        hidden = self.model(**tokens).float()
        pooled = hidden.masked_fill(~mask[..., None].bool(), 0).sum(1) / mask.sum(1)[:, None]
        return self.linear["linear"](pooled).cpu().numpy()
