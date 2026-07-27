"""Common adapter/tokenizer loader for pinned trained checkpoints."""

from __future__ import annotations

from typing import Any, cast

import torch
from torch import nn

from pseudoroute.models.adapters.hf_deepseek_v2 import (
    HFDeepseekV2Adapter,
    transformers_fx_compatibility_shim,
)
from pseudoroute.models.adapters.hf_gpt_oss import HFGptOssAdapter
from pseudoroute.models.adapters.hf_mixtral import HFMixtralAdapter
from pseudoroute.models.adapters.hf_olmoe import HFOlmoeAdapter
from pseudoroute.models.adapters.hf_qwen2_moe import HFQwen2MoeAdapter
from pseudoroute.models.base import MoEModelAdapter
from pseudoroute.trained.config import TrainedModelConfig


def load_adapter(model_config: TrainedModelConfig, cache_dir: str) -> MoEModelAdapter:
    revision = model_config.revision
    model_id = model_config.model_id
    if model_config.adapter == "hf_olmoe":
        return HFOlmoeAdapter.from_pretrained(
            model_id,
            revision=revision,
            device="cuda:0",
            cache_dir=cache_dir,
            local_files_only=True,
        )
    if model_config.adapter == "hf_qwen2_moe":
        return HFQwen2MoeAdapter.from_pretrained(
            model_id,
            revision=revision,
            device="cuda:0",
            cache_dir=cache_dir,
            local_files_only=True,
        )
    if model_config.adapter == "hf_deepseek_v2":
        return HFDeepseekV2Adapter.from_pretrained(
            model_id,
            revision=revision,
            device="cuda:0",
            cache_dir=cache_dir,
            local_files_only=True,
        )
    if model_config.adapter == "hf_gpt_oss":
        return HFGptOssAdapter.from_pretrained(
            model_id,
            revision=revision,
            device="cuda:0",
            cache_dir=cache_dir,
            local_files_only=True,
        )
    if model_config.adapter == "hf_mixtral":
        from transformers import AutoModelForCausalLM

        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            revision=revision,
            cache_dir=cache_dir,
            local_files_only=True,
            trust_remote_code=False,
            dtype="auto",
            device_map="balanced",
            max_memory={0: "76GiB", 1: "76GiB", "cpu": "180GiB"},
        )
        module = cast(nn.Module, model)
        module.eval()
        return HFMixtralAdapter(module, model_id=model_id, revision=revision)
    raise ValueError(f"unsupported trained adapter: {model_config.adapter}")


def load_tokenizer(model_config: TrainedModelConfig, cache_dir: str) -> Any:
    from transformers import AutoTokenizer

    kwargs: dict[str, object] = {
        "revision": model_config.revision,
        "cache_dir": cache_dir,
        "local_files_only": True,
        "trust_remote_code": model_config.adapter == "hf_deepseek_v2",
    }
    if model_config.adapter == "hf_deepseek_v2":
        with transformers_fx_compatibility_shim():
            return AutoTokenizer.from_pretrained(model_config.model_id, **kwargs)
    return AutoTokenizer.from_pretrained(model_config.model_id, **kwargs)


def input_device(adapter: MoEModelAdapter) -> torch.device:
    explicit = getattr(adapter, "input_device", None)
    if isinstance(explicit, torch.device):
        return explicit
    model = cast(Any, adapter).model
    return cast(torch.device, model.model.embed_tokens.weight.device)
