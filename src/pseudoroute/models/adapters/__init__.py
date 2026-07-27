"""Architecture-specific model adapters."""

from pseudoroute.models.adapters.hf_deepseek_v2 import HFDeepseekV2Adapter
from pseudoroute.models.adapters.hf_gpt_oss import HFGptOssAdapter
from pseudoroute.models.adapters.hf_mixtral import HFMixtralAdapter
from pseudoroute.models.adapters.hf_olmoe import HFOlmoeAdapter
from pseudoroute.models.adapters.hf_qwen2_moe import HFQwen2MoeAdapter
from pseudoroute.models.adapters.tiny import TinyMoEAdapter

__all__ = [
    "HFDeepseekV2Adapter",
    "HFGptOssAdapter",
    "HFMixtralAdapter",
    "HFOlmoeAdapter",
    "HFQwen2MoeAdapter",
    "TinyMoEAdapter",
]
