"""Architecture-specific model adapters."""

from pseudoroute.models.adapters.hf_mixtral import HFMixtralAdapter
from pseudoroute.models.adapters.tiny import TinyMoEAdapter

__all__ = ["HFMixtralAdapter", "TinyMoEAdapter"]
