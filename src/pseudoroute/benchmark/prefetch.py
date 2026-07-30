"""Common layer-ahead prefetch policy for Qwen3-MoE and GPT-OSS."""

from __future__ import annotations

import hashlib
import json
import types
from abc import ABC, abstractmethod
from contextlib import AbstractContextManager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, cast

import torch
import torch.nn.functional as functional
from safetensors.torch import load_file, save_file
from torch import Tensor, nn

PolicyName = Literal["vanilla", "router_pf", "oracle_pf"]
SubsetPolicyName = Literal["natural", "lossless", "hard"]


@dataclass(frozen=True)
class NativeRoute:
    logits: Tensor
    weights: Tensor
    ids: Tensor


@dataclass(frozen=True)
class SubsetRouteRecord:
    layer: int
    natural: NativeRoute
    executed: NativeRoute
    allowed: tuple[int, ...]


@dataclass(frozen=True)
class DefaultVectorArtifact:
    count: Tensor
    mean: Tensor
    fingerprint: str
    definition: str = "mean_unweighted_selected_expert_output"


@dataclass
class PrefetchStats:
    compared_tokens: int = 0
    exact_topk_tokens: int = 0
    selected_slots: int = 0
    matching_selected_slots: int = 0
    calls: int = 0

    def update(self, predicted: NativeRoute, natural: NativeRoute) -> None:
        predicted_ids = predicted.ids.detach()
        natural_ids = natural.ids.detach()
        self.compared_tokens += int(natural_ids.shape[0])
        self.exact_topk_tokens += int((predicted_ids == natural_ids).all(dim=-1).sum())
        self.selected_slots += int(natural_ids.numel())
        matches = (natural_ids.unsqueeze(-1) == predicted_ids.unsqueeze(-2)).any(dim=-1)
        self.matching_selected_slots += int(matches.sum())

    def as_dict(self) -> dict[str, int | float]:
        exact = self.exact_topk_tokens / self.compared_tokens if self.compared_tokens else 1.0
        hit = self.matching_selected_slots / self.selected_slots if self.selected_slots else 1.0
        return {
            "calls": self.calls,
            "compared_tokens": self.compared_tokens,
            "exact_topk_tokens": self.exact_topk_tokens,
            "exact_topk_agreement": exact,
            "selected_slots": self.selected_slots,
            "matching_selected_slots": self.matching_selected_slots,
            "selected_route_hit_rate": hit,
        }


class StreamingExpertMeans:
    """Batch-reduced streaming means without retaining token activations."""

    def __init__(self, layers: int, experts: int, hidden: int) -> None:
        self.count = torch.zeros(layers, experts, dtype=torch.int64)
        self.total = torch.zeros(layers, experts, hidden, dtype=torch.float64)

    def update(self, layer: int, expert: int, values: Tensor) -> None:
        if values.ndim != 2:
            raise ValueError("expert calibration values must be [tokens, hidden]")
        self.count[layer, expert] += int(values.shape[0])
        self.total[layer, expert] += values.detach().double().sum(dim=0).cpu()

    def finalize(self) -> DefaultVectorArtifact:
        mean = (self.total / self.count.clamp_min(1).unsqueeze(-1)).float()
        digest = hashlib.sha256()
        digest.update(self.count.contiguous().numpy().tobytes())
        digest.update(mean.contiguous().numpy().tobytes())
        return DefaultVectorArtifact(self.count.clone(), mean, digest.hexdigest())


class PrefetchModelOps(ABC):
    """Architecture-specific primitives under one policy implementation."""

    def __init__(self, model: nn.Module) -> None:
        self.model = model
        self.validate()

    @property
    @abstractmethod
    def layers(self) -> tuple[nn.Module, ...]: ...

    @property
    @abstractmethod
    def num_experts(self) -> int: ...

    @property
    @abstractmethod
    def top_k(self) -> int: ...

    @property
    @abstractmethod
    def hidden_size(self) -> int: ...

    @property
    def num_layers(self) -> int:
        return len(self.layers)

    @abstractmethod
    def validate(self) -> None: ...

    @abstractmethod
    def mlp(self, layer: int) -> nn.Module: ...

    @abstractmethod
    def post_attention_norm(self, layer: int) -> nn.Module: ...

    @abstractmethod
    def route(self, layer: int, hidden: Tensor) -> NativeRoute: ...

    @abstractmethod
    def experts(self, layer: int, hidden: Tensor, route: NativeRoute) -> Tensor: ...

    @abstractmethod
    def format_mlp_output(self, value: Tensor, route: NativeRoute) -> object: ...

    @abstractmethod
    def experts_and_collect(
        self,
        layer: int,
        hidden: Tensor,
        route: NativeRoute,
        collector: StreamingExpertMeans,
    ) -> Tensor: ...


class Qwen3MoePrefetchOps(PrefetchModelOps):
    @property
    def layers(self) -> tuple[nn.Module, ...]:
        return tuple(cast(Any, self.model).model.layers)

    @property
    def num_experts(self) -> int:
        return int(cast(Any, self.model).config.num_experts)

    @property
    def top_k(self) -> int:
        return int(cast(Any, self.model).config.num_experts_per_tok)

    @property
    def hidden_size(self) -> int:
        return int(cast(Any, self.model).config.hidden_size)

    def validate(self) -> None:
        config = cast(Any, self.model).config
        if getattr(config, "model_type", None) != "qwen3_moe":
            raise ValueError("Qwen3 prefetch adapter requires model_type=qwen3_moe")
        if len(self.layers) != int(config.num_hidden_layers):
            raise ValueError("Qwen3 layer count mismatch")
        for index, layer in enumerate(self.layers):
            if not hasattr(cast(Any, layer).mlp, "gate"):
                raise ValueError(f"Qwen3 layer {index} is not sparse MoE")

    def mlp(self, layer: int) -> nn.Module:
        return cast(nn.Module, cast(Any, self.layers[layer]).mlp)

    def post_attention_norm(self, layer: int) -> nn.Module:
        return cast(nn.Module, cast(Any, self.layers[layer]).post_attention_layernorm)

    def route(self, layer: int, hidden: Tensor) -> NativeRoute:
        logits, weights, ids = cast(Any, self.mlp(layer)).gate(hidden)
        return NativeRoute(logits, weights, ids)

    def experts(self, layer: int, hidden: Tensor, route: NativeRoute) -> Tensor:
        return cast(Tensor, cast(Any, self.mlp(layer)).experts(hidden, route.ids, route.weights))

    def format_mlp_output(self, value: Tensor, route: NativeRoute) -> object:
        return value

    def experts_and_collect(
        self,
        layer: int,
        hidden: Tensor,
        route: NativeRoute,
        collector: StreamingExpertMeans,
    ) -> Tensor:
        experts = cast(Any, self.mlp(layer)).experts
        result = torch.zeros_like(hidden)
        unique_experts = route.ids.unique()  # type: ignore[no-untyped-call]
        for expert in unique_experts.tolist():
            positions = (route.ids == int(expert)).nonzero(as_tuple=False)
            token_indices = positions[:, 0]
            topk_positions = positions[:, 1]
            current = hidden[token_indices]
            gate, up = functional.linear(current, experts.gate_up_proj[expert]).chunk(2, dim=-1)
            unweighted = functional.linear(experts.act_fn(gate) * up, experts.down_proj[expert])
            collector.update(layer, int(expert), unweighted)
            weighted = unweighted * route.weights[token_indices, topk_positions, None]
            result.index_add_(0, token_indices, weighted.to(result.dtype))
        return result


class GptOssPrefetchOps(PrefetchModelOps):
    @property
    def layers(self) -> tuple[nn.Module, ...]:
        return tuple(cast(Any, self.model).model.layers)

    @property
    def num_experts(self) -> int:
        return int(cast(Any, self.model).config.num_local_experts)

    @property
    def top_k(self) -> int:
        return int(cast(Any, self.model).config.num_experts_per_tok)

    @property
    def hidden_size(self) -> int:
        return int(cast(Any, self.model).config.hidden_size)

    def validate(self) -> None:
        config = cast(Any, self.model).config
        if getattr(config, "model_type", None) != "gpt_oss":
            raise ValueError("GPT-OSS prefetch adapter requires model_type=gpt_oss")
        quantization = getattr(config, "quantization_config", None)
        method = (
            quantization.get("quant_method")
            if isinstance(quantization, dict)
            else getattr(quantization, "quant_method", None)
        )
        if method != "mxfp4":
            raise ValueError("GPT-OSS benchmark must preserve native MXFP4 experts")

    def mlp(self, layer: int) -> nn.Module:
        return cast(nn.Module, cast(Any, self.layers[layer]).mlp)

    def post_attention_norm(self, layer: int) -> nn.Module:
        return cast(nn.Module, cast(Any, self.layers[layer]).post_attention_layernorm)

    def route(self, layer: int, hidden: Tensor) -> NativeRoute:
        logits, weights, ids = cast(Any, self.mlp(layer)).router(hidden)
        return NativeRoute(logits, weights, ids)

    def _kernel_experts(self, layer: int, hidden: Tensor, logits: Tensor, top_k: int) -> Tensor:
        from transformers.integrations import mxfp4 as mxfp4_module

        mxfp4 = cast(Any, mxfp4_module)
        with mxfp4.on_device(logits.device):
            routing_data, gather_idx, scatter_idx = mxfp4.triton_kernels_hub.routing.routing(
                logits, top_k
            )
        experts = cast(Any, self.mlp(layer)).experts
        return cast(Tensor, experts(hidden, routing_data, gather_idx, scatter_idx=scatter_idx))

    def experts(self, layer: int, hidden: Tensor, route: NativeRoute) -> Tensor:
        return self._kernel_experts(layer, hidden, route.logits, self.top_k)

    def format_mlp_output(self, value: Tensor, route: NativeRoute) -> object:
        return value, route.logits

    def experts_and_collect(
        self,
        layer: int,
        hidden: Tensor,
        route: NativeRoute,
        collector: StreamingExpertMeans,
    ) -> Tensor:
        result = torch.zeros_like(hidden)
        unique_experts = route.ids.unique()  # type: ignore[no-untyped-call]
        for expert in unique_experts.tolist():
            positions = (route.ids == int(expert)).nonzero(as_tuple=False)
            token_indices = positions[:, 0]
            topk_positions = positions[:, 1]
            current = hidden[token_indices]
            logits = torch.full(
                (current.shape[0], self.num_experts),
                -torch.inf,
                dtype=route.logits.dtype,
                device=hidden.device,
            )
            logits[:, int(expert)] = 0
            unweighted = self._kernel_experts(layer, current, logits, 1)
            collector.update(layer, int(expert), unweighted)
            weighted = unweighted * route.weights[token_indices, topk_positions, None]
            result.index_add_(0, token_indices, weighted.to(result.dtype))
        return result


def build_prefetch_ops(model: nn.Module, architecture: str) -> PrefetchModelOps:
    if architecture == "qwen3_moe":
        return Qwen3MoePrefetchOps(model)
    if architecture == "gpt_oss":
        return GptOssPrefetchOps(model)
    raise ValueError(f"unsupported prefetch architecture: {architecture}")


def masked_route(
    ops: PrefetchModelOps, layer: int, hidden: Tensor, allowed: tuple[int, ...]
) -> NativeRoute:
    """Apply an explicit outside-subset logit mask before native route selection."""
    if len(allowed) < ops.top_k:
        raise ValueError("hard subset must contain at least native top-k experts")
    natural = ops.route(layer, hidden)
    allowed_tensor = torch.tensor(allowed, dtype=torch.long, device=natural.logits.device)
    if allowed_tensor.unique().numel() != len(allowed):  # type: ignore[no-untyped-call]
        raise ValueError("hard subset contains duplicate experts")
    if int(allowed_tensor.min()) < 0 or int(allowed_tensor.max()) >= ops.num_experts:
        raise ValueError("hard subset expert is outside the layer-local ID range")
    keep = torch.zeros(ops.num_experts, dtype=torch.bool, device=natural.logits.device)
    keep[allowed_tensor] = True
    logits = natural.logits.masked_fill(~keep, -torch.inf)
    if isinstance(ops, Qwen3MoePrefetchOps):
        probabilities = torch.softmax(logits, dtype=torch.float32, dim=-1)
        weights, ids = probabilities.topk(ops.top_k, dim=-1)
        if bool(cast(Any, ops.model).config.norm_topk_prob):
            weights = weights / weights.sum(dim=-1, keepdim=True)
        weights = weights.to(logits.dtype)
    elif isinstance(ops, GptOssPrefetchOps):
        selected_logits, ids = logits.topk(ops.top_k, dim=-1)
        weights = torch.softmax(selected_logits, dim=-1, dtype=selected_logits.dtype)
    else:
        raise TypeError(f"masked subset routing is unsupported for {type(ops).__name__}")
    return NativeRoute(logits, weights, ids)


def physical_expert_bytes(ops: PrefetchModelOps, layer: int) -> int:
    """Return checkpoint-physical bytes for one routed expert in a layer."""
    experts = cast(Any, ops.mlp(layer)).experts
    if isinstance(ops, Qwen3MoePrefetchOps):
        gate_up = cast(Tensor, experts.gate_up_proj)
        down = cast(Tensor, experts.down_proj)
        return int(
            gate_up[0].numel() * gate_up.element_size() + down[0].numel() * down.element_size()
        )
    if isinstance(ops, GptOssPrefetchOps):
        physical = 0
        for projection in ("gate_up_proj", "down_proj"):
            tensor = getattr(experts, projection)
            storage = getattr(tensor, "storage", None)
            if storage is not None and not callable(storage):
                data = storage.data
                physical += int(data.numel() * data.element_size())
                precision = getattr(experts, f"{projection}_precision_config")
                scale_data = precision.weight_scale.storage.data
                physical += int(scale_data.numel() * scale_data.element_size())
            else:
                physical += int(tensor.numel() * tensor.element_size())
            bias = getattr(experts, f"{projection}_bias")
            physical += int(bias.numel() * bias.element_size())
        if physical % ops.num_experts:
            raise ValueError("GPT-OSS MXFP4 storage is not expert-divisible")
        return physical // ops.num_experts
    raise TypeError(f"expert byte accounting is unsupported for {type(ops).__name__}")


class _PatchedMlpContext(AbstractContextManager["_PatchedMlpContext"]):
    def __init__(
        self,
        ops: PrefetchModelOps,
        *,
        policy: PolicyName | None = None,
        defaults: DefaultVectorArtifact | None = None,
        collector: StreamingExpertMeans | None = None,
    ) -> None:
        if (policy is None) == (collector is None):
            raise ValueError("select exactly one of policy execution or calibration")
        if policy == "router_pf" and defaults is None:
            raise ValueError("router_pf requires default vectors")
        self.ops = ops
        self.policy = policy
        self.collector = collector
        self.stats = PrefetchStats()
        self._original: list[Any] = []
        self._handles: list[torch.utils.hooks.RemovableHandle] = []
        self._residuals: dict[int, Tensor] = {}
        self._pending: NativeRoute | None = None
        self._defaults = (
            defaults.mean.to(
                device=next(ops.model.parameters()).device,
                dtype=next(ops.model.parameters()).dtype,
            )
            if defaults is not None
            else None
        )

    def _capture_residual(self, layer: int, inputs: tuple[object, ...]) -> None:
        value = inputs[0]
        if not isinstance(value, Tensor):
            raise RuntimeError("post-attention norm input is not a tensor")
        self._residuals[layer] = value

    def _calibration_forward(self, layer: int, hidden_states: Tensor) -> object:
        shape = hidden_states.shape
        flat = hidden_states.reshape(-1, shape[-1])
        route = self.ops.route(layer, flat)
        if self.collector is None:
            raise AssertionError("calibration collector missing")
        value = self.ops.experts_and_collect(layer, flat, route, self.collector)
        return self.ops.format_mlp_output(value.reshape(shape), route)

    def _policy_forward(self, layer: int, hidden_states: Tensor) -> object:
        shape = hidden_states.shape
        flat = hidden_states.reshape(-1, shape[-1])
        natural = self.ops.route(layer, flat)
        if layer == 0:
            self._pending = None
        if self.policy == "router_pf" and layer > 0:
            if self._pending is None:
                raise RuntimeError("router_pf is missing the previous layer prediction")
            executed = self._pending
            self.stats.update(executed, natural)
        else:
            executed = natural
            if self.policy == "oracle_pf" and layer > 0:
                self.stats.update(natural, natural)
        value = self.ops.experts(layer, flat, executed)
        self.stats.calls += 1
        if self.policy == "router_pf" and layer + 1 < self.ops.num_layers:
            residual = self._residuals.get(layer)
            if residual is None:
                raise RuntimeError(
                    f"router_pf did not capture layer {layer} post-attention residual"
                )
            if self._defaults is None:
                raise AssertionError("router_pf defaults missing")
            default_layer = self._defaults[layer]
            mixture = (executed.weights.unsqueeze(-1) * default_layer[executed.ids]).sum(dim=1)
            quasi_input = residual.reshape(-1, shape[-1]) + mixture.to(residual.dtype)
            quasi = self.ops.post_attention_norm(layer + 1)(quasi_input)
            self._pending = self.ops.route(layer + 1, quasi)
        else:
            self._pending = None
        return self.ops.format_mlp_output(value.reshape(shape), executed)

    def __enter__(self) -> _PatchedMlpContext:
        for index in range(self.ops.num_layers):
            norm = self.ops.post_attention_norm(index)
            self._handles.append(
                norm.register_forward_pre_hook(
                    lambda _module, inputs, layer=index: self._capture_residual(layer, inputs)
                )
            )
            mlp = self.ops.mlp(index)
            self._original.append(mlp.forward)

            def replacement(
                _module: nn.Module, hidden_states: Tensor, *, layer: int = index
            ) -> object:
                if self.collector is not None:
                    return self._calibration_forward(layer, hidden_states)
                return self._policy_forward(layer, hidden_states)

            mlp.forward = types.MethodType(replacement, mlp)
        return self

    def __exit__(self, *exc: object) -> None:
        for index, original in enumerate(self._original):
            self.ops.mlp(index).forward = original
        for handle in self._handles:
            handle.remove()
        self._original.clear()
        self._handles.clear()


def policy_context(
    ops: PrefetchModelOps, policy: PolicyName, defaults: DefaultVectorArtifact | None = None
) -> _PatchedMlpContext:
    return _PatchedMlpContext(ops, policy=policy, defaults=defaults)


class NativeRouteCaptureContext(AbstractContextManager["NativeRouteCaptureContext"]):
    """Capture native router outputs with hooks and no forward replacement."""

    def __init__(
        self,
        ops: PrefetchModelOps,
        allowed_by_layer: dict[int, tuple[int, ...]] | None = None,
    ) -> None:
        self.ops = ops
        self.allowed_by_layer = dict(allowed_by_layer or {})
        self._records: list[SubsetRouteRecord] = []
        self._handles: list[torch.utils.hooks.RemovableHandle] = []

    @staticmethod
    def _cpu_route(route: NativeRoute) -> NativeRoute:
        return NativeRoute(
            route.logits.detach().cpu(),
            route.weights.detach().cpu(),
            route.ids.detach().cpu(),
        )

    def _record(self, layer: int, route: NativeRoute) -> None:
        cpu = self._cpu_route(route)
        self._records.append(
            SubsetRouteRecord(
                layer,
                cpu,
                cpu,
                self.allowed_by_layer.get(layer, ()),
            )
        )

    def _capture_qwen(self, layer: int, output: object) -> None:
        if not isinstance(output, tuple) or len(output) != 3:
            raise RuntimeError("Qwen native gate did not return logits/weights/ids")
        logits, weights, ids = output
        if not all(isinstance(value, Tensor) for value in output):
            raise RuntimeError("Qwen native gate route contains a non-tensor")
        self._record(
            layer,
            NativeRoute(cast(Tensor, logits), cast(Tensor, weights), cast(Tensor, ids)),
        )

    def _capture_gpt(self, layer: int, output: object) -> None:
        if not isinstance(output, tuple) or len(output) != 2 or not isinstance(output[1], Tensor):
            raise RuntimeError("GPT native MXFP4 MLP did not return router logits")
        logits = output[1]
        selected, ids = logits.topk(self.ops.top_k, dim=-1)
        weights = torch.softmax(selected, dim=-1, dtype=selected.dtype)
        self._record(layer, NativeRoute(logits, weights, ids))

    def drain(self) -> tuple[SubsetRouteRecord, ...]:
        records = tuple(self._records)
        self._records.clear()
        return records

    def __enter__(self) -> NativeRouteCaptureContext:
        for layer in range(self.ops.num_layers):
            if isinstance(self.ops, Qwen3MoePrefetchOps):
                gate = cast(nn.Module, cast(Any, self.ops.mlp(layer)).gate)
                self._handles.append(
                    gate.register_forward_hook(
                        lambda _module, _inputs, output, index=layer: self._capture_qwen(
                            index, output
                        )
                    )
                )
            elif isinstance(self.ops, GptOssPrefetchOps):
                self._handles.append(
                    self.ops.mlp(layer).register_forward_hook(
                        lambda _module, _inputs, output, index=layer: self._capture_gpt(
                            index, output
                        )
                    )
                )
            else:
                raise TypeError(f"native route hooks unsupported for {type(self.ops).__name__}")
        return self

    def __exit__(self, *exc: object) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()


class SubsetExecutionContext(AbstractContextManager["SubsetExecutionContext"]):
    """Reversible MLP patch for route capture, lossless residency, and hard masking."""

    def __init__(
        self,
        ops: PrefetchModelOps,
        policy: SubsetPolicyName,
        allowed_by_layer: dict[int, tuple[int, ...]] | None = None,
    ) -> None:
        self.ops = ops
        self.policy = policy
        self.allowed_by_layer = dict(allowed_by_layer or {})
        self._original: list[Any] = []
        self._records: list[SubsetRouteRecord] = []

    def set_allowed(self, allowed_by_layer: dict[int, tuple[int, ...]]) -> None:
        expected = set(range(self.ops.num_layers))
        if set(allowed_by_layer) != expected:
            raise ValueError("subset must cover every routed layer")
        for layer, subset in allowed_by_layer.items():
            if len(subset) < self.ops.top_k or len(subset) > self.ops.num_experts:
                raise ValueError(f"invalid layer {layer} subset size")
        self.allowed_by_layer = dict(allowed_by_layer)

    @staticmethod
    def _cpu_route(route: NativeRoute) -> NativeRoute:
        return NativeRoute(
            route.logits.detach().cpu(),
            route.weights.detach().cpu(),
            route.ids.detach().cpu(),
        )

    def _forward(self, layer: int, hidden_states: Tensor) -> object:
        shape = hidden_states.shape
        flat = hidden_states.reshape(-1, shape[-1])
        natural = self.ops.route(layer, flat)
        allowed = self.allowed_by_layer.get(layer, ())
        if self.policy == "natural":
            executed = natural
        else:
            if not allowed:
                raise RuntimeError(f"{self.policy} subset missing for layer {layer}")
            executed = (
                natural
                if self.policy == "lossless"
                else masked_route(self.ops, layer, flat, allowed)
            )
        value = self.ops.experts(layer, flat, executed)
        self._records.append(
            SubsetRouteRecord(
                layer,
                self._cpu_route(natural),
                self._cpu_route(executed),
                allowed,
            )
        )
        return self.ops.format_mlp_output(value.reshape(shape), executed)

    def drain(self) -> tuple[SubsetRouteRecord, ...]:
        records = tuple(self._records)
        self._records.clear()
        return records

    def __enter__(self) -> SubsetExecutionContext:
        for index in range(self.ops.num_layers):
            mlp = self.ops.mlp(index)
            self._original.append(mlp.forward)

            def replacement(
                _module: nn.Module, hidden_states: Tensor, *, layer: int = index
            ) -> object:
                return self._forward(layer, hidden_states)

            mlp.forward = types.MethodType(replacement, mlp)
        return self

    def __exit__(self, *exc: object) -> None:
        for index, original in enumerate(self._original):
            self.ops.mlp(index).forward = original
        self._original.clear()


def calibrate_default_vectors(
    ops: PrefetchModelOps, token_batches: tuple[dict[str, Tensor], ...]
) -> DefaultVectorArtifact:
    collector = StreamingExpertMeans(ops.num_layers, ops.num_experts, ops.hidden_size)
    with _PatchedMlpContext(ops, collector=collector), torch.inference_mode():
        for inputs in token_batches:
            cast(Any, ops.model)(**inputs, use_cache=False, return_dict=True)
    return collector.finalize()


def save_default_vectors(
    root: Path, artifact: DefaultVectorArtifact, metadata: dict[str, object]
) -> None:
    root.mkdir(parents=True, exist_ok=True)
    save_file(
        {"count": artifact.count.contiguous(), "mean": artifact.mean.contiguous()},
        str(root / "default_vectors.safetensors"),
    )
    manifest = {
        "schema_version": 1,
        "safe_format": "safetensors",
        "definition": artifact.definition,
        "fingerprint": artifact.fingerprint,
        "counts_min": int(artifact.count.min()),
        "counts_max": int(artifact.count.max()),
        "zero_count_experts": int((artifact.count == 0).sum()),
        "zero_count_indices": (artifact.count == 0).nonzero().tolist(),
        **metadata,
    }
    (root / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def load_default_vectors(root: Path) -> DefaultVectorArtifact:
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("safe_format") != "safetensors":
        raise ValueError("default vectors must use safetensors")
    tensors = load_file(str(root / "default_vectors.safetensors"))
    artifact = DefaultVectorArtifact(
        tensors["count"],
        tensors["mean"],
        str(manifest["fingerprint"]),
        str(manifest["definition"]),
    )
    digest = hashlib.sha256()
    digest.update(artifact.count.contiguous().numpy().tobytes())
    digest.update(artifact.mean.contiguous().numpy().tobytes())
    if digest.hexdigest() != artifact.fingerprint:
        raise ValueError("default-vector fingerprint mismatch")
    return artifact
