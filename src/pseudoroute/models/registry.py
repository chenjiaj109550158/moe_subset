"""Adapter registry without model-name conditionals in core logic."""

from __future__ import annotations

from collections.abc import Callable

from pseudoroute.models.base import MoEModelAdapter

AdapterFactory = Callable[..., MoEModelAdapter]
_FACTORIES: dict[str, AdapterFactory] = {}


def register_adapter(name: str) -> Callable[[AdapterFactory], AdapterFactory]:
    def register(factory: AdapterFactory) -> AdapterFactory:
        if name in _FACTORIES:
            raise ValueError(f"adapter already registered: {name}")
        _FACTORIES[name] = factory
        return factory

    return register


def create_adapter(name: str, **kwargs: object) -> MoEModelAdapter:
    try:
        factory = _FACTORIES[name]
    except KeyError as error:
        supported = ", ".join(sorted(_FACTORIES)) or "none"
        raise ValueError(
            f"unsupported adapter {name!r}; registered adapters: {supported}"
        ) from error
    return factory(**kwargs)


def registered_adapters() -> tuple[str, ...]:
    return tuple(sorted(_FACTORIES))
