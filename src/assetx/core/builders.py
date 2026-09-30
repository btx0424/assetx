from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from assetx.core.asset import MujocoAsset


AssetBuilderFn = Callable[..., MujocoAsset]


@dataclass(frozen=True)
class AssetBuilder:
    name: str
    fn: AssetBuilderFn

    def __call__(self, *args: object, **kwargs: object) -> MujocoAsset:
        return self.fn(*args, **kwargs)


_REGISTRY: dict[str, AssetBuilder] = {}


def asset_builder(
    fn: AssetBuilderFn | None = None,
    *,
    name: str | None = None,
) -> AssetBuilderFn | Callable[[AssetBuilderFn], AssetBuilderFn]:
    def register(builder_fn: AssetBuilderFn) -> AssetBuilderFn:
        builder_name = name or builder_fn.__name__
        _REGISTRY[builder_name] = AssetBuilder(builder_name, builder_fn)
        setattr(builder_fn, "__assetx_builder_name__", builder_name)
        return builder_fn

    if fn is None:
        return register
    return register(fn)


def get_asset_builder(name: str) -> AssetBuilderFn:
    return _REGISTRY[name].fn


def list_asset_builders() -> tuple[str, ...]:
    return tuple(sorted(_REGISTRY))
