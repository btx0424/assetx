from __future__ import annotations

from abc import ABC, abstractmethod

from assetx.core.asset import MujocoAsset


class Transform(ABC):
    @abstractmethod
    def transform(self, asset: MujocoAsset) -> MujocoAsset:
        raise NotImplementedError


class Compose(Transform):
    def __init__(self, transforms: list[Transform]) -> None:
        self.transforms = transforms

    def transform(self, asset: MujocoAsset) -> MujocoAsset:
        for transform in self.transforms:
            asset = transform.transform(asset)
        return asset


def apply_transforms(asset: MujocoAsset, *transforms: Transform) -> MujocoAsset:
    return Compose(list(transforms)).transform(asset)
