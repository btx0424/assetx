from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path

import mujoco


@dataclass(frozen=True)
class JointCfg:
    name: str = ""
    type: str = "fixed"
    axis: tuple[float, float, float] = (0, 0, 1)
    range: tuple[float, float] = (0, 0)

    def __post_init__(self) -> None:
        if self.type not in {"fixed", "hinge", "slide"}:
            raise ValueError(f"Invalid joint type: {self.type}")


@dataclass(frozen=True)
class MujocoAsset:
    xml_path: Path
    spec: mujoco.MjSpec
    meshdir: Path

    @property
    def model_dir(self) -> Path:
        return self.xml_path.parent

    @property
    def resolved_meshdir(self) -> Path:
        if self.meshdir.is_absolute():
            return self.meshdir
        return (self.model_dir / self.meshdir).resolve()

    @staticmethod
    def from_file(xml_path: str | Path) -> "MujocoAsset":
        path = Path(xml_path).resolve()
        spec = mujoco.MjSpec.from_file(str(path))
        if len(spec.worldbody.bodies) > 1:
            raise ValueError("MujocoAsset must have only one body in the worldbody")
        root_body = spec.worldbody.first_body()
        root_body.pos = (0, 0, 0)
        root_body.quat = (1, 0, 0, 0)
        meshdir = Path(spec.meshdir)
        resolved_meshdir = (path.parent / meshdir).resolve()
        if not resolved_meshdir.exists():
            raise FileNotFoundError(f"Meshdir {resolved_meshdir} not found")
        return MujocoAsset(path, spec, meshdir)

    def save(self, path: str | Path, *, copy_meshes: bool = False) -> "MujocoAsset":
        root = Path(path)
        if root.exists() and root.is_file():
            raise ValueError(
                f"path must be a directory, not a file: {path!r}. "
                "Pass the output directory where model.xml and meshes will be written."
            )
        root.mkdir(parents=True, exist_ok=False)
        self.spec.compile()
        self.spec.to_file(str(root / "model.xml"))
        dest = root / self.spec.meshdir
        shutil.copytree(self.resolved_meshdir, dest, symlinks=not copy_meshes)
        return MujocoAsset.from_file(root / "model.xml")
