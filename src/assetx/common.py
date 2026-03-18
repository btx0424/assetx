from __future__ import annotations

import mujoco
import shutil
from dataclasses import dataclass
from pathlib import Path


@dataclass
class JointCfg:
    name: str = ""
    type: str = "fixed"
    axis: tuple[float, float, float] = (0, 0, 1)
    range: tuple[float, float] = (0, 0)

    def __post_init__(self):
        if self.type not in ["fixed", "hinge", "slide"]:
            raise ValueError(f"Invalid joint type: {self.type}")


@dataclass
class MujocoAsset:
    xml_path: str
    spec: mujoco.MjSpec
    meshdir: Path

    @staticmethod
    def from_file(xml_path: str) -> MujocoAsset:
        spec = mujoco.MjSpec.from_file(xml_path)
        if len(spec.worldbody.bodies) > 1:
            raise ValueError("MujocoAsset must have only one body in the worldbody")
        root_body = spec.worldbody.first_body()
        root_body.pos = (0, 0, 0)
        root_body.quat = (1, 0, 0, 0)
        # resolve the actual meshdir from the spec
        meshdir = Path(xml_path).parent / spec.meshdir
        if not meshdir.exists():
            raise FileNotFoundError(f"Meshdir {meshdir} not found")
        return MujocoAsset(xml_path, spec, meshdir)

    def save(self, path: str | Path, *, copy_meshes: bool = False) -> MujocoAsset:
        """Save the asset to a directory.

        Writes model.xml and the mesh directory under the given path.
        The directory must not already exist.

        Args:
            path: Output directory path (not an XML file path). Created if missing.
            copy_meshes: If False (default), the mesh tree is copied with symlinks
                preserved (no mesh bytes copied when the asset uses symlinks, e.g. from
                assemble). If True, symlinks are dereferenced for a full, self-contained copy.

        Returns:
            A new MujocoAsset loaded from the saved files.

        Raises:
            ValueError: If path exists and is a file (path must be a directory).
        """
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
        shutil.copytree(self.meshdir, dest, symlinks=not copy_meshes)
        return MujocoAsset.from_file(str(root / "model.xml"))
