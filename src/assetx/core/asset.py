from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import shutil
from typing import Any

import mujoco
from assetx.conversion.mjcf2urdf import write_urdf


@dataclass(frozen=True)
class JointCfg:
    name: str = ""
    type: str = "free"
    axis: tuple[float, float, float] = (0, 0, 1)
    limited: bool = True
    range: tuple[float, float] = (0, 0)

    def __post_init__(self) -> None:
        if self.type not in {"hinge", "slide", "free", "fixed"}:
            raise ValueError(f"Invalid joint type: {self.type}")


@dataclass(frozen=True)
class MujocoAsset:
    xml_path: Path
    spec: mujoco.MjSpec
    meshdir: Path
    # Keeps TemporaryDirectory (or similar) alive while this asset references tmp files.
    _tmpdir: Any = field(default=None, repr=False, compare=False, hash=False)

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

    def save(
        self,
        path: str | Path,
        *,
        copy_meshes: bool = False,
        save_urdf: bool = True,
    ) -> "MujocoAsset":
        root = Path(path)
        if root.exists() and root.is_file():
            raise ValueError(
                f"path must be a directory, not a file: {path!r}. "
                "Pass the output directory where model.xml and meshes will be written."
            )
        root.mkdir(parents=True, exist_ok=True)
        requested_xml = root / "model.xml"
        self.spec.to_file(str(requested_xml))

        # MuJoCo may ignore requested filename and emit <model_name>.xml.
        if requested_xml.exists():
            output_xml = requested_xml
        else:
            xml_candidates = sorted(root.glob("*.xml"), key=lambda p: p.stat().st_mtime, reverse=True)
            if not xml_candidates:
                raise FileNotFoundError(f"No XML file written to output directory {root}")
            output_xml = xml_candidates[0]

        # Copy or link meshdir as-is (keep original meshdir semantics).
        dest_meshdir = root / self.spec.meshdir
        dest_meshdir.parent.mkdir(parents=True, exist_ok=True)
        source_meshdir = self.resolved_meshdir
        if not source_meshdir.exists():
            raise FileNotFoundError(f"Meshdir {source_meshdir} not found")

        if dest_meshdir.resolve() == root.resolve():
            # meshdir="." points to the output root; mirror directory contents.
            for entry in source_meshdir.iterdir():
                if entry.suffix.lower() == ".xml":
                    continue
                target = root / entry.name
                if target.exists() or target.is_symlink():
                    if target.is_dir() and not target.is_symlink():
                        shutil.rmtree(target)
                    else:
                        target.unlink()
                if entry.is_dir():
                    if copy_meshes:
                        shutil.copytree(entry, target)
                    else:
                        target.symlink_to(entry, target_is_directory=True)
                else:
                    if copy_meshes:
                        shutil.copy2(entry, target)
                    else:
                        target.symlink_to(entry)
        else:
            if dest_meshdir.exists() or dest_meshdir.is_symlink():
                if dest_meshdir.is_dir() and not dest_meshdir.is_symlink():
                    shutil.rmtree(dest_meshdir)
                else:
                    dest_meshdir.unlink()
            if copy_meshes:
                shutil.copytree(source_meshdir, dest_meshdir)
            else:
                dest_meshdir.symlink_to(source_meshdir, target_is_directory=True)

        if save_urdf:
            write_urdf(
                self.spec,
                output_xml.with_suffix(".urdf"),
                robot_name=self.spec.modelname or output_xml.stem,
                meshdir=str(self.spec.meshdir),
            )
        return MujocoAsset(output_xml.resolve(), self.spec, Path(str(self.spec.meshdir)))
