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
    # Optional lifetime hook (kept for API compat; assemble no longer uses temp).
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
        meshdir = Path(spec.meshdir) if spec.meshdir else Path(".")
        resolved_meshdir = (path.parent / meshdir).resolve()
        if not resolved_meshdir.exists():
            raise FileNotFoundError(f"Meshdir {resolved_meshdir} not found")
        return MujocoAsset(path, spec, meshdir)

    def _resolve_mesh_source(self, mesh_file: str) -> Path:
        path = Path(mesh_file)
        if path.is_absolute():
            return path.resolve()
        for base in (self.resolved_meshdir, self.model_dir):
            candidate = (base / path).resolve()
            if candidate.is_file():
                return candidate
        return (self.resolved_meshdir / path).resolve()

    @staticmethod
    def _rewrite_mesh_files_relative(xml_path: Path, meshdir: Path) -> None:
        """Rewrite absolute ``file=`` attrs under ``meshdir`` to meshdir-relative paths.

        MuJoCo ``MjSpec.to_file`` emits absolute mesh paths from the compiled model.
        Artifacts should stay portable with ``meshdir="meshes"`` + relative files.
        """
        text = xml_path.read_text(encoding="utf-8")
        prefix = str(meshdir.resolve())
        for needle in (prefix + "/", prefix + "\\"):
            text = text.replace(f'file="{needle}', 'file="')
        xml_path.write_text(text, encoding="utf-8")

    @staticmethod
    def _ensure_explicit_sphere_types(xml_path: Path) -> None:
        """Re-insert ``type="sphere"`` omitted by MuJoCo ``to_file``.

        XML's default geom type is sphere, so MuJoCo drops ``type="sphere"``.
        On reload, a model-level ``<default><geom type="mesh"/></default>`` then
        incorrectly makes those geoms meshes.
        """
        import re

        def _repl(match: re.Match[str]) -> str:
            tag = match.group(0)
            if re.search(r"\stype=", tag) or re.search(r"\smesh=", tag):
                return tag
            if tag.endswith("/>"):
                return tag[:-2] + ' type="sphere"/>'
            return tag

        text = xml_path.read_text(encoding="utf-8")
        xml_path.write_text(
            re.sub(r"<geom\b[^>]*/>", _repl, text),
            encoding="utf-8",
        )
    def save(
        self,
        path: str | Path,
        *,
        copy_meshes: bool = True,
        save_urdf: bool = True,
    ) -> "MujocoAsset":
        """Write ``model.xml`` (and meshes) under ``path``.

        Every mesh referenced by the spec is materialized under
        ``path/meshes/{source_dir_name}/{filename}`` with **relative** ``file``
        attributes on disk. Set ``copy_meshes=False`` to symlink each source
        file instead of copying.
        """
        root = Path(path)
        if root.exists() and root.is_file():
            raise ValueError(
                f"path must be a directory, not a file: {path!r}. "
                "Pass the output directory where model.xml and meshes will be written."
            )
        root.mkdir(parents=True, exist_ok=True)

        dest_meshdir = root / "meshes"
        dest_meshdir.mkdir(parents=True, exist_ok=True)

        used_relpaths: set[str] = set()
        for mesh in self.spec.meshes:
            if not mesh.file:
                continue
            src = self._resolve_mesh_source(mesh.file)
            if not src.is_file():
                raise FileNotFoundError(
                    f"Mesh file not found: {src} (from {mesh.file!r})"
                )

            # Prefer a stable vendor/package folder name over generic "meshes"/"assets".
            parent_name = src.parent.name or "mesh"
            if parent_name.lower() in {"meshes", "assets", "mesh", "xml", "urdf"}:
                parent_name = src.parent.parent.name or parent_name
            rel = Path(parent_name) / src.name
            rel_key = str(rel).replace("\\", "/")
            if rel_key in used_relpaths:
                existing = dest_meshdir / rel
                if existing.exists() and existing.resolve() == src.resolve():
                    # Point compile at the materialized file (absolute).
                    mesh.file = str(existing.resolve())
                    continue
                stem, suffix = src.stem, src.suffix
                index = 1
                while True:
                    rel = Path(parent_name) / f"{stem}_{index}{suffix}"
                    rel_key = str(rel).replace("\\", "/")
                    if rel_key not in used_relpaths:
                        break
                    index += 1
            used_relpaths.add(rel_key)

            dest = dest_meshdir / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            if dest.exists() or dest.is_symlink():
                if dest.is_dir() and not dest.is_symlink():
                    shutil.rmtree(dest)
                else:
                    dest.unlink()
            if copy_meshes:
                shutil.copy2(src, dest)
            else:
                dest.symlink_to(src)

            # Absolute into the output tree so compile/to_file succeed without chdir.
            mesh.file = str(dest.resolve())

        self.spec.meshdir = "meshes"
        # to_file requires a compiled model and writes absolute mesh paths.
        self.spec.compile()
        requested_xml = root / "model.xml"
        self.spec.to_file(str(requested_xml))

        if requested_xml.exists():
            output_xml = requested_xml
        else:
            xml_candidates = sorted(
                root.glob("*.xml"), key=lambda p: p.stat().st_mtime, reverse=True
            )
            if not xml_candidates:
                raise FileNotFoundError(
                    f"No XML file written to output directory {root}"
                )
            output_xml = xml_candidates[0]

        self._rewrite_mesh_files_relative(output_xml, dest_meshdir)
        self._ensure_explicit_sphere_types(output_xml)

        if save_urdf:
            # Reload relative XML so the URDF also gets portable mesh paths.
            rel_spec = mujoco.MjSpec.from_file(str(output_xml))
            write_urdf(
                rel_spec,
                output_xml.with_suffix(".urdf"),
                robot_name=rel_spec.modelname or output_xml.stem,
                meshdir="meshes",
            )

        return MujocoAsset.from_file(output_xml)
