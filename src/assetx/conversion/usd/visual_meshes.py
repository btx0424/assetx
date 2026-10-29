"""Bake non-USD visual sources (e.g. Collada ``.dae``) into material-preserving USDA.

MuJoCo only loads STL/OBJ/MSH, so URDF Collada visuals are converted to STL for
compile. That bake drops per-part PBR colors. For USD we re-author those meshes
from the original DAE as small USDA assets and swap them onto ``*/visuals/*``
prims after the newton converter runs. Plain STL/OBJ sources are left alone.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

_SOURCES_NAME = "visual_mesh_sources.json"
_DAE_SUFFIXES = {".dae", ".zae"}


def sources_path_for_mjcf(mjcf: Path) -> Path:
    """Sidecar JSON next to an MJCF (``model_visual_mesh_sources.json``)."""
    return mjcf.with_name(f"{mjcf.stem}_visual_mesh_sources.json")


def find_visual_mesh_sources(mjcf: Path) -> Path | None:
    """Locate a sources map beside ``mjcf`` or named ``visual_mesh_sources.json``."""
    for candidate in (
        sources_path_for_mjcf(mjcf),
        mjcf.parent / _SOURCES_NAME,
    ):
        if candidate.is_file():
            return candidate
    return None


def write_visual_mesh_sources(path: Path, prim_to_source: dict[str, Path]) -> None:
    """Write ``{usd_prim_name: absolute_source_path}`` for DAE origins."""
    payload = {
        name: str(Path(source).resolve())
        for name, source in prim_to_source.items()
        if Path(source).suffix.lower() in _DAE_SUFFIXES
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_visual_mesh_sources(path: Path) -> dict[str, Path]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Expected object in {path}")
    return {str(stem): Path(source) for stem, source in data.items()}


def bake_visual_mesh_sources(
    prim_to_source: dict[str, Path],
    out_dir: Path,
) -> dict[str, Path]:
    """Convert each DAE source to ``out_dir/<prim_name>.usda``."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    baked: dict[str, Path] = {}
    for prim_name, source in prim_to_source.items():
        source = Path(source)
        if source.suffix.lower() not in _DAE_SUFFIXES:
            continue
        if not source.is_file():
            raise FileNotFoundError(f"Visual mesh source not found: {source}")
        dest = out_dir / f"{prim_name}.usda"
        write_dae_as_usda(source, dest, root_name=prim_name)
        baked[prim_name] = dest
    return baked


def write_dae_as_usda(dae_path: Path, usda_path: Path, *, root_name: str | None = None) -> None:
    """Author a USDA with one Mesh per DAE geometry, keeping PBR base colors."""
    import trimesh

    dae_path = Path(dae_path)
    loaded = trimesh.load(str(dae_path), process=False)
    parts = _scene_parts(loaded)
    if not parts:
        raise ValueError(f"No triangle geometry in {dae_path}")

    root = _usd_ident(root_name or dae_path.stem)
    looks: list[str] = []
    meshes: list[str] = []
    used_names: set[str] = set()
    for index, (part_name, mesh, color) in enumerate(parts):
        mat_name = f"mat_{index}"
        mesh_name = _unique_ident(part_name or f"part_{index}", used_names)
        looks.append(_material_block(root, mat_name, color))
        meshes.append(
            _mesh_block(
                mesh_name,
                mesh,
                color,
                material_path=f"/{root}/Looks/{mat_name}",
            )
        )

    body = "\n".join(
        [
            "#usda 1.0",
            "(",
            f'    defaultPrim = "{root}"',
            '    upAxis = "Z"',
            "    metersPerUnit = 1",
            ")",
            "",
            f'def Xform "{root}" (',
            '    kind = "component"',
            ")",
            "{",
            '    def Scope "Looks"',
            "    {",
            _indent("\n".join(looks), 2),
            "    }",
            "",
            _indent("\n\n".join(meshes), 1),
            "}",
            "",
        ]
    )
    usda_path = Path(usda_path)
    usda_path.parent.mkdir(parents=True, exist_ok=True)
    usda_path.write_text(body, encoding="utf-8")


def _scene_parts(loaded) -> list[tuple[str, object, tuple[float, float, float]]]:
    import trimesh

    if isinstance(loaded, trimesh.Trimesh):
        return [("mesh", loaded, _mesh_rgb01(loaded))]

    if not isinstance(loaded, trimesh.Scene):
        raise TypeError(f"Unsupported mesh type: {type(loaded)!r}")

    parts: list[tuple[str, object, tuple[float, float, float]]] = []
    for node in loaded.graph.nodes_geometry:
        transform, geom_name = loaded.graph.get(node)
        geom = loaded.geometry[geom_name]
        if not isinstance(geom, trimesh.Trimesh) or len(geom.faces) == 0:
            continue
        mesh = geom.copy()
        mesh.apply_transform(transform)
        parts.append((str(geom_name), mesh, _mesh_rgb01(geom)))
    return parts


def _mesh_rgb01(mesh) -> tuple[float, float, float]:
    """Diffuse RGB in 0–1. Trimesh stores uint8; some loaders leave 0–1 floats."""
    visual = getattr(mesh, "visual", None)
    material = getattr(visual, "material", None) if visual is not None else None
    for attr in ("baseColorFactor", "main_color"):
        factor = getattr(material, attr, None) if material is not None else None
        if factor is None or len(factor) < 3:
            continue
        rgb = [float(c) for c in list(factor)[:3]]
        if max(rgb) > 1.0:
            rgb = [c / 255.0 for c in rgb]
        return (rgb[0], rgb[1], rgb[2])
    return (200.0 / 255.0, 200.0 / 255.0, 200.0 / 255.0)


def _usd_ident(name: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_]", "_", name)
    if not cleaned:
        return "mesh"
    if cleaned[0].isdigit():
        cleaned = f"m_{cleaned}"
    return cleaned


def _unique_ident(name: str, used: set[str]) -> str:
    token = _usd_ident(name)
    if token not in used:
        used.add(token)
        return token
    index = 2
    while f"{token}_{index}" in used:
        index += 1
    token = f"{token}_{index}"
    used.add(token)
    return token


def _fmt_float(value: float) -> str:
    return f"{float(value):.8g}"


def _fmt_color(color: tuple[float, float, float]) -> str:
    return f"({_fmt_float(color[0])}, {_fmt_float(color[1])}, {_fmt_float(color[2])})"


def _material_block(root: str, name: str, color: tuple[float, float, float]) -> str:
    c = _fmt_color(color)
    surface = f"/{root}/Looks/{name}/PreviewSurface.outputs:surface"
    return "\n".join(
        [
            f'def Material "{name}"',
            "{",
            f"    token outputs:surface.connect = <{surface}>",
            '    def Shader "PreviewSurface"',
            "    {",
            '        uniform token info:id = "UsdPreviewSurface"',
            f"        color3f inputs:diffuseColor = {c}",
            "        float inputs:metallic = 0",
            "        float inputs:roughness = 1",
            "        token outputs:surface",
            "    }",
            "}",
        ]
    )


def _mesh_block(
    name: str,
    mesh,
    color: tuple[float, float, float],
    *,
    material_path: str,
) -> str:
    counts = ", ".join("3" for _ in range(len(mesh.faces)))
    indices = ", ".join(str(int(i)) for face in mesh.faces for i in face)
    points = ", ".join(
        f"({_fmt_float(v[0])}, {_fmt_float(v[1])}, {_fmt_float(v[2])})" for v in mesh.vertices
    )
    c = _fmt_color(color)
    return "\n".join(
        [
            f'def Mesh "{name}" (',
            '    prepend apiSchemas = ["MaterialBindingAPI"]',
            ")",
            "{",
            "    uniform bool doubleSided = 1",
            f"    int[] faceVertexCounts = [{counts}]",
            f"    int[] faceVertexIndices = [{indices}]",
            f"    point3f[] points = [{points}]",
            f"    color3f[] primvars:displayColor = [{c}]",
            f"    rel material:binding = <{material_path}>",
            "}",
        ]
    )


def _indent(text: str, levels: int) -> str:
    prefix = "    " * levels
    return "\n".join(prefix + line if line else line for line in text.splitlines())
