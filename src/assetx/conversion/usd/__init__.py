"""USD helpers for robot conversion.

Submodules are imported lazily: ``geoms`` / ``robot`` need ``pxr`` at import
time, while ``newton`` (MJCF/URDF -> USD) must work in hosts without it.
"""

from importlib import import_module

_EXPORTS = {
    "BodyGeom": "geoms",
    "export_meshes": "geoms",
    "extract_body_geoms": "geoms",
    "extract_meshes": "geoms",
    "mjcf_to_usd": "newton",
    "urdf_to_usd": "newton",
    "flatten_articulation": "postprocess",
    "KinematicTree": "robot",
    "build_kinematic_tree": "robot",
    "build_mjcf": "robot",
    "convert_usd_to_mjcf": "robot",
    "load_usd": "robot",
}

__all__ = sorted(_EXPORTS)


def __getattr__(name: str):
    module = _EXPORTS.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    return getattr(import_module(f"{__name__}.{module}"), name)
