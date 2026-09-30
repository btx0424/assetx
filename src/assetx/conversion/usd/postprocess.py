"""Post-process newton-converter USD output into the Isaac Lab articulation layout.

Flat links under the robot prim, ``{link}/visuals`` / ``{link}/collisions``
geometry groups and ``/<robot>/joints``, matching Isaac Sim's importer.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path


def remove_world_fixed_joints(usd_path: str | Path) -> list[str]:
    """Delete fixed joints welding a rigid body to the world; return their paths.

    Specs are deleted from every layer that defines them, and those layers are
    saved in place.
    """
    from pxr import Sdf, Usd, UsdPhysics

    stage = Usd.Stage.Open(str(usd_path))
    if stage is None:
        raise RuntimeError(f"Failed to open USD stage: {usd_path}")

    def _is_body(targets: list[Sdf.Path]) -> bool:
        return any(
            stage.GetPrimAtPath(t).HasAPI(UsdPhysics.RigidBodyAPI) for t in targets
        )

    world_joints = []
    for prim in stage.Traverse():
        if not prim.IsA(UsdPhysics.FixedJoint):
            continue
        joint = UsdPhysics.Joint(prim)
        body0 = _is_body(joint.GetBody0Rel().GetTargets())
        body1 = _is_body(joint.GetBody1Rel().GetTargets())
        if body0 != body1:
            world_joints.append(prim)

    removed: list[str] = []
    dirty_layers: set[Sdf.Layer] = set()
    for prim in world_joints:
        for spec in prim.GetPrimStack():
            layer, path = spec.layer, spec.path
            parent = layer.GetPrimAtPath(path.GetParentPath())
            if parent is not None and path.name in parent.nameChildren:
                del parent.nameChildren[path.name]
                dirty_layers.add(layer)
        removed.append(str(prim.GetPath()))

    for layer in dirty_layers:
        layer.Save()
    return removed


_LIST_OP_FIELDS = ("prependedItems", "appendedItems", "deletedItems", "orderedItems")


def _read_list_op(proxy) -> dict[str, list]:
    if proxy.isExplicit:
        return {"explicitItems": list(proxy.explicitItems)}
    return {f: list(getattr(proxy, f)) for f in _LIST_OP_FIELDS if len(getattr(proxy, f))}


def _write_list_op(proxy, items: dict[str, list], remap) -> None:
    for field, paths in items.items():
        setattr(proxy, field, [remap(p) for p in paths])


def _open_single_layer(usd_path: str | Path):
    from pxr import Usd

    stage = Usd.Stage.Open(str(usd_path))
    if stage is None:
        raise RuntimeError(f"Failed to open USD stage: {usd_path}")
    layer = stage.GetRootLayer()
    if len(layer.subLayerPaths) or any(
        spec.layer != layer
        for prim in stage.Traverse()
        for spec in prim.GetPrimStack()
        if not spec.layer.anonymous
    ):
        raise ValueError(
            "Articulation post-processing requires a single-layer USD "
            "(convert with --no-layer-structure / flat=True)"
        )
    root = stage.GetDefaultPrim()
    if not root:
        raise ValueError(f"{usd_path} has no default prim")
    return stage, layer, root


def _move_prims(layer, moves: dict) -> None:
    """Move prim specs within ``layer`` and rewrite every path reference.

    Destination parents must already exist. Relationship targets and attribute
    connections anywhere in the layer that point into a moved subtree are
    remapped (longest matching source prefix wins).
    """
    from pxr import Sdf

    def remap(path: Sdf.Path) -> Sdf.Path:
        best = None
        for old in moves:
            if path.HasPrefix(old) and (
                best is None or old.pathElementCount > best.pathElementCount
            ):
                best = old
        return path.ReplacePrefix(best, moves[best]) if best is not None else path

    # Snapshot targets before moving: Sdf.CopySpec rewrites targets that point
    # inside the copied subtree, which would hide them from ``remap``.
    targets: list[tuple[Sdf.Path, str, dict[str, list]]] = []

    def _collect(path: Sdf.Path) -> None:
        obj = layer.GetObjectAtPath(path)
        if isinstance(obj, Sdf.RelationshipSpec):
            targets.append((path, "targetPathList", _read_list_op(obj.targetPathList)))
        elif isinstance(obj, Sdf.AttributeSpec) and obj.connectionPathList.GetAddedOrExplicitItems():
            targets.append((path, "connectionPathList", _read_list_op(obj.connectionPathList)))

    layer.Traverse(Sdf.Path.absoluteRootPath, _collect)

    # Deepest first: each move's source path is still intact (ancestors unmoved).
    with Sdf.ChangeBlock():
        for old in sorted(moves, key=lambda p: p.pathElementCount, reverse=True):
            new = moves[old]
            if not Sdf.CopySpec(layer, old, layer, new):
                raise RuntimeError(f"Failed to copy {old} -> {new}")
            parent = layer.GetPrimAtPath(old.GetParentPath())
            del parent.nameChildren[old.name]

        for spec_path, field, items in targets:
            obj = layer.GetObjectAtPath(remap(spec_path))
            _write_list_op(getattr(obj, field), items, remap)


VISUALS_GROUP = "visuals"
COLLISIONS_GROUP = "collisions"


def _geometry_role(prim) -> str | None:
    """``"collisions"`` / ``"visuals"`` for a geometry child of a link, else ``None``.

    Gprims are classified by ``UsdPhysics.CollisionAPI``. Plain Xform wrappers
    are classified by their Gprim descendants when those agree; mixed wrappers,
    joints, nested bodies, and other prims return ``None`` (left in place).
    """
    from pxr import Usd, UsdGeom, UsdPhysics

    if prim.HasAPI(UsdPhysics.RigidBodyAPI) or prim.IsA(UsdPhysics.Joint):
        return None
    if prim.IsA(UsdGeom.Gprim):
        return COLLISIONS_GROUP if prim.HasAPI(UsdPhysics.CollisionAPI) else VISUALS_GROUP
    if prim.GetTypeName() != "Xform":
        return None
    roles = set()
    for desc in Usd.PrimRange(prim):
        if desc.HasAPI(UsdPhysics.RigidBodyAPI) or desc.IsA(UsdPhysics.Joint):
            return None
        if desc.IsA(UsdGeom.Gprim):
            roles.add(
                COLLISIONS_GROUP if desc.HasAPI(UsdPhysics.CollisionAPI) else VISUALS_GROUP
            )
    return roles.pop() if len(roles) == 1 else None


def group_body_geometry(usd_path: str | Path) -> dict[str, str]:
    """Group each rigid body's geometry into ``visuals`` / ``collisions`` children.

    Matches Isaac Sim's importer layout (``/Robot/<link>/visuals/...``,
    ``/Robot/<link>/collisions/...``), which Isaac-side consumers look up by
    path (e.g. active-adaptation's mesh extraction for cameras / raycasters).
    Groups are identity-transform Xforms, so world poses are unchanged; path
    references into moved prims (material bindings, ...) are rewritten.
    Bodies that already have these groups are left alone. Requires a
    single-layer stage. Returns the ``{old_path: new_path}`` moves.
    """
    from pxr import Sdf, Usd, UsdPhysics

    stage, layer, root = _open_single_layer(usd_path)
    moves: dict[Sdf.Path, Sdf.Path] = {}
    groups: set[tuple[Sdf.Path, str]] = set()
    for body in Usd.PrimRange(root):
        if not body.HasAPI(UsdPhysics.RigidBodyAPI):
            continue
        names = {c.GetName() for c in body.GetChildren()}
        if VISUALS_GROUP in names or COLLISIONS_GROUP in names:
            continue
        for child in body.GetChildren():
            role = _geometry_role(child)
            if role is None:
                continue
            groups.add((body.GetPath(), role))
            moves[child.GetPath()] = body.GetPath().AppendChild(role).AppendChild(child.GetName())

    if not moves:
        return {}
    with Sdf.ChangeBlock():
        for body_path, role in groups:
            Sdf.PrimSpec(layer.GetPrimAtPath(body_path), role, Sdf.SpecifierDef, "Xform")
    _move_prims(layer, moves)
    layer.Save()
    return {str(k): str(v) for k, v in moves.items()}


JOINTS_GROUP = "joints"


def _remove_empty_scopes(stage, root) -> None:
    for child in list(root.GetChildren()):
        if child.GetTypeName() == "Scope" and not child.GetChildren():
            stage.RemovePrim(child.GetPath())


def group_joint_prims(usd_path: str | Path) -> dict[str, str]:
    """Move every physics joint under the robot to ``/<robot>/joints/<joint>``.

    Matches Isaac Sim's importer layout. Joint frames are defined relative to
    ``body0`` / ``body1``, so the move does not change the articulation; path
    references to joints (e.g. MuJoCo actuator targets) are rewritten. Clashing
    names become ``<parent>_<name>``. Scopes left empty are removed. Requires a single-layer stage. Returns the
    ``{old_path: new_path}`` moves.
    """
    from pxr import Sdf, Usd, UsdPhysics

    stage, layer, root = _open_single_layer(usd_path)
    group_path = root.GetPath().AppendChild(JOINTS_GROUP)
    existing = stage.GetPrimAtPath(group_path)
    if existing and existing.HasAPI(UsdPhysics.RigidBodyAPI):
        raise ValueError(f"Cannot group joints: {group_path} is a rigid body")

    joints = [
        p
        for p in Usd.PrimRange(root)
        if p.IsA(UsdPhysics.Joint) and p.GetPath().GetParentPath() != group_path
    ]
    counts = Counter(p.GetName() for p in joints)
    moves: dict[Sdf.Path, Sdf.Path] = {}
    taken = {c.GetName() for c in existing.GetChildren()} if existing else set()
    for prim in joints:
        name = prim.GetName()
        # mujoco-usd-converter names every weld "PhysicsFixedJoint"; qualify
        # clashing names with the prim they were defined under (the link).
        if counts[name] > 1 or name in taken:
            name = f"{prim.GetParent().GetName()}_{name}"
        if name in taken:
            raise ValueError(f"Cannot group joints: duplicate joint name {name!r} ({prim.GetPath()})")
        taken.add(name)
        moves[prim.GetPath()] = group_path.AppendChild(name)

    if not moves:
        return {}
    if not existing:
        Sdf.PrimSpec(layer.GetPrimAtPath(root.GetPath()), JOINTS_GROUP, Sdf.SpecifierDef, "Scope")
    _move_prims(layer, moves)
    _remove_empty_scopes(stage, root)
    layer.Save()
    return {str(k): str(v) for k, v in moves.items()}


def flatten_articulation(
    usd_path: str | Path,
    *,
    group_geometry: bool = True,
    group_joints: bool = True,
) -> dict[str, str]:
    """Reparent every rigid body to be a direct child of the default prim.

    The newton converters nest rigid bodies following the kinematic tree
    (``/robot/Geometry/base/hip/thigh/...``). Isaac Lab expects links as
    siblings under the robot prim (``/robot/base``, ``/robot/hip``, ...):
    PhysX does not support nested rigid bodies without an xform-stack reset,
    and prim-path regexes such as ``/Robot/.*_foot`` match one level only.

    World poses are preserved, and every relationship target / attribute
    connection pointing into a moved subtree (joint bodies, filtered pairs,
    material bindings, ...) is rewritten. Requires a single-layer stage (the
    converters' ``--no-layer-structure`` output). Scopes left empty are removed.
    Afterwards :func:`group_body_geometry` (``group_geometry``) and
    :func:`group_joint_prims` (``group_joints``) run, both on by default.
    Returns the body ``{old_path: new_path}`` moves.
    """
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics

    stage, layer, root = _open_single_layer(usd_path)
    root_path = root.GetPath()

    bodies = [
        p
        for p in Usd.PrimRange(root)
        if p.HasAPI(UsdPhysics.RigidBodyAPI) and p.GetPath().GetParentPath() != root_path
    ]
    if not bodies:
        del stage
        _post_group(usd_path, group_geometry, group_joints)
        return {}

    xform_cache = UsdGeom.XformCache()
    root_inv = xform_cache.GetLocalToWorldTransform(root).GetInverse()
    moves: dict[Sdf.Path, Sdf.Path] = {}
    local_pose: dict[Sdf.Path, Gf.Matrix4d] = {}
    taken = {c.GetName() for c in root.GetChildren()}
    for body in bodies:
        name = body.GetName()
        if name in taken:
            raise ValueError(
                f"Cannot flatten: {root_path}/{name} already exists (from {body.GetPath()})"
            )
        taken.add(name)
        new_path = root_path.AppendChild(name)
        moves[body.GetPath()] = new_path
        local_pose[new_path] = xform_cache.GetLocalToWorldTransform(body) * root_inv

    _move_prims(layer, moves)

    for new_path, matrix in local_pose.items():
        prim = stage.GetPrimAtPath(new_path)
        xformable = UsdGeom.Xformable(prim)
        xform = Gf.Transform(matrix)
        for op in xformable.GetOrderedXformOps():
            prim.RemoveProperty(op.GetName())
        xformable.ClearXformOpOrder()
        xformable.AddTranslateOp().Set(Gf.Vec3d(xform.GetTranslation()))
        xformable.AddOrientOp().Set(Gf.Quatf(xform.GetRotation().GetQuat()))
        scale = Gf.Vec3f(xform.GetScale())
        if not Gf.IsClose(scale, Gf.Vec3f(1.0), 1e-6):
            xformable.AddScaleOp().Set(scale)

    _remove_empty_scopes(stage, root)
    layer.Save()
    del stage
    _post_group(usd_path, group_geometry, group_joints)
    return {str(k): str(v) for k, v in moves.items()}


def _post_group(usd_path: str | Path, geometry: bool, joints: bool) -> None:
    if geometry:
        group_body_geometry(usd_path)
    if joints:
        group_joint_prims(usd_path)


def main(argv: list[str] | None = None) -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("usd", type=Path)
    parser.add_argument("--remove-world-fixed-joints", action="store_true")
    parser.add_argument("--flatten", action="store_true")
    args = parser.parse_args(argv)
    if args.remove_world_fixed_joints:
        for path in remove_world_fixed_joints(args.usd):
            print(f"Removed world-fixed joint {path}")
    if args.flatten:
        moves = flatten_articulation(args.usd)
        print(f"Flattened {len(moves)} nested rigid bodies under the robot prim")


# Run standalone (``python postprocess.py``) inside the converter's environment,
# so hosts without their own ``pxr`` (e.g. Isaac Sim before Kit starts) can
# still post-process. Keep this module free of assetx / third-party imports.
if __name__ == "__main__":
    main()
