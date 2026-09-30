---
name: collision-simplification
description: Replace elongated MJCF collision meshes with capsules in assetx recipes using ApproximateWithCapsule, and judge the fit from assetx render images rather than voxel IoU alone. Use when simplifying collisions, approximating *_uleg, *_lleg, or arm-link meshes, tuning coverage_quantile or radius_scale, or deciding which geoms must stay meshes (torso, jaws, teeth, feet).
---

# Collision simplification

Replace a collision mesh with a capsule only when the mesh is a straight tube and the rendered capsule still matches the visual mesh. Voxel IoU is a screen, not the decision.

## Fit

Use `ApproximateWithCapsule` in the recipe, after geom names are normalized. One group per mesh. Keep the original geom name so `.*_collision.*` patterns still match. The transform copies `classname`, so `class="collision"` (group, solref) survives.

```python
ApproximateWithCapsule(
    [["fl_lleg_collision0"]],
    names=["fl_lleg_collision0"],
    replace=True,
    coverage_quantile=0.95,  # drop corner outliers; recall stays ~0.99
)
```

- Start with `num_capsules=1`. A second capsule is worth it only when the part bends and the render shows a real gap. On Spot's straight legs a chain did not improve the fit.
- `coverage_quantile=0.95` is the default to try. `1.0` inflates the radius to cover a few corner vertices.
- Surface sampling is seeded (`mesh_surface_samples_local`, `seed=0`). Do not unseed it; cooks must be reproducible.
- Front and rear legs that share one mesh get the same fit. Left and right vendor meshes are not exact mirrors, so do not force them to be.
- Bodies with an explicit `<inertial>` do not change mass when the mesh is swapped. Do not retune inertia for the capsule.

`radius_scale` is a visual correction applied after the first render, not a default. Spot's upper-leg capsules use `0.6` because full coverage was thicker than the yellow visual mesh. A flatter link (the shin is 4.6 cm by 7.4 cm) needs a scale closer to 1 so the narrow side is not left outside the capsule.

## Decide from the image

```bash
uv run assetx render <recipe> --out /tmp/<recipe>-preview
# headless: MUJOCO_GL=egl
```

`visual.png` is the visual meshes. `collision.png` colors each body's collision geoms. Compare the two. Reject a fit that rounds into a neighbor or fills a gap the mesh left open, even if its IoU is high.

Worked screen on Spot / Spot arm (95% coverage, one capsule):

| Geom | Elongation | IoU | Verdict |
|------|------------|-----|---------|
| `*_uleg`, `*_lleg` | ~3–5 | ~0.5–0.7 | Replace. One capsule; a chain did not help. |
| `arm_link_hr0`, `arm_link_el1` main mesh | ~3–5 | ~0.6 | Replace the tube. Leave `arm_link_el1`'s lip mesh; it is a flat flange. |
| `base_link` | 3.5 | 0.73 | Keep the mesh. A capsule rounds the torso into the legs and the arm. |
| `arm_link_el0`, `arm_link_sh0` | ~1 | <0.5 | Keep. Elbow and motor housings are not tubes. |
| `arm_link_wr0`, `arm_link_wr1`, `arm_link_fngr` | ~1 | <0.5 | Keep. Jaws and teeth define the grasp; a capsule fills the finger gap. |

Feet are already spheres. Hip bodies have no collision geoms; do not invent any as part of a simplification.

## After editing a recipe

The recipe module is part of the cook hash. Re-cook and check:

```bash
uv run assetx cook <recipe> --out-root <ROBOT_MODEL_DIR>
uv run assetx render <recipe>
```

`core/preview.py` is outside the hash, so render-only edits do not stale bundles.
