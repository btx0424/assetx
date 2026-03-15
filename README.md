# assetx

**Composable, reproducible robot descriptions from base assets and recipes.**

## Introduction

Robot descriptions today are often single, hand-edited files. Changing one robot or reusing a sub-assembly means copying XML, fixing paths, and hoping nothing breaks. Sharing a “variant” means sharing another full copy. Reproducibility is hard because the exact steps from sources to final model are unclear.

The goal of **assetx** is to turn this into a simple pipeline:

1. **Base assets** — Canonical descriptions of building blocks (e.g. a mobile base, an arm, a gripper). Today these are MJCF models; the idea generalizes to other formats. Base assets are the single source of truth: versioned, shared, and not edited per-robot.

2. **Recipes** — A declarative or programmatic description of how to build a robot from those assets. A recipe does two things:
   - **Assemble**: attach assets together (e.g. mount an arm on a base at a given link, with a given pose and joint type).
   - **Transform**: apply structural edits to the combined model (merge bodies, replace geometry types, remove subtrees, rename bodies, etc.).
   Recipes can be reused across different base assets: for example, a single “mount arm” recipe can be applied to different quadrupeds to produce multiple robot variants.

3. **Final robot** — The output of the recipe is a complete robot description. Because it is fully determined by “these base assets + this recipe,” the same inputs always produce the same output. Reproducibility comes from specifying the pipeline, not from sharing a frozen file.

By making composition explicit (assemble + transform), we get:

- **Reproducibility**: Same base assets and recipe → same robot. No hidden manual edits.
- **Easier composition**: Reuse and combine base assets instead of copying and patching monolithic URDF/MJCF.
- **Recipe reuse**: The same recipe can be applied to different base assets (e.g. a “mount arm” recipe on different quadrupeds) to get many robot variants from one pipeline.
- **Clear provenance**: The final model is traceable back to specific assets and recipe steps.

The current implementation is a step toward this vision: it already supports assembling MJCF assets and a set of transforms, but the pipeline, recipe format, and asset lifecycle are still evolving. This README describes the direction we want to reach.
