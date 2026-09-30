"""Built-in robot recipes. Importing this package registers them."""

from assetx.recipes.registry import (
    Recipe,
    cache_dir,
    fetch_vendor,
    get_recipe,
    list_recipes,
    recipe,
)
from assetx.recipes import spot  # noqa: F401  (registers recipes)

__all__ = [
    "Recipe",
    "cache_dir",
    "fetch_vendor",
    "get_recipe",
    "list_recipes",
    "recipe",
]
