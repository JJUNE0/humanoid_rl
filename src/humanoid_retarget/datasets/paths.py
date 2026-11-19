"""Portable motion roots; manifest records retain their original relative names."""

from pathlib import Path


def manifest_root(settings, repository):
    value = Path(settings.get("manifest_root", repository))
    return (Path(repository) / value).resolve()


def relocate_motion(settings, *, root=None, manifest=None, cache=None):
    """Change storage locations without changing motions, splits, or fingerprints."""
    settings = dict(settings)
    if root is not None:
        settings["root"] = str(root)
        settings["manifest_root"] = str(root)
    if manifest is not None:
        settings["manifest"] = str(manifest)
    if cache is not None:
        settings["cache_dir"] = str(cache)
    return settings
