"""Data pipeline: channel contract, normalization, augmentation, clicks, datasets.

The dataset builders pull in heavy, optional dependencies (``h5py`` for the h5
training set, ``pandas`` for scene txt). They are imported lazily so that using
the *model* alone — e.g. a serving backend that only needs
``from intertreeseg.models import PointTransformerV3`` — does not require the
data-loading dependencies to be installed.
"""
from .channels import ChannelSpec
from .clicks import gen_clicks, gen_click_center, reclick

# name -> submodule, imported on first access (PEP 562)
_LAZY = {
    "build_h5_dataset": ".h5_dataset",
    "load_scene_txt": ".scene_dataset",
    "build_dataloaders": ".loader",
}


def __getattr__(name):
    module = _LAZY.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib

    return getattr(importlib.import_module(module, __name__), name)


def __dir__():
    return sorted(list(globals().keys()) + list(_LAZY.keys()))


__all__ = [
    "ChannelSpec",
    "gen_clicks",
    "gen_click_center",
    "reclick",
    "build_h5_dataset",
    "load_scene_txt",
    "build_dataloaders",
]
