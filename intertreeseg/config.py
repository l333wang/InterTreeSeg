"""Configuration loading and merging for InterTreeSeg.

A single YAML file drives the whole pipeline: model backbone, data / channel
selection, clicks, optimizer / scheduler, training, and inference. This module
loads that YAML into an attribute-accessible :class:`addict.Dict`, supports
deep-merging (base config + variant overrides), and dotted-key CLI overrides
(``--set model.num_classes=3``).

Design note: unlike the legacy code, the ``model.backbone`` block is *not*
ignored — :func:`intertreeseg.models.build_model` consumes it in full. See
``configs/default.yaml`` for the schema.
"""

from __future__ import annotations

import copy
from typing import Any, Mapping

import yaml
from addict import Dict


def _to_plain(obj: Any) -> Any:
    """Recursively convert addict.Dict back to plain dict/list (for dumping)."""
    if isinstance(obj, Mapping):
        return {k: _to_plain(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_plain(v) for v in obj]
    return obj


def deep_merge(base: Mapping, override: Mapping) -> dict:
    """Recursively merge ``override`` into ``base`` (override wins), returning a new dict.

    Nested mappings are merged key-by-key; any non-mapping value in ``override``
    replaces the corresponding value in ``base`` wholesale (e.g. a list of
    channels is replaced, not concatenated).
    """
    result = dict(copy.deepcopy(dict(base)))
    for key, val in override.items():
        if (
            key in result
            and isinstance(result[key], Mapping)
            and isinstance(val, Mapping)
        ):
            result[key] = deep_merge(result[key], val)
        else:
            result[key] = copy.deepcopy(val)
    return result


def _coerce_scalar(text: str) -> Any:
    """Parse a CLI override value using YAML rules (int/float/bool/null/list/str)."""
    return yaml.safe_load(text)


def apply_dotted_overrides(cfg: dict, overrides: Mapping[str, str]) -> dict:
    """Apply ``{"a.b.c": "value"}`` overrides in-place-ish, returning the dict.

    Values are parsed with YAML semantics so ``model.num_classes=3`` becomes an
    int and ``data.feat_channels=[3,4]`` becomes a list.
    """
    for dotted, raw in overrides.items():
        keys = dotted.split(".")
        node = cfg
        for k in keys[:-1]:
            if k not in node or not isinstance(node[k], dict):
                node[k] = {}
            node = node[k]
        node[keys[-1]] = _coerce_scalar(raw) if isinstance(raw, str) else raw
    return cfg


def load_config(
    path: str,
    base: str | None = None,
    overrides: Mapping[str, str] | None = None,
) -> Dict:
    """Load a YAML config into an attribute-accessible :class:`addict.Dict`.

    Parameters
    ----------
    path:
        Path to the YAML config to load.
    base:
        Optional base config to deep-merge under ``path`` (``path`` wins). Use
        this for variant configs that only specify a diff from ``default.yaml``.
    overrides:
        Optional dotted-key overrides applied last (e.g. from ``--set``).
    """
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    if base is not None:
        with open(base, "r", encoding="utf-8") as f:
            base_data = yaml.safe_load(f) or {}
        data = deep_merge(base_data, data)

    if overrides:
        data = apply_dotted_overrides(data, dict(overrides))

    return Dict(data)


def dump_config(cfg: Mapping, path: str) -> None:
    """Write ``cfg`` to ``path`` as YAML (converts addict.Dict to plain dict)."""
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(_to_plain(cfg), f, sort_keys=False, allow_unicode=True)
