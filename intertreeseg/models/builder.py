"""Model factory: the single place PointTransformerV3 is constructed.

``in_channels`` is derived from the data channel spec (via :class:`ChannelSpec`),
so it can never drift from what the dataset produces. The full ``model.backbone``
config block is consumed here — unlike the legacy code, which ignored it and
hardcoded backbone parameters in three separate call sites.
"""

from __future__ import annotations

from typing import Any, Mapping

from ..data.channels import ChannelSpec
from .ptv3 import PointTransformerV3

# Backbone keys forwarded verbatim from cfg.model.backbone to the PTv3 constructor.
# `in_channels` is intentionally excluded (derived), as are the four *_patch_size
# and *_conditions lists which need tuple coercion (handled below).
_PASSTHROUGH_KEYS = (
    "mlp_ratio",
    "qkv_bias",
    "qk_scale",
    "attn_drop",
    "proj_drop",
    "drop_path",
    "pre_norm",
    "shuffle_orders",
    "enable_rpe",
    "enable_flash",
    "upcast_attention",
    "upcast_softmax",
    "cls_mode",
    "pdnorm_bn",
    "pdnorm_ln",
    "pdnorm_decouple",
    "pdnorm_adaptive",
    "pdnorm_affine",
)

_TUPLE_KEYS = (
    "order",
    "stride",
    "enc_depths",
    "enc_channels",
    "enc_num_head",
    "enc_patch_size",
    "dec_depths",
    "dec_channels",
    "dec_num_head",
    "dec_patch_size",
    "pdnorm_conditions",
)


def build_model(cfg: Mapping[str, Any]) -> PointTransformerV3:
    """Construct a :class:`PointTransformerV3` from a full config.

    ``in_channels`` is derived from ``cfg.data`` through :class:`ChannelSpec`.
    If ``cfg.model.backbone.in_channels`` is also present it must match, else a
    ``ValueError`` is raised (guards against silent config drift).
    """
    spec = ChannelSpec.from_cfg(cfg["data"])
    in_channels = spec.in_channels

    model_cfg = cfg["model"]
    bb = model_cfg["backbone"]

    pinned = bb.get("in_channels", None)
    if pinned is not None and int(pinned) != in_channels:
        raise ValueError(
            f"config drift: model.backbone.in_channels={pinned} but the data "
            f"channel spec derives in_channels={in_channels} "
            f"(feat_channels={list(spec.feat_channels)}, "
            f"n_click_channels={spec.n_click_channels}). Remove the pin or fix the data block."
        )

    kwargs: dict[str, Any] = {
        "in_channels": in_channels,
        "num_classes": int(model_cfg["num_classes"]),
    }
    for key in _TUPLE_KEYS:
        if key in bb and bb.get(key) is not None:
            kwargs[key] = tuple(bb[key])
    for key in _PASSTHROUGH_KEYS:
        if key in bb and bb.get(key) is not None:
            kwargs[key] = bb[key]

    return PointTransformerV3(**kwargs)
