"""Channel contract: the single source of truth for the input feature layout.

Both the dataset (feature assembly) and the model factory (``in_channels``
sizing) read the same :class:`ChannelSpec`, so the network's input width can
never drift from what the data actually produces.

Canonical per-block layout (for ``F = len(feat_channels)`` raw feature channels
and ``K = n_click_channels`` click channels)::

    column index:  0    1    2   | 3 .. 3+F-1        | 3+F .. 3+F+K-1
    content:       x    y    z    | raw feat channels | click channels
                   (coord)        | (intensity, ...)  | (pos, neg, ...)
    width = 3 + F + K

Model input slicing (unchanged from the legacy ``points[:, :, 3:]`` idiom)::

    coord = block[..., 0:3]      # -> data_dict['coord']
    feat  = block[..., 3:]       # width F + K = in_channels -> data_dict['feat']

Default (``feat_channels=[]``, ``K=2``) gives ``in_channels=2`` and
``click_start=3`` — byte-for-byte identical to the original pipeline, which is
what keeps the released checkpoints loadable.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence, Tuple, Union

import numpy as np

# A per-channel normalization spec is either a bare name ("none", "standardize",
# "minmax") or a ("scale", factor) pair.
NormSpec = Union[str, Tuple[str, float]]


def _as_int_tuple(seq: Sequence[int] | None) -> Tuple[int, ...]:
    if seq is None:
        return ()
    return tuple(int(x) for x in seq)


def _normalize_norm_specs(specs: Sequence[Any] | None, n: int) -> Tuple[NormSpec, ...]:
    """Coerce a raw feat_norm config into a length-``n`` tuple of hashable specs."""
    if not specs:
        return tuple(["none"] * n)
    out: list[NormSpec] = []
    for s in specs:
        if isinstance(s, (list, tuple)):
            name, param = s[0], float(s[1])
            out.append((str(name), param))
        else:
            out.append(str(s))
    if len(out) != n:
        raise ValueError(
            f"feat_norm has {len(out)} entries but feat_channels has {n}; "
            f"provide one norm spec per feature channel (or leave empty for 'none')."
        )
    return tuple(out)


@dataclass(frozen=True)
class ChannelSpec:
    """Immutable description of how source columns map to model inputs."""

    coord_channels: Tuple[int, ...] = (0, 1, 2)
    feat_channels: Tuple[int, ...] = ()
    n_click_channels: int = 2
    feat_norm: Tuple[NormSpec, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if len(self.coord_channels) != 3:
            raise ValueError(
                f"coord_channels must have exactly 3 entries (xyz), got {self.coord_channels}"
            )
        if self.n_click_channels < 0:
            raise ValueError("n_click_channels must be >= 0")
        # Fill / validate feat_norm to match feat_channels length.
        object.__setattr__(
            self, "feat_norm", _normalize_norm_specs(self.feat_norm, len(self.feat_channels))
        )

    # --- derived quantities (never hand-written elsewhere) ---
    @property
    def F(self) -> int:
        """Number of raw (non-click) feature channels."""
        return len(self.feat_channels)

    @property
    def click_start(self) -> int:
        """Index of the first click channel within the assembled block."""
        return 3 + self.F

    @property
    def in_channels(self) -> int:
        """Feature width fed to the model embedding stem (= F + K)."""
        return self.F + self.n_click_channels

    @property
    def block_width(self) -> int:
        """Total column count of an assembled block (xyz + feats + clicks)."""
        return 3 + self.F + self.n_click_channels

    # --- construction from config ---
    @classmethod
    def from_cfg(cls, data_cfg: Mapping[str, Any]) -> "ChannelSpec":
        """Build a ChannelSpec from a config ``data`` block.

        Reads ``coord_channels``, ``feat_channels``, ``feat_norm`` and
        ``clicks.n_channels``.
        """
        clicks = data_cfg.get("clicks", {}) or {}
        return cls(
            coord_channels=_as_int_tuple(data_cfg.get("coord_channels", (0, 1, 2))),
            feat_channels=_as_int_tuple(data_cfg.get("feat_channels", ())),
            n_click_channels=int(clicks.get("n_channels", 2)),
            feat_norm=tuple(data_cfg.get("feat_norm", ()) or ()),
        )

    # --- source-column selection helpers ---
    def select_coord(self, raw: np.ndarray) -> np.ndarray:
        """Slice the xyz columns out of a raw source array ``(..., C_source)``."""
        return raw[..., list(self.coord_channels)]

    def select_feat(self, raw: np.ndarray) -> np.ndarray:
        """Slice the raw feature columns out of a raw source array.

        Returns an ``(..., F)`` array (possibly ``F == 0`` -> shape ``(..., 0)``).
        """
        if self.F == 0:
            return raw[..., :0]
        return raw[..., list(self.feat_channels)]
