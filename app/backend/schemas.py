"""Request/response models for the REST + WebSocket API (pydantic v2).

NOTE: the Point env is Python 3.9, whose runtime does not accept PEP-604
`X | None` unions when pydantic evaluates field annotations. Use typing.Optional
here (dataclasses elsewhere keep the newer syntax via `from __future__`).
"""
from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field


class LoadRequest(BaseModel):
    file_path: str


class LoadResponse(BaseModel):
    session_id: str
    num_points: int          # working (possibly decimated) point count
    num_points_full: int
    decimated: bool
    bounds: dict


class ClickModel(BaseModel):
    x: float
    y: float
    z: float
    positive: bool = True


class BBox(BaseModel):
    x_min: float
    y_min: float
    x_max: float
    y_max: float
    z_min: Optional[float] = None
    z_max: Optional[float] = None


class InferRequest(BaseModel):
    bbox: Optional[BBox] = None
    clicks: List[ClickModel] = Field(default_factory=list)


class InferResponse(BaseModel):
    mask_indices: List[int]
    count: int
    elapsed_ms: float


class AttributeModel(BaseModel):
    id: Optional[int] = None
    species: Optional[str] = None
    dbh: Optional[float] = None
    health_status: Optional[str] = None
    notes: Optional[str] = None
    height: Optional[float] = None
    crown_width: Optional[float] = None
    stem_x: Optional[float] = None
    stem_y: Optional[float] = None


class CommitRequest(BaseModel):
    # No point list: the server commits its current full-resolution working mask.
    attributes: AttributeModel = Field(default_factory=AttributeModel)
    # interaction stats for evaluation (recorded per tree)
    n_clicks: Optional[int] = None
    elapsed_s: Optional[float] = None


class ScreenSelection(BaseModel):
    """A 2D lasso/brush selection + the camera needed to project full-res points."""
    view_proj: List[float]                     # three.js Matrix4.elements (16, column-major)
    vw: float
    vh: float
    kind: str                                  # "polygon" | "disc"
    polygon: Optional[List[List[float]]] = None
    cx: Optional[float] = None
    cy: Optional[float] = None
    r: Optional[float] = None


class EditMaskRequest(BaseModel):
    selection: ScreenSelection
    add: bool = True                           # add to / remove from the current mask


class AssignScreenRequest(BaseModel):
    selection: ScreenSelection
    target: int = 0                            # instance id, or 0 for unlabeled


class ProgressState(BaseModel):
    source_path: str
    labels: List[int]
    instances: List[dict]
    next_id: int
