"""Session state: one loaded scene + its instance labels and attributes.

Full resolution is the source of truth: labels, the spatial index, inference,
and export all operate on the full-resolution cloud. A decimated copy exists
ONLY for rendering in the browser; `render_to_full` maps a rendered point back
to its full-resolution index. This keeps annotation quality independent of the
render decimation — every original point gets an accurate label.
"""
from __future__ import annotations

import os
import threading
import uuid
from dataclasses import dataclass, field, asdict

import numpy as np

from . import config
from .geometry import compute_geometry
from .io.reader import get_reader
from .spatial.index import SpatialIndex

_GEOM_FIELDS = ("height", "crown_width", "stem_x", "stem_y")


@dataclass
class Instance:
    id: int
    color: list[float]
    species: str = "Unknown"
    dbh: float | None = None
    health_status: str = "Alive"
    notes: str = ""
    height: float = 0.0
    crown_width: float = 0.0
    stem_x: float = 0.0
    stem_y: float = 0.0
    point_count: int = 0
    manual_fields: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


def _instance_color(idx: int) -> list[float]:
    import colorsys
    h = (idx * 0.61803398875) % 1.0
    r, g, b = colorsys.hsv_to_rgb(h, 0.65, 0.95)
    return [round(r, 4), round(g, 4), round(b, 4)]


class Session:
    def __init__(self, session_id: str, source_path: str):
        self.id = session_id
        self.source_path = source_path
        self.lock = threading.RLock()

        raw = get_reader(source_path).read(source_path)   # (N_full, C)
        self.full_points = raw
        self.n_full = raw.shape[0]
        self.full_xyz = np.ascontiguousarray(raw[:, :3], dtype=np.float64)

        # authoritative full-resolution labels + spatial index
        self.full_labels = np.zeros(self.n_full, dtype=np.int32)
        self.full_index = SpatialIndex(self.full_xyz)

        # decimated render set (rendering only) — sampled to an EXACT fixed target
        # point count (evenly spaced across the cloud), so render cost is bounded
        # regardless of input size. Labels/inference/export stay full-resolution.
        self.decimated = self.n_full > config.MAX_RENDER_POINTS
        if self.decimated:
            self.render_to_full = np.linspace(
                0, self.n_full - 1, config.MAX_RENDER_POINTS
            ).astype(np.int64)
        else:
            self.render_to_full = np.arange(self.n_full)
        self.n = self.render_to_full.shape[0]
        self.points = raw[self.render_to_full]            # decimated (n, C)

        self.instances: dict[int, Instance] = {}
        self._next_id = 1
        # current (uncommitted) working tree mask, as FULL-RES indices
        self.current_mask = np.empty(0, dtype=np.int64)
        # the current bbox block (full-res indices) — manual edits are restricted to it
        self.current_block = np.empty(0, dtype=np.int64)

    # ---- scene meta --------------------------------------------------------
    def bounds(self) -> dict:
        return {"min": self.full_xyz.min(axis=0).tolist(), "max": self.full_xyz.max(axis=0).tolist()}

    def render_xyz(self) -> np.ndarray:
        return self.points[:, :3]

    def display_labels(self) -> np.ndarray:
        """Instance id per RENDERED point (derived from full labels)."""
        return self.full_labels[self.render_to_full]

    def full_mask_to_display(self, full_idx: np.ndarray) -> np.ndarray:
        """Rendered-point indices whose full-res point is in the given full mask."""
        flag = np.zeros(self.n_full, dtype=bool)
        flag[full_idx] = True
        return np.nonzero(flag[self.render_to_full])[0]

    # ---- inference working mask -------------------------------------------
    def set_current_block(self, block_idx: np.ndarray) -> None:
        with self.lock:
            self.current_block = np.asarray(block_idx, dtype=np.int64)

    def set_current_mask(self, full_idx: np.ndarray) -> np.ndarray:
        with self.lock:
            self.current_mask = np.asarray(full_idx, dtype=np.int64)
            return self.full_mask_to_display(self.current_mask)

    def edit_current_mask(self, full_idx: np.ndarray, add: bool) -> np.ndarray:
        with self.lock:
            full_idx = np.asarray(full_idx, dtype=np.int64)
            # restrict manual edits to the current bbox block (not the whole scene)
            if self.current_block.size > 0:
                flag = np.zeros(self.n_full, dtype=bool)
                flag[self.current_block] = True
                full_idx = full_idx[flag[full_idx]]
            cur = set(self.current_mask.tolist())
            for i in full_idx.tolist():
                cur.add(i) if add else cur.discard(i)
            self.current_mask = np.fromiter(cur, dtype=np.int64, count=len(cur))
            return self.full_mask_to_display(self.current_mask)

    def clear_current_mask(self) -> None:
        with self.lock:
            self.current_mask = np.empty(0, dtype=np.int64)
            self.current_block = np.empty(0, dtype=np.int64)

    # ---- instances ---------------------------------------------------------
    def commit_current(self, attributes: dict) -> tuple[Instance, np.ndarray]:
        """Commit the current working mask as a new instance. Returns (instance, display_idx)."""
        with self.lock:
            if self.current_mask.size == 0:
                raise ValueError("No current mask to commit")
            iid = int(attributes.get("id") or self._next_id)
            if iid in self.instances:
                raise ValueError(f"Instance id {iid} already exists")
            self._next_id = max(self._next_id, iid) + 1

            self.full_labels[self.current_mask] = iid
            inst = Instance(id=iid, color=_instance_color(iid))
            for key in ("species", "dbh", "health_status", "notes"):
                if attributes.get(key) is not None:
                    setattr(inst, key, attributes[key])
            self._apply_geometry(inst, self.current_mask)
            self.instances[iid] = inst
            display = self.full_mask_to_display(self.current_mask)
            self.current_mask = np.empty(0, dtype=np.int64)
            self.current_block = np.empty(0, dtype=np.int64)
            return inst, display

    def _apply_geometry(self, inst: Instance, full_idx: np.ndarray) -> None:
        geom = compute_geometry(self.full_xyz[full_idx])
        inst.point_count = geom["point_count"]
        for f in _GEOM_FIELDS:
            if f not in inst.manual_fields:
                setattr(inst, f, geom[f])

    def update_instance(self, iid: int, attributes: dict) -> Instance:
        with self.lock:
            inst = self.instances[iid]
            for key, val in attributes.items():
                if not hasattr(inst, key) or key in ("id", "color", "manual_fields", "point_count"):
                    continue
                setattr(inst, key, val)
                if key in _GEOM_FIELDS and key not in inst.manual_fields:
                    inst.manual_fields.append(key)
            return inst

    def rename_instance(self, old_id: int, new_id: int) -> Instance:
        """Change a tree's instance id, relabeling all its full-res points."""
        with self.lock:
            if old_id not in self.instances:
                raise KeyError(old_id)
            if new_id == old_id:
                return self.instances[old_id]
            if new_id in self.instances:
                raise ValueError(f"Instance id {new_id} already exists")
            if new_id <= 0:
                raise ValueError("Instance id must be a positive integer")
            self.full_labels[self.full_labels == old_id] = new_id
            inst = self.instances.pop(old_id)
            inst.id = new_id  # keep the existing color so the tree looks unchanged
            self.instances[new_id] = inst
            self._next_id = max(self._next_id, new_id + 1)
            return inst

    def delete_instance(self, iid: int) -> None:
        with self.lock:
            self.full_labels[self.full_labels == iid] = 0
            self.instances.pop(iid, None)

    def assign_full(self, full_idx: np.ndarray, target: int) -> list[int]:
        """Manual edit of committed labels: set full-res points to `target`
        (0 = unlabeled). Returns the affected instance ids (for geometry refresh)."""
        with self.lock:
            full_idx = np.asarray(full_idx, dtype=np.int64)
            affected = set(np.unique(self.full_labels[full_idx]).tolist())
            self.full_labels[full_idx] = target
            affected.add(target)
            for iid in affected:
                if iid in self.instances:
                    self._recompute_one(iid)
            return sorted(affected)

    def _recompute_one(self, iid: int) -> None:
        idx = np.nonzero(self.full_labels == iid)[0]
        inst = self.instances[iid]
        if idx.size == 0:
            inst.point_count = 0
            return
        self._apply_geometry(inst, idx)

    def recompute_geometry(self) -> list[Instance]:
        with self.lock:
            for iid in self.instances:
                self._recompute_one(iid)
            return list(self.instances.values())

    # ---- persistence -------------------------------------------------------
    def progress_state(self) -> dict:
        return {
            "source_path": self.source_path,
            "labels": self.full_labels.tolist(),
            "instances": [i.to_dict() for i in self.instances.values()],
            "next_id": self._next_id,
        }

    def load_progress(self, state: dict) -> None:
        with self.lock:
            labels = np.asarray(state["labels"], dtype=np.int32)
            if labels.shape[0] != self.n_full:
                raise ValueError("Progress labels length does not match scene")
            self.full_labels = labels
            self.instances = {}
            for d in state["instances"]:
                d = dict(d)
                inst = Instance(id=d["id"], color=d.get("color") or _instance_color(d["id"]))
                for k, v in d.items():
                    if hasattr(inst, k):
                        setattr(inst, k, v)
                self.instances[inst.id] = inst
            self._next_id = int(state.get("next_id", (max(self.instances) + 1) if self.instances else 1))


class SessionManager:
    def __init__(self):
        self._sessions: dict[str, Session] = {}
        self._lock = threading.Lock()

    def create(self, source_path: str) -> Session:
        if not os.path.isfile(source_path):
            raise FileNotFoundError(source_path)
        sid = uuid.uuid4().hex[:12]
        session = Session(sid, source_path)
        with self._lock:
            self._sessions[sid] = session
        return session

    def get(self, sid: str) -> Session:
        with self._lock:
            if sid not in self._sessions:
                raise KeyError(sid)
            return self._sessions[sid]

    def close(self, sid: str) -> None:
        with self._lock:
            self._sessions.pop(sid, None)


manager = SessionManager()
