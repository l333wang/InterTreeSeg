"""Session state: one loaded scene + its instance labels and attributes.

Full resolution is the source of truth: labels, the spatial index, inference,
and export all operate on the full-resolution cloud. A decimated copy exists
ONLY for rendering in the browser; `render_to_full` maps a rendered point back
to its full-resolution index. This keeps annotation quality independent of the
render decimation — every original point gets an accurate label.
"""
from __future__ import annotations

import gc
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

        # ground-truth per-point instance id (for IoU evaluation), if the file
        # has that column; None when the scene carries no GT.
        col = config.GT_INSTANCE_COL
        self.gt_instance = (
            raw[:, col].astype(np.int64) if raw.shape[1] > col else None
        )
        # per-confirmed-tree evaluation records (clicks / time / IoU vs GT)
        self.eval_records: list[dict] = []
        # total hands-on annotation time (seconds) from the manual Start/Stop timer
        self.session_time_s: float | None = None
        # undo/redo snapshots of the editable state
        self._undo: list[dict] = []
        self._redo: list[dict] = []

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

    def field_values(self, name: str) -> np.ndarray:
        """Per-RENDER-point scalar for a coloring field.

        name: height (z) | intensity | gt_instance (treeID) | gt_semantic.
        Missing columns fall back to zeros.
        """
        ncol = self.full_points.shape[1]
        if name == "intensity":
            c = config.TXT_COLS.get("feat", 3)
            col = self.full_points[:, c] if ncol > c else np.zeros(self.n_full)
        elif name == "gt_instance":
            col = self.gt_instance.astype(np.float64) if self.gt_instance is not None else np.zeros(self.n_full)
        elif name == "gt_semantic":
            c = config.TXT_COLS.get("gt_semantic", 5)
            col = self.full_points[:, c] if ncol > c else np.zeros(self.n_full)
        else:  # height
            col = self.full_xyz[:, 2]
        return np.ascontiguousarray(col[self.render_to_full], dtype=np.float32)

    def display_labels(self) -> np.ndarray:
        """Instance id per RENDERED point (derived from full labels)."""
        return self.full_labels[self.render_to_full]

    def full_mask_to_display(self, full_idx: np.ndarray) -> np.ndarray:
        """Rendered-point indices whose full-res point is in the given full mask."""
        flag = np.zeros(self.n_full, dtype=bool)
        flag[full_idx] = True
        return np.nonzero(flag[self.render_to_full])[0]

    # ---- undo / redo -------------------------------------------------------
    _MAX_UNDO = 30

    def _snapshot(self) -> dict:
        return {
            "labels": self.full_labels.copy(),
            "mask": self.current_mask.copy(),
            "block": self.current_block.copy(),
            "instances": {i: Instance(**asdict(x)) for i, x in self.instances.items()},
            "next_id": self._next_id,
            "eval": [dict(r) for r in self.eval_records],
            "session_time_s": self.session_time_s,
        }

    def _apply(self, snap: dict) -> None:
        self.full_labels = snap["labels"].copy()
        self.current_mask = snap["mask"].copy()
        self.current_block = snap["block"].copy()
        self.instances = {i: Instance(**asdict(x)) for i, x in snap["instances"].items()}
        self._next_id = snap["next_id"]
        self.eval_records = [dict(r) for r in snap["eval"]]
        self.session_time_s = snap["session_time_s"]

    def push_undo(self) -> None:
        """Snapshot the current editable state before a mutating operation."""
        with self.lock:
            self._undo.append(self._snapshot())
            if len(self._undo) > self._MAX_UNDO:
                self._undo.pop(0)
            self._redo.clear()

    def undo(self) -> bool:
        with self.lock:
            if not self._undo:
                return False
            self._redo.append(self._snapshot())
            self._apply(self._undo.pop())
            return True

    def redo(self) -> bool:
        with self.lock:
            if not self._redo:
                return False
            self._undo.append(self._snapshot())
            self._apply(self._redo.pop())
            return True

    def current_display_mask(self) -> np.ndarray:
        return self.full_mask_to_display(self.current_mask)

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

    def toggle_current_mask(self, full_idx: np.ndarray) -> np.ndarray:
        """Brush-toggle: flip each point's membership in the working mask.

        A point in the mask (label) is removed (-> background); a point not in it
        is added. Restricted to the current bbox block, like manual edits.
        """
        with self.lock:
            full_idx = np.asarray(full_idx, dtype=np.int64)
            if self.current_block.size > 0:
                flag = np.zeros(self.n_full, dtype=bool)
                flag[self.current_block] = True
                full_idx = full_idx[flag[full_idx]]
            cur = set(self.current_mask.tolist())
            for i in full_idx.tolist():
                cur.discard(i) if i in cur else cur.add(i)
            self.current_mask = np.fromiter(cur, dtype=np.int64, count=len(cur))
            return self.full_mask_to_display(self.current_mask)

    def paint_instance_points(self, full_idx: np.ndarray, iid: int, foreground: bool) -> dict:
        """Brush-paint a committed instance: foreground adds unlabeled points to
        ``iid``; background removes points currently labeled ``iid``. Other trees
        are never stolen from."""
        with self.lock:
            full_idx = np.asarray(full_idx, dtype=np.int64)
            lab = self.full_labels[full_idx]
            if foreground:
                sel = full_idx[lab == 0]
                self.full_labels[sel] = iid
                out = {"added": int(sel.size), "removed": 0}
            else:
                sel = full_idx[lab == iid]
                self.full_labels[sel] = 0
                out = {"added": 0, "removed": int(sel.size)}
            if iid in self.instances:
                self._recompute_one(iid)
            out["instance_id"] = iid
            return out

    def toggle_instance_points(self, full_idx: np.ndarray, iid: int) -> dict:
        """Brush-toggle for a committed instance: points labeled ``iid`` become
        background (0); unlabeled points become ``iid``. Points belonging to other
        trees are left untouched (never stolen). Returns counts + affected ids."""
        with self.lock:
            full_idx = np.asarray(full_idx, dtype=np.int64)
            lab = self.full_labels[full_idx]
            to_remove = full_idx[lab == iid]
            to_add = full_idx[lab == 0]
            self.full_labels[to_remove] = 0
            self.full_labels[to_add] = iid
            if iid in self.instances:
                self._recompute_one(iid)
            return {"added": int(to_add.size), "removed": int(to_remove.size), "instance_id": iid}

    # ---- instances ---------------------------------------------------------
    def commit_current(
        self, attributes: dict, n_clicks: int | None = None, elapsed_s: float | None = None
    ) -> tuple[Instance, np.ndarray, dict]:
        """Commit the current working mask as a new instance.

        Returns (instance, display_idx, eval_record). ``n_clicks`` / ``elapsed_s``
        are the user's interaction stats for this tree (recorded for evaluation).
        """
        with self.lock:
            if self.current_mask.size == 0:
                raise ValueError("No current mask to commit")
            iid = int(attributes.get("id") or self._next_id)
            if iid in self.instances:
                raise ValueError(f"Instance id {iid} already exists")
            self._next_id = max(self._next_id, iid) + 1

            mask = self.current_mask
            ev = self.evaluate_mask(mask)  # IoU vs GT before clearing

            self.full_labels[mask] = iid
            inst = Instance(id=iid, color=_instance_color(iid))
            for key in ("species", "dbh", "health_status", "notes"):
                if attributes.get(key) is not None:
                    setattr(inst, key, attributes[key])
            self._apply_geometry(inst, mask)
            self.instances[iid] = inst
            display = self.full_mask_to_display(mask)

            record = {
                "instance_id": iid,
                "clicks": None if n_clicks is None else int(n_clicks),
                "time_s": None if elapsed_s is None else round(float(elapsed_s), 2),
                "point_count": inst.point_count,
                "matched_gt": ev["matched_gt"],
                "iou": ev["iou"],
                "gt_point_count": ev["gt_point_count"],
            }
            self.eval_records.append(record)

            self.current_mask = np.empty(0, dtype=np.int64)
            self.current_block = np.empty(0, dtype=np.int64)
            return inst, display, record

    # ---- evaluation (IoU vs ground truth) ---------------------------------
    def evaluate_mask(self, full_idx: np.ndarray) -> dict:
        """Per-instance IoU of a predicted point set vs its best-matching GT tree.

        The matched GT tree is the ground-truth instance with the largest overlap
        with the prediction (ignoring the ground/unlabeled id). Returns
        ``matched_gt`` / ``iou`` / ``gt_point_count`` (iou is None if no GT).
        """
        full_idx = np.asarray(full_idx, dtype=np.int64)
        if self.gt_instance is None or full_idx.size == 0:
            return {"matched_gt": None, "iou": None, "gt_point_count": 0}
        gt = self.gt_instance
        pred_gt = gt[full_idx]
        valid = pred_gt[pred_gt != config.GT_IGNORE_ID]
        if valid.size == 0:
            return {"matched_gt": None, "iou": 0.0, "gt_point_count": 0}
        vals, counts = np.unique(valid, return_counts=True)
        matched = int(vals[np.argmax(counts)])
        gt_pts = np.nonzero(gt == matched)[0]
        inter = np.intersect1d(full_idx, gt_pts, assume_unique=False).size
        union = full_idx.size + gt_pts.size - inter
        iou = float(inter / union) if union > 0 else 0.0
        return {"matched_gt": matched, "iou": round(iou, 4), "gt_point_count": int(gt_pts.size)}

    def evaluation(self) -> dict:
        """Per-tree evaluation records + a session summary."""
        recs = self.eval_records
        ious = [r["iou"] for r in recs if r["iou"] is not None]
        times = [r["time_s"] for r in recs if r["time_s"] is not None]
        clicks = [r["clicks"] for r in recs if r["clicks"] is not None]
        summary = {
            "n_trees": len(recs),
            "has_gt": self.gt_instance is not None,
            "session_time_s": self.session_time_s,       # manual Start/Stop total (authoritative "final time")
            "total_clicks": int(sum(clicks)) if clicks else 0,
            "mean_clicks": round(sum(clicks) / len(clicks), 2) if clicks else None,
            "total_tree_time_s": round(sum(times), 2) if times else 0.0,  # summed per-tree auto times
            "mean_time_s": round(sum(times) / len(times), 2) if times else None,
            "mean_iou": round(sum(ious) / len(ious), 4) if ious else None,
            "detection_rate_0p5": round(sum(1 for i in ious if i > 0.5) / len(ious), 4) if ious else None,
        }
        return {"records": recs, "summary": summary}

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
        # Single-user: only one scene lives in memory at a time. Free the previous
        # scene (full-res points + KD-tree + labels) BEFORE loading the new one,
        # so repeated loads don't accumulate memory and peak stays ~1 scene.
        with self._lock:
            self._sessions.clear()
        gc.collect()
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
