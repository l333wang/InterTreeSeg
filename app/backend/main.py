"""FastAPI app: REST + WebSocket for the point-cloud tree annotation tool.

Run (from app/backend):
    conda run -n Point python -m uvicorn main:app --reload --port 8000
"""
from __future__ import annotations

import json
import os
import shutil
import tempfile

import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response, StreamingResponse

from . import config
from .inference import Click, get_inference_service
from .io.exporter import export_attributes_csv, export_evaluation_csv, export_points_txt
from .schemas import (
    AssignScreenRequest,
    CommitRequest,
    EditMaskRequest,
    InferRequest,
    InferResponse,
    LoadRequest,
    LoadResponse,
    ProgressState,
)
from .session import manager
from .spatial import project

app = FastAPI(title="InterTreeSeg API")

# Dev CORS: the Vite frontend runs on a different port.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# --------------------------------------------------------------------------- #
# Session lifecycle + scene data
# --------------------------------------------------------------------------- #
def _load_response(s) -> LoadResponse:
    return LoadResponse(
        session_id=s.id,
        num_points=s.n,
        num_points_full=int(s.full_points.shape[0]),
        decimated=s.decimated,
        bounds=s.bounds(),
    )


@app.post("/api/sessions/load", response_model=LoadResponse)
def load_scene(req: LoadRequest):
    """Load a point cloud from a path on the server (for scripting / batch)."""
    try:
        s = manager.create(req.file_path)
    except FileNotFoundError:
        raise HTTPException(404, f"File not found: {req.file_path}")
    except Exception as e:
        raise HTTPException(400, str(e))
    return _load_response(s)


@app.post("/api/sessions/upload", response_model=LoadResponse)
async def upload_scene(file: UploadFile = File(...)):
    """Load a point cloud uploaded from the user's machine (browser file picker)."""
    suffix = os.path.splitext(file.filename or "")[1] or ".txt"
    tmpdir = tempfile.mkdtemp(prefix="anno_upload_")
    dest = os.path.join(tmpdir, file.filename or f"upload{suffix}")
    try:
        with open(dest, "wb") as out:
            shutil.copyfileobj(file.file, out)
    finally:
        await file.close()
    try:
        s = manager.create(dest)
    except Exception as e:
        raise HTTPException(400, str(e))
    return _load_response(s)


@app.get("/api/sessions/{sid}/points")
def get_points(sid: str):
    """Binary stream: float32 [x, y, z, scalar] interleaved, 4 floats/point.
    scalar = z (the frontend colors by height)."""
    s = _session(sid)
    xyz = s.render_xyz().astype(np.float32)
    scalar = xyz[:, 2:3]
    interleaved = np.concatenate([xyz, scalar], axis=1).ravel()
    return Response(content=interleaved.tobytes(), media_type="application/octet-stream")


@app.get("/api/sessions/{sid}/field")
def get_field(sid: str, name: str = "height"):
    """Binary stream: float32 scalar per RENDERED point, for the given color field
    (height | intensity | gt_instance | gt_semantic)."""
    s = _session(sid)
    return Response(content=s.field_values(name).tobytes(), media_type="application/octet-stream")


@app.get("/api/sessions/{sid}/labels")
def get_labels(sid: str):
    """Binary stream: int32 instance id per RENDERED point (0 = unlabeled)."""
    s = _session(sid)
    return Response(content=s.display_labels().astype(np.int32).tobytes(),
                    media_type="application/octet-stream")


# --------------------------------------------------------------------------- #
# Inference
# --------------------------------------------------------------------------- #
@app.post("/api/sessions/{sid}/infer", response_model=InferResponse)
def infer(sid: str, req: InferRequest):
    s = _session(sid)
    return _run_infer(s, req)


def _run_infer(s, req: InferRequest) -> InferResponse:
    # Inference runs on FULL-resolution points for annotation quality.
    if req.bbox is not None:
        b = req.bbox
        cand = s.full_index.crop_bbox_xy(b.x_min, b.y_min, b.x_max, b.y_max, b.z_min, b.z_max)
    else:
        cand = np.arange(s.n_full)
    # Exclude points already committed to another tree: once a tree is confirmed
    # its points leave the working pool, so trees never overlap and the next tree
    # cannot re-include an earlier tree's points.
    cand = cand[s.full_labels[cand] == 0]
    s.set_current_block(cand)                        # manual edits restricted to this block
    clicks = [Click(c.x, c.y, c.z, c.positive) for c in req.clicks]
    if cand.size == 0:                               # whole crop already labeled
        s.set_current_mask(np.empty(0, dtype=np.int64))
        return InferResponse(mask_indices=[], count=0, elapsed_ms=0.0)
    res = get_inference_service().infer(s.full_xyz, cand, clicks)
    display = s.set_current_mask(res.mask_indices)   # store full mask, map to rendered points
    return InferResponse(
        mask_indices=display.tolist(),               # decimated indices for browser highlight
        count=int(res.mask_indices.size),            # full-resolution point count
        elapsed_ms=round(res.elapsed_ms, 2),
    )


# --------------------------------------------------------------------------- #
# Instances
# --------------------------------------------------------------------------- #
@app.post("/api/sessions/{sid}/commit")
def commit(sid: str, req: CommitRequest):
    """Commit the current full-resolution working mask as a new instance."""
    s = _session(sid)
    s.push_undo()
    try:
        inst, display, ev = s.commit_current(req.attributes.model_dump(), req.n_clicks, req.elapsed_s)
    except ValueError as e:
        raise HTTPException(409, str(e))
    return {"instance": inst.to_dict(), "display_indices": display.tolist(), "eval": ev}


@app.post("/api/sessions/{sid}/mask/edit")
def edit_mask(sid: str, req: EditMaskRequest):
    """Manual lasso/brush edit of the current working mask, at full resolution."""
    s = _session(sid)
    s.push_undo()
    full_idx = _select_full(s, req.selection)
    display = s.edit_current_mask(full_idx, req.add)
    return {"mask_indices": display.tolist(), "count": int(s.current_mask.size)}


@app.post("/api/sessions/{sid}/mask/clear")
def clear_mask(sid: str):
    s = _session(sid)
    s.push_undo()
    s.clear_current_mask()
    return {"ok": True}


@app.post("/api/sessions/{sid}/undo")
def undo(sid: str):
    s = _session(sid)
    ok = s.undo()
    return {"ok": ok, "mask_indices": s.current_display_mask().tolist(),
            "can_undo": bool(s._undo), "can_redo": bool(s._redo)}


@app.post("/api/sessions/{sid}/redo")
def redo(sid: str):
    s = _session(sid)
    ok = s.redo()
    return {"ok": ok, "mask_indices": s.current_display_mask().tolist(),
            "can_undo": bool(s._undo), "can_redo": bool(s._redo)}


@app.post("/api/sessions/{sid}/brush")
def brush(sid: str, body: dict):
    """3D ball-query brush that paints labels.

    Geometry (one of):
      - ball    : {center: [x,y,z], radius: float}  — 3D sphere at the picked surface.
      - through : {selection: {kind:"disc", view_proj, vw, vh, cx, cy, r}} — the
                  on-screen brush circle projected through the whole depth (reaches
                  interior points the surface ball can't).
      - lasso   : {selection: {kind:"polygon", view_proj, vw, vh, polygon}} — points
                  whose screen projection falls inside a freehand polygon (all depths).
    plus: target: "mask" | <instance id>, mode: "fg" | "bg" | "toggle".
    """
    s = _session(sid)
    s.push_undo()
    sel = body.get("selection")
    if sel is not None:
        if sel.get("kind") == "polygon":
            idx = project.select_polygon(s.full_xyz, sel["view_proj"], sel["vw"], sel["vh"], sel["polygon"])
        else:
            idx = project.select_disc(s.full_xyz, sel["view_proj"], sel["vw"], sel["vh"], sel["cx"], sel["cy"], sel["r"])
    else:
        idx = s.full_index.ball(body["center"], float(body.get("radius", 0.0)))
    target = body.get("target", "mask")
    mode = body.get("mode", "fg")
    if target == "mask":
        if mode == "toggle":
            display = s.toggle_current_mask(idx)
        else:
            display = s.edit_current_mask(idx, mode == "fg")
        return {"mask_indices": display.tolist(), "count": int(s.current_mask.size)}
    iid = int(target)
    if mode == "toggle":
        return s.toggle_instance_points(idx, iid)
    return s.paint_instance_points(idx, iid, mode == "fg")


@app.get("/api/sessions/{sid}/instances")
def list_instances(sid: str):
    s = _session(sid)
    return {"instances": [i.to_dict() for i in sorted(s.instances.values(), key=lambda x: x.id)]}


@app.patch("/api/sessions/{sid}/instances/{iid}")
def update_instance(sid: str, iid: int, attributes: dict):
    s = _session(sid)
    if iid not in s.instances:
        raise HTTPException(404, f"No instance {iid}")
    s.push_undo()
    return {"instance": s.update_instance(iid, attributes).to_dict()}


@app.post("/api/sessions/{sid}/instances/{iid}/rename")
def rename_instance(sid: str, iid: int, body: dict):
    """Change a committed tree's ID (relabels its points)."""
    s = _session(sid)
    try:
        new_id = int(body.get("new_id"))
    except (TypeError, ValueError):
        raise HTTPException(400, "new_id must be an integer")
    s.push_undo()
    try:
        inst = s.rename_instance(iid, new_id)
    except KeyError:
        raise HTTPException(404, f"No instance {iid}")
    except ValueError as e:
        raise HTTPException(409, str(e))
    return {"instance": inst.to_dict()}


@app.delete("/api/sessions/{sid}/instances/{iid}")
def delete_instance(sid: str, iid: int):
    s = _session(sid)
    s.push_undo()
    s.delete_instance(iid)
    return {"ok": True}


@app.post("/api/sessions/{sid}/assign")
def assign_points(sid: str, req: AssignScreenRequest):
    """Manual edit of a committed instance: reassign full-res points inside the
    screen selection to `target` (0 = unlabeled)."""
    s = _session(sid)
    s.push_undo()
    full_idx = _select_full(s, req.selection)
    affected = s.assign_full(full_idx, req.target)
    return {"ok": True, "target": req.target, "affected": affected}


def _select_full(s, sel) -> np.ndarray:
    """Full-res point indices inside a screen-space lasso/brush selection."""
    if sel.kind == "polygon":
        return project.select_polygon(s.full_xyz, sel.view_proj, sel.vw, sel.vh, sel.polygon or [])
    return project.select_disc(s.full_xyz, sel.view_proj, sel.vw, sel.vh, sel.cx or 0, sel.cy or 0, sel.r or 0)


@app.post("/api/sessions/{sid}/compute_geometry")
def compute_geometry_all(sid: str):
    s = _session(sid)
    updated = s.recompute_geometry()
    return {"instances": [i.to_dict() for i in updated]}


# --------------------------------------------------------------------------- #
# Config helpers (dropdown options for the UI)
# --------------------------------------------------------------------------- #
@app.get("/api/config")
def get_config():
    return {"species": config.DEFAULT_SPECIES, "health_status": config.DEFAULT_HEALTH_STATUS}


# --------------------------------------------------------------------------- #
# Export / progress
# --------------------------------------------------------------------------- #
@app.get("/api/sessions/{sid}/export/points")
def export_points(sid: str):
    s = _session(sid)
    data = export_points_txt(s)
    return StreamingResponse(
        iter([data]),
        media_type="text/plain",
        headers={"Content-Disposition": "attachment; filename=labels.txt"},
    )


@app.get("/api/sessions/{sid}/export/attributes")
def export_attributes(sid: str):
    s = _session(sid)
    data = export_attributes_csv(s)
    return StreamingResponse(
        iter([data]),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=trees.csv"},
    )


@app.post("/api/sessions/{sid}/session_time")
def set_session_time(sid: str, body: dict):
    """Record the manual Start/Stop timer total (seconds) as the session's final time."""
    s = _session(sid)
    s.session_time_s = round(float(body.get("seconds", 0.0)), 2)
    return {"ok": True, "session_time_s": s.session_time_s}


@app.get("/api/sessions/{sid}/evaluation")
def get_evaluation(sid: str):
    """Per-tree interaction/IoU records + session summary."""
    return _session(sid).evaluation()


@app.get("/api/sessions/{sid}/export/evaluation")
def export_evaluation(sid: str):
    s = _session(sid)
    data = export_evaluation_csv(s)
    return StreamingResponse(
        iter([data]),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=evaluation.csv"},
    )


@app.get("/api/sessions/{sid}/progress")
def get_progress(sid: str):
    return _session(sid).progress_state()


@app.post("/api/sessions/{sid}/progress")
def put_progress(sid: str, state: ProgressState):
    s = _session(sid)
    try:
        s.load_progress(state.model_dump())
    except ValueError as e:
        raise HTTPException(400, str(e))
    return {"ok": True}


# --------------------------------------------------------------------------- #
# Realtime inference over WebSocket
# --------------------------------------------------------------------------- #
@app.websocket("/api/sessions/{sid}/ws")
async def ws_infer(websocket: WebSocket, sid: str):
    await websocket.accept()
    try:
        s = manager.get(sid)
    except KeyError:
        await websocket.close(code=4004)
        return
    try:
        while True:
            raw = await websocket.receive_text()
            req = InferRequest(**json.loads(raw))
            resp = _run_infer(s, req)
            await websocket.send_text(resp.model_dump_json())
    except WebSocketDisconnect:
        return


@app.on_event("startup")
def _warmup_model():
    """Load the real model at boot so the first user click isn't slow."""
    if config.INFERENCE_BACKEND.lower() != "mock":
        try:
            get_inference_service()
            print(f"[startup] inference backend '{config.INFERENCE_BACKEND}' ready")
        except Exception as e:  # don't crash the server if model load fails
            print(f"[startup] WARNING: failed to init inference backend: {e}")


@app.get("/api/health")
def health():
    return {"status": "ok", "inference_backend": config.INFERENCE_BACKEND}


def _session(sid: str):
    try:
        return manager.get(sid)
    except KeyError:
        raise HTTPException(404, f"No session {sid}")
