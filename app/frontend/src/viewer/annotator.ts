// Imperative controller shared by the Viewer and the side panels. Owns the
// three.js scene and a local copy of per-point instance labels, and mediates
// between the zustand store, the REST API, and the scene renderer.
import * as THREE from "three";
import { PointCloudScene } from "./scene";
import { api, type Bounds, type Instance, type ScreenSelection, type BrushBody, type DiscSel, type PolySel } from "../api/client";
import { useStore } from "../state/store";
import { t } from "../i18n";

class Annotator {
  scene: PointCloudScene | null = null;
  private labels: Int32Array | null = null;

  init(container: HTMLElement) {
    if (!this.scene) this.scene = new PointCloudScene(container);
  }

  get ready() {
    return !!this.scene && !!this.labels;
  }

  async loadScene(sid: string, bounds: Bounds) {
    const setLoading = useStore.getState().setLoading;
    setLoading(t("load.downloading"));
    const [pts, labels] = await Promise.all([api.points(sid), api.labels(sid)]);
    this.labels = labels;
    setLoading(t("load.rendering"));
    // yield a frame so the "rendering" overlay paints before the heavy build loop
    await new Promise((r) => setTimeout(r, 0));
    this.scene!.loadPoints(pts, bounds);
    this.recolor();
    setLoading(null);
  }

  private colorMap(): Map<number, [number, number, number]> {
    const m = new Map<number, [number, number, number]>();
    for (const i of useStore.getState().instances) m.set(i.id, i.color);
    return m;
  }

  /** Recompute base point colors from a chosen data field (height/intensity/GT). */
  async applyColorField() {
    const st = useStore.getState();
    if (!st.sessionId || !this.scene) return;
    const field = st.colorField;
    const values = await api.field(st.sessionId, field);
    const categorical = field === "gt_instance" || field === "gt_semantic";
    this.scene.setBaseColorField(values, categorical);
    this.recolor();
  }

  recolor() {
    if (!this.scene || !this.labels) return;
    const st = useStore.getState();
    // hideMask hides the current selection ENTIRELY (label highlight + its raw
    // points), so the tree being annotated disappears — lets you inspect what's
    // behind / around it. Toggle off to bring it back.
    const maskSet = st.maskIndices.length ? new Set(st.maskIndices) : null;
    // Focus mode: while annotating one tree, dim the unlabeled background so the
    // highlighted tree stands out (not while hiding).
    const focusing = !st.hideMask && (st.maskIndices.length > 0 || st.clicks.length > 0 || st.bbox != null);
    const bgDim = focusing ? 0.5 : 0.9;
    this.scene.recolor(this.labels, maskSet, st.hiddenInstances, this.colorMap(), bgDim, st.hideMask);
  }

  /** What a brush/lasso stroke edits: the working mask, or a selected instance. */
  private brushTarget(): "mask" | number | null {
    const st = useStore.getState();
    if (st.maskIndices.length > 0 || st.clicks.length > 0) return "mask";
    if (st.selectedInstance != null) return st.selectedInstance;
    return null;
  }

  /** Apply a paint geometry (ball or screen selection) with the current fg/bg mode. */
  private async applyBrush(
    target: "mask" | number,
    geom: { center: [number, number, number]; radius: number } | { selection: DiscSel | PolySel }
  ) {
    const st = useStore.getState();
    if (!st.sessionId) return;
    const res = await api.brush(st.sessionId, { target, mode: st.brushMode, ...geom } as BrushBody);
    if (target === "mask") {
      useStore.getState().setMask(res.mask_indices ?? [], st.lastInferMs);
      this.recolor();
    } else {
      await this.refreshLabels();
      await this.refreshInstances();
    }
  }

  /** Surface ball brush: 3D sphere around the picked point (pixel radius → world). */
  async brushBall(center: THREE.Vector3, pixelRadius: number) {
    const target = this.brushTarget();
    if (target == null || !this.scene) { useStore.getState().setStatus(t("st.manualNeed")); return; }
    const radius = this.scene.pixelToWorldRadius(center, pixelRadius);
    await this.applyBrush(target, { center: [center.x, center.y, center.z], radius });
  }

  /** Through-depth brush: the on-screen circle projected through ALL depths. */
  async brushThrough(cx: number, cy: number, pixelRadius: number) {
    const target = this.brushTarget();
    if (target == null || !this.scene) { useStore.getState().setStatus(t("st.manualNeed")); return; }
    await this.applyBrush(target, { selection: { kind: "disc", ...this.scene.getViewProj(), cx, cy, r: pixelRadius } });
  }

  /** Lasso: freehand polygon selection through ALL depths. */
  async lassoSelect(polygon: [number, number][]) {
    const target = this.brushTarget();
    if (target == null || !this.scene) { useStore.getState().setStatus(t("st.manualNeed")); return; }
    await this.applyBrush(target, { selection: { kind: "polygon", ...this.scene.getViewProj(), polygon } });
  }

  // ---- inference ---------------------------------------------------------
  async runInfer() {
    const st = useStore.getState();
    if (!st.sessionId || st.clicks.length === 0) return;
    try {
      const res = await api.infer(st.sessionId, st.bbox, st.clicks);
      useStore.getState().setMask(res.mask_indices, res.elapsed_ms);
      this.recolor();
      useStore.getState().setStatus(
        t("st.infer", { n: res.count, ms: res.elapsed_ms, c: st.clicks.length })
      );
    } catch (e) {
      useStore.getState().setStatus(t("st.inferFail", { msg: (e as Error).message }));
    }
  }

  // ---- commit ------------------------------------------------------------
  previewGeometry(indices: number[]) {
    const pos = this.scene?.positionsRef();
    if (!pos || indices.length === 0) return { height: 0, crown_width: 0, stem_x: 0, stem_y: 0 };
    let xmin = Infinity, xmax = -Infinity, ymin = Infinity, ymax = -Infinity,
        zmin = Infinity, zmax = -Infinity;
    for (const i of indices) {
      const x = pos[i * 3], y = pos[i * 3 + 1], z = pos[i * 3 + 2];
      xmin = Math.min(xmin, x); xmax = Math.max(xmax, x);
      ymin = Math.min(ymin, y); ymax = Math.max(ymax, y);
      zmin = Math.min(zmin, z); zmax = Math.max(zmax, z);
    }
    // stem = centroid of lowest 10%
    const zc = zmin + (zmax - zmin) * 0.1;
    let sx = 0, sy = 0, k = 0;
    for (const i of indices) {
      if (pos[i * 3 + 2] <= zc) { sx += pos[i * 3]; sy += pos[i * 3 + 1]; k++; }
    }
    if (k === 0) k = 1;
    return {
      height: +(zmax - zmin).toFixed(3),
      crown_width: +Math.max(xmax - xmin, ymax - ymin).toFixed(3),
      stem_x: +(sx / k).toFixed(3),
      stem_y: +(sy / k).toFixed(3),
    };
  }

  async commitCurrent(attributes: Partial<Instance>) {
    const st = useStore.getState();
    if (!st.sessionId || st.maskIndices.length === 0) return;
    const n_clicks = st.clicks.length;
    const elapsed_s = st.treeStartTime != null ? (Date.now() - st.treeStartTime) / 1000 : undefined;
    const res = await api.commit(st.sessionId, attributes, { n_clicks, elapsed_s });
    const inst = res.instance;
    for (const i of res.display_indices) this.labels![i] = inst.id;
    const store = useStore.getState();
    store.addInstance(inst);
    store.addEvalRecord(res.eval);
    store.clearCurrent();
    this.scene!.clearClickMarkers();
    this.scene!.setBBoxHelper(null, 0, 0);
    store.setTool("bbox");   // back to top-view bbox, ready to frame the next tree
    this.recolor();
    const ev = res.eval;
    store.setStatus(
      t("st.committedEval", {
        id: inst.id,
        n: inst.point_count,
        iou: ev.iou == null ? "—" : ev.iou.toFixed(3),
        c: ev.clicks ?? n_clicks,
        s: (ev.time_s ?? elapsed_s ?? 0).toFixed(1),
      })
    );
    // refresh the session evaluation summary (for the top-bar chip)
    api.evaluation(st.sessionId).then((e) => useStore.getState().setEvalSummary(e.summary)).catch(() => {});
  }

  // ---- manual editing (full-resolution, via backend screen projection) ---
  private selection(kind: "polygon" | "disc", extra: Partial<ScreenSelection>): ScreenSelection {
    return { ...this.scene!.getViewProj(), kind, ...extra };
  }

  /** Lasso/brush edit of the CURRENT working mask (add or remove). */
  async editMaskScreen(sel: ScreenSelection, add: boolean) {
    const st = useStore.getState();
    if (!st.sessionId) return;
    const res = await api.editMask(st.sessionId, sel, add);
    useStore.getState().setMask(res.mask_indices, st.lastInferMs);
    this.recolor();
  }

  /** Lasso/brush edit of a committed instance (target 0 = unlabeled). */
  async assignScreen(sel: ScreenSelection, target: number) {
    const st = useStore.getState();
    if (!st.sessionId) return;
    await api.assign(st.sessionId, sel, target);
    await this.refreshLabels();
    await this.refreshInstances();
  }

  makeSelection = (kind: "polygon" | "disc", extra: Partial<ScreenSelection>) =>
    this.selection(kind, extra);

  /** Undo / redo the last editing operation (server-side snapshot stack). */
  async undo() { await this._undoRedo("undo"); }
  async redo() { await this._undoRedo("redo"); }
  private async _undoRedo(which: "undo" | "redo") {
    const st = useStore.getState();
    if (!st.sessionId) return;
    const res = which === "undo" ? await api.undo(st.sessionId) : await api.redo(st.sessionId);
    if (!res.ok) { st.setStatus(t(which === "undo" ? "st.nothingUndo" : "st.nothingRedo")); return; }
    // reset transient interaction; restore the working mask + labels from the server
    st.setBBox(null); st.clearClicks();
    this.scene?.clearClickMarkers();
    this.scene?.setBBoxHelper(null, 0, 0);
    useStore.getState().setMask(res.mask_indices, null);
    await this.refreshLabels();
    await this.refreshInstances();
    useStore.getState().setStatus(t(which === "undo" ? "st.undone" : "st.redone"));
  }

  /** Discard the current working tree (clicks, bbox, mask) both client and server. */
  async clearWorking() {
    const st = useStore.getState();
    this.scene?.clearClickMarkers();
    this.scene?.setBBoxHelper(null, 0, 0);
    st.clearCurrent();
    st.setTool("bbox");   // back to top-view bbox default state
    this.recolor();
    if (st.sessionId) await api.clearMask(st.sessionId);
  }

  async refreshLabels() {
    const st = useStore.getState();
    if (!st.sessionId) return;
    this.labels = await api.labels(st.sessionId);
    this.recolor();
  }

  async refreshInstances() {
    const st = useStore.getState();
    if (!st.sessionId) return;
    const { instances } = await api.instances(st.sessionId);
    useStore.getState().setInstances(instances);
    this.recolor();
  }

  /** Change a committed tree's ID (relabels its points). */
  async renameInstance(oldId: number, newId: number) {
    const st = useStore.getState();
    if (!st.sessionId || newId === oldId) return;
    try {
      await api.renameInstance(st.sessionId, oldId, newId);
      await this.refreshLabels();
      await this.refreshInstances();
      useStore.getState().selectInstance(newId);
      useStore.getState().setStatus(t("st.renamed", { from: oldId, to: newId }));
    } catch (e) {
      useStore.getState().setStatus(t("st.renameFail", { msg: (e as Error).message }));
      await this.refreshInstances();  // revert the edited field to its real value
    }
  }

  async deleteInstance(iid: number) {
    const st = useStore.getState();
    if (!st.sessionId) return;
    await api.deleteInstance(st.sessionId, iid);
    useStore.getState().removeInstance(iid);
    await this.refreshLabels();
  }
}

export const annotator = new Annotator();
export { THREE };
