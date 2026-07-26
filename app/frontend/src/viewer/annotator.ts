// Imperative controller shared by the Viewer and the side panels. Owns the
// three.js scene and a local copy of per-point instance labels, and mediates
// between the zustand store, the REST API, and the scene renderer.
import * as THREE from "three";
import { PointCloudScene } from "./scene";
import { api, type Bounds, type Instance, type ScreenSelection } from "../api/client";
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

  recolor() {
    if (!this.scene || !this.labels) return;
    const st = useStore.getState();
    const maskSet = st.maskIndices.length ? new Set(st.maskIndices) : null;
    // Focus mode: while annotating one tree (bbox / clicks / mask active), dim the
    // unlabeled height-colored background so the highlighted tree stands out.
    const focusing = st.maskIndices.length > 0 || st.clicks.length > 0 || st.bbox != null;
    const bgDim = focusing ? 0.5 : 0.9;
    this.scene.recolor(this.labels, maskSet, st.hiddenInstances, this.colorMap(), bgDim);
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
    const res = await api.commit(st.sessionId, attributes);   // server uses its full-res mask
    const inst = res.instance;
    for (const i of res.display_indices) this.labels![i] = inst.id;
    useStore.getState().addInstance(inst);
    useStore.getState().clearCurrent();
    this.scene!.clearClickMarkers();
    this.scene!.setBBoxHelper(null, 0, 0);
    this.recolor();
    useStore.getState().setStatus(t("st.committed", { id: inst.id, n: inst.point_count }));
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

  /** Discard the current working tree (clicks, bbox, mask) both client and server. */
  async clearWorking() {
    const st = useStore.getState();
    this.scene?.clearClickMarkers();
    this.scene?.setBBoxHelper(null, 0, 0);
    st.clearCurrent();
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
