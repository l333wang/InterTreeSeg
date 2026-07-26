import { create } from "zustand";
import type { AppConfig, BBox, Click, Instance, Bounds } from "../api/client";

export type Tool = "orbit" | "bbox" | "pos" | "neg" | "lasso" | "brush";
export type Lang = "en" | "zh";

interface AppState {
  config: AppConfig | null;

  // session / scene
  sessionId: string | null;
  bounds: Bounds | null;
  numPoints: number;
  numPointsFull: number;
  decimated: boolean;

  // interaction
  tool: Tool;
  realtime: boolean;
  brushRadius: number;

  // current (uncommitted) tree working state
  bbox: BBox | null;
  clicks: Click[];
  maskIndices: number[];
  lastInferMs: number | null;

  // committed instances
  instances: Instance[];
  selectedInstance: number | null;
  hiddenInstances: Set<number>;

  // ui
  status: string;
  dialogOpen: boolean;
  lang: Lang;
  loading: boolean;
  loadingMsg: string;

  // actions
  setConfig: (c: AppConfig) => void;
  setLang: (l: Lang) => void;
  setLoading: (msg: string | null) => void;
  setSession: (s: {
    sessionId: string; bounds: Bounds; numPoints: number;
    numPointsFull: number; decimated: boolean;
  }) => void;
  setTool: (t: Tool) => void;
  toggleRealtime: () => void;
  setBrushRadius: (r: number) => void;
  setBBox: (b: BBox | null) => void;
  addClick: (c: Click) => void;
  clearClicks: () => void;
  setMask: (idx: number[], ms: number | null) => void;
  clearCurrent: () => void;
  setInstances: (list: Instance[]) => void;
  addInstance: (i: Instance) => void;
  removeInstance: (iid: number) => void;
  selectInstance: (iid: number | null) => void;
  toggleVisibility: (iid: number) => void;
  setStatus: (s: string) => void;
  setDialogOpen: (v: boolean) => void;
}

export const useStore = create<AppState>((set) => ({
  config: null,
  sessionId: null,
  bounds: null,
  numPoints: 0,
  numPointsFull: 0,
  decimated: false,

  tool: "orbit",
  realtime: true,
  brushRadius: 18, // screen-space brush radius in pixels

  bbox: null,
  clicks: [],
  maskIndices: [],
  lastInferMs: null,

  instances: [],
  selectedInstance: null,
  hiddenInstances: new Set(),

  status: "Open a point cloud file to start.",
  dialogOpen: false,
  lang: "en",
  loading: false,
  loadingMsg: "",

  setConfig: (config) => set({ config }),
  setLang: (lang) => set({ lang }),
  setLoading: (msg) => set(msg === null ? { loading: false } : { loading: true, loadingMsg: msg }),
  setSession: (s) =>
    set({
      sessionId: s.sessionId,
      bounds: s.bounds,
      numPoints: s.numPoints,
      numPointsFull: s.numPointsFull,
      decimated: s.decimated,
      instances: [],
      selectedInstance: null,
      bbox: null,
      clicks: [],
      maskIndices: [],
    }),
  setTool: (tool) => set({ tool }),
  toggleRealtime: () => set((s) => ({ realtime: !s.realtime })),
  setBrushRadius: (brushRadius) => set({ brushRadius }),
  setBBox: (bbox) => set({ bbox }),
  addClick: (c) => set((s) => ({ clicks: [...s.clicks, c] })),
  clearClicks: () => set({ clicks: [] }),
  setMask: (maskIndices, lastInferMs) => set({ maskIndices, lastInferMs }),
  clearCurrent: () => set({ bbox: null, clicks: [], maskIndices: [], lastInferMs: null }),
  setInstances: (instances) => set({ instances }),
  addInstance: (i) => set((s) => ({ instances: [...s.instances, i] })),
  removeInstance: (iid) =>
    set((s) => ({
      instances: s.instances.filter((x) => x.id !== iid),
      selectedInstance: s.selectedInstance === iid ? null : s.selectedInstance,
    })),
  selectInstance: (selectedInstance) => set({ selectedInstance }),
  toggleVisibility: (iid) =>
    set((s) => {
      const h = new Set(s.hiddenInstances);
      h.has(iid) ? h.delete(iid) : h.add(iid);
      return { hiddenInstances: h };
    }),
  setStatus: (status) => set({ status }),
  setDialogOpen: (dialogOpen) => set({ dialogOpen }),
}));
