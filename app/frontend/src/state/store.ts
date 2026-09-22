import { create } from "zustand";
import type { AppConfig, BBox, Click, Instance, Bounds, EvalRecord, EvalSummary } from "../api/client";

export type Tool = "orbit" | "bbox" | "pos" | "neg" | "brush" | "lasso";
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
  brushMode: "fg" | "bg";        // brush/lasso paints foreground (label) or background (erase)
  brushShape: "ball" | "through"; // ball = 3D sphere at surface; through = screen disc through depth
  colorField: "height" | "intensity" | "gt_instance" | "gt_semantic"; // point-cloud coloring field

  // current (uncommitted) tree working state
  bbox: BBox | null;
  clicks: Click[];
  maskIndices: number[];
  lastInferMs: number | null;
  treeStartTime: number | null;   // ms epoch when work on the current tree began
  hideMask: boolean;              // hide the in-progress mask overlay (inspect raw points)

  // evaluation (click count / time / IoU per confirmed tree)
  evalRecords: EvalRecord[];
  evalSummary: EvalSummary | null;

  // manual session stopwatch ("final time")
  timerRunning: boolean;
  timerStartedAt: number | null;   // ms epoch of current run
  timerAccumMs: number;            // accumulated across runs

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
  setBrushMode: (m: "fg" | "bg") => void;
  setBrushShape: (s: "ball" | "through") => void;
  setColorField: (f: "height" | "intensity" | "gt_instance" | "gt_semantic") => void;
  setBBox: (b: BBox | null) => void;
  addClick: (c: Click) => void;
  clearClicks: () => void;
  setMask: (idx: number[], ms: number | null) => void;
  clearCurrent: () => void;
  toggleHideMask: () => void;
  addEvalRecord: (r: EvalRecord) => void;
  setEvalSummary: (s: EvalSummary | null) => void;
  startTimer: () => void;
  stopTimer: () => number;   // returns total elapsed seconds
  resetTimer: () => void;
  setInstances: (list: Instance[]) => void;
  addInstance: (i: Instance) => void;
  removeInstance: (iid: number) => void;
  selectInstance: (iid: number | null) => void;
  toggleVisibility: (iid: number) => void;
  setStatus: (s: string) => void;
  setDialogOpen: (v: boolean) => void;
}

export const useStore = create<AppState>((set, get) => ({
  config: null,
  sessionId: null,
  bounds: null,
  numPoints: 0,
  numPointsFull: 0,
  decimated: false,

  tool: "orbit",
  realtime: true,
  brushRadius: 18, // screen-space brush radius in pixels
  brushMode: "fg",
  brushShape: "ball",
  colorField: "height",

  bbox: null,
  clicks: [],
  maskIndices: [],
  lastInferMs: null,
  treeStartTime: null,
  hideMask: false,

  evalRecords: [],
  evalSummary: null,

  timerRunning: false,
  timerStartedAt: null,
  timerAccumMs: 0,

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
      treeStartTime: null,
      hideMask: false,
      evalRecords: [],
      evalSummary: null,
      timerRunning: false,
      timerStartedAt: null,
      timerAccumMs: 0,
    }),
  setTool: (tool) => set({ tool }),
  toggleRealtime: () => set((s) => ({ realtime: !s.realtime })),
  setBrushRadius: (brushRadius) => set({ brushRadius }),
  setBrushMode: (brushMode) => set({ brushMode }),
  setBrushShape: (brushShape) => set({ brushShape }),
  setColorField: (colorField) => set({ colorField }),
  // starting a tree = first bbox or first click; stamp the start time once
  setBBox: (bbox) => set((s) => ({ bbox, treeStartTime: s.treeStartTime ?? (bbox ? Date.now() : null) })),
  addClick: (c) => set((s) => ({ clicks: [...s.clicks, c], treeStartTime: s.treeStartTime ?? Date.now() })),
  clearClicks: () => set({ clicks: [] }),
  setMask: (maskIndices, lastInferMs) => set({ maskIndices, lastInferMs }),
  clearCurrent: () => set({ bbox: null, clicks: [], maskIndices: [], lastInferMs: null, treeStartTime: null }),
  toggleHideMask: () => set((s) => ({ hideMask: !s.hideMask })),
  addEvalRecord: (r) => set((s) => ({ evalRecords: [...s.evalRecords, r] })),
  setEvalSummary: (evalSummary) => set({ evalSummary }),
  startTimer: () => set((s) => (s.timerRunning ? {} : { timerRunning: true, timerStartedAt: Date.now() })),
  stopTimer: () => {
    const s = get();
    if (!s.timerRunning) return s.timerAccumMs / 1000;
    const accum = s.timerAccumMs + (Date.now() - (s.timerStartedAt ?? Date.now()));
    set({ timerRunning: false, timerStartedAt: null, timerAccumMs: accum });
    return accum / 1000;
  },
  resetTimer: () => set({ timerRunning: false, timerStartedAt: null, timerAccumMs: 0 }),
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
