import { useEffect } from "react";
import Viewer from "./viewer/Viewer";
import Toolbar from "./panels/Toolbar";
import TopBar from "./panels/TopBar";
import InstancePanel from "./panels/InstancePanel";
import AttributeDialog from "./panels/AttributeDialog";
import { useStore, type Tool } from "./state/store";
import { annotator } from "./viewer/annotator";
import { api } from "./api/client";
import { useT } from "./i18n";

const KEYS: Record<string, Tool> = {
  b: "bbox", p: "pos", n: "neg", l: "lasso", k: "brush", t: "orbit",
};

export default function App() {
  const t = useT();
  const status = useStore((s) => s.status);
  const lastMs = useStore((s) => s.lastInferMs);
  const instances = useStore((s) => s.instances);
  const numPoints = useStore((s) => s.numPoints);

  // load dropdown config once
  useEffect(() => {
    api.getConfig().then(useStore.getState().setConfig).catch(() => {});
  }, []);

  // keyboard shortcuts
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.target instanceof HTMLInputElement || e.target instanceof HTMLSelectElement) return;
      const st = useStore.getState();
      const k = e.key.toLowerCase();
      if (k in KEYS) { st.setTool(KEYS[k]); }
      else if (e.key === "Enter" && st.maskIndices.length > 0) st.setDialogOpen(true);
      else if (e.key === "Escape") annotator.clearWorking();
      else if (k === "t") annotator.scene?.topView();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, []);

  const labeledPts = instances.reduce((s, i) => s + i.point_count, 0);
  const pct = numPoints ? ((labeledPts / numPoints) * 100).toFixed(1) : "0";

  return (
    <div className="app">
      <TopBar />
      <div className="app-body">
        <Toolbar />
        <Viewer />
        <InstancePanel />
      </div>
      <div className="bottombar">
        <span>{status}</span>
        <span>
          {t("bottom.progress", { k: instances.length, pct })}
          {lastMs != null && t("bottom.infer", { ms: lastMs })}
        </span>
      </div>
      <AttributeDialog />
    </div>
  );
}
