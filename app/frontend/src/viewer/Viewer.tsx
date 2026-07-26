import { useEffect, useRef, useState } from "react";
import { annotator } from "./annotator";
import { useStore } from "../state/store";
import type { BBox, ScreenSelection } from "../api/client";
import { t } from "../i18n";

// Refined thin red crosshair (center gap + dot); hotspot at 12,12.
const CROSSHAIR =
  "url(\"data:image/svg+xml,%3Csvg%20xmlns='http://www.w3.org/2000/svg'%20width='24'%20height='24'%3E" +
  "%3Cg%20stroke='%23ff3b3b'%20stroke-width='1'%3E" +
  "%3Cline%20x1='12'%20y1='3'%20x2='12'%20y2='10'/%3E%3Cline%20x1='12'%20y1='14'%20x2='12'%20y2='21'/%3E" +
  "%3Cline%20x1='3'%20y1='12'%20x2='10'%20y2='12'/%3E%3Cline%20x1='14'%20y1='12'%20x2='21'%20y2='12'/%3E%3C/g%3E" +
  "%3Ccircle%20cx='12'%20cy='12'%20r='0.8'%20fill='%23ff3b3b'/%3E%3C/svg%3E\") 12 12, crosshair";

export default function Viewer() {
  const containerRef = useRef<HTMLDivElement>(null);
  const sessionId = useStore((s) => s.sessionId);
  const bounds = useStore((s) => s.bounds);
  const tool = useStore((s) => s.tool);
  const loading = useStore((s) => s.loading);
  const loadingMsg = useStore((s) => s.loadingMsg);
  const brushRadius = useStore((s) => s.brushRadius);
  const [brushCursor, setBrushCursor] = useState<{ x: number; y: number } | null>(null);

  const drag = useRef<{ x: number; y: number; moved: boolean; active: boolean }>({
    x: 0, y: 0, moved: false, active: false,
  });
  const lassoRef = useRef<[number, number][]>([]);
  const [selRect, setSelRect] = useState<{ x: number; y: number; w: number; h: number } | null>(null);
  const [lassoPts, setLassoPts] = useState<[number, number][]>([]);

  // create scene once
  useEffect(() => {
    if (containerRef.current) annotator.init(containerRef.current);
  }, []);

  // load scene when session changes
  useEffect(() => {
    if (sessionId && bounds && annotator.scene) annotator.loadScene(sessionId, bounds);
  }, [sessionId, bounds]);

  // recolor when mask / instances / visibility change
  useEffect(() => {
    let prev = pick();
    const unsub = useStore.subscribe(() => {
      const cur = pick();
      if (cur !== prev) { prev = cur; annotator.recolor(); }
    });
    return unsub;
    function pick() {
      const s = useStore.getState();
      return s.maskIndices.length + ":" + s.instances.map((i) => i.id).join(",") +
        ":" + [...s.hiddenInstances].join(",") +
        ":" + (s.bbox ? 1 : 0) + ":" + s.clicks.length;   // focus-dim triggers
    }
  }, []);

  // view + rotation behaviour per tool
  useEffect(() => {
    const sc = annotator.scene;
    if (!sc) return;
    // Only the top-view box tool changes the camera (ortho top-down for framing).
    // orbit/pan and the others operate FROM THE CURRENT viewpoint — no reset.
    // Explicit resets are the "Reset view" / "Top view" buttons.
    if (tool === "bbox") sc.topView();
    sc.setRotateEnabled(tool === "orbit" || tool === "pos" || tool === "neg");
    // Guide the user: manual tools need a target (working tree or selected instance).
    if (tool === "lasso" || tool === "brush") {
      const st = useStore.getState();
      if (st.maskIndices.length === 0 && st.clicks.length === 0 && st.selectedInstance == null) {
        st.setStatus(t("st.manualNeed"));
      }
    }
    if (tool !== "brush") setBrushCursor(null);
  }, [tool]);

  const rel = (e: React.PointerEvent) => {
    const r = containerRef.current!.getBoundingClientRect();
    return { x: e.clientX - r.left, y: e.clientY - r.top };
  };

  const onDown = (e: React.PointerEvent) => {
    const p = rel(e);
    drag.current = { x: p.x, y: p.y, moved: false, active: true };
    if (tool === "lasso") { lassoRef.current = [[p.x, p.y]]; setLassoPts([[p.x, p.y]]); }
  };

  const onMove = (e: React.PointerEvent) => {
    const p = rel(e);
    if (tool === "brush") setBrushCursor(p);
    if (!drag.current.active) return;
    if (Math.abs(p.x - drag.current.x) + Math.abs(p.y - drag.current.y) > 3) drag.current.moved = true;
    if (tool === "bbox") {
      setSelRect({
        x: Math.min(p.x, drag.current.x), y: Math.min(p.y, drag.current.y),
        w: Math.abs(p.x - drag.current.x), h: Math.abs(p.y - drag.current.y),
      });
    } else if (tool === "lasso") {
      lassoRef.current.push([p.x, p.y]);
      setLassoPts([...lassoRef.current]);
    }
  };

  const onUp = (e: React.PointerEvent) => {
    const p = rel(e);
    const d = drag.current;
    drag.current.active = false;
    const sc = annotator.scene;
    if (!sc) return;
    const st = useStore.getState();

    if (tool === "bbox" && d.moved) {
      // Exact rectangle (ortho top-down): the drawn box IS the world XY block,
      // cut vertically over the full height — CloudCompare-style.
      const bbox: BBox = sc.bboxFromScreenCorners(d.x, d.y, p.x, p.y);
      st.setBBox(bbox);
      sc.setBBoxHelper(bbox, st.bounds!.min[2], st.bounds!.max[2]);
      st.setTool("pos");
      st.setStatus(t("st.box"));
      setSelRect(null);
    } else if (tool === "lasso") {
      const poly = lassoRef.current;
      setLassoPts([]);
      lassoRef.current = [];
      if (poly.length >= 3) applyManual(annotator.makeSelection("polygon", { polygon: poly }), !e.altKey);
    } else if (tool === "brush") {
      applyManual(annotator.makeSelection("disc", { cx: p.x, cy: p.y, r: st.brushRadius }), !e.altKey);
    } else if ((tool === "pos" || tool === "neg") && !d.moved) {
      const hit = sc.pick(e.clientX, e.clientY);
      if (hit) {
        const positive = tool === "pos";
        sc.addClickMarker(hit.world, positive);
        st.addClick({ x: hit.world.x, y: hit.world.y, z: hit.world.z, positive });
        if (st.realtime) annotator.runInfer();
        else st.setStatus(t("st.clickAdded", { sign: t(positive ? "st.pos" : "st.neg") }));
      }
    }
  };

  function applyManual(sel: ScreenSelection, add: boolean) {
    const st = useStore.getState();
    if (st.maskIndices.length > 0 || st.clicks.length > 0) {
      annotator.editMaskScreen(sel, add);
      st.setStatus(t(add ? "st.manualAdd" : "st.manualRemove"));
    } else if (st.selectedInstance != null) {
      annotator.assignScreen(sel, add ? st.selectedInstance : 0);
      st.setStatus(t("st.manualInst", { id: st.selectedInstance }));
    } else {
      st.setStatus(t("st.manualNeed"));
    }
  }

  return (
    <div
      className="viewport"
      ref={containerRef}
      onPointerDown={onDown}
      onPointerMove={onMove}
      onPointerUp={onUp}
      onPointerLeave={() => setBrushCursor(null)}
      style={{ cursor: tool === "orbit" ? "grab" : tool === "brush" ? "none" : CROSSHAIR }}
    >
      {selRect && (
        <div className="select-rect"
             style={{ left: selRect.x, top: selRect.y, width: selRect.w, height: selRect.h }} />
      )}
      {lassoPts.length > 1 && (
        <svg style={{ position: "absolute", inset: 0, pointerEvents: "none" }}>
          <polyline points={lassoPts.map((p) => p.join(",")).join(" ")}
                    fill="rgba(74,158,255,0.12)" stroke="#4a9eff" strokeDasharray="4 3" strokeWidth={1.5} />
        </svg>
      )}
      {tool === "brush" && brushCursor && (
        <div style={{
          position: "absolute", pointerEvents: "none",
          left: brushCursor.x - brushRadius, top: brushCursor.y - brushRadius,
          width: brushRadius * 2, height: brushRadius * 2, borderRadius: "50%",
          border: "1.5px solid #4a9eff", background: "rgba(74,158,255,0.10)",
        }} />
      )}
      {loading && (
        <div className="loading-overlay">
          <div className="spinner" />
          <div className="loading-msg">{loadingMsg}</div>
          <div className="hint">{t("load.wait")}</div>
        </div>
      )}
    </div>
  );
}
