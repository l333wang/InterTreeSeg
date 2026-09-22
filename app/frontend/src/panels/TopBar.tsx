import { useEffect, useRef, useState } from "react";
import { useStore } from "../state/store";
import { annotator } from "../viewer/annotator";
import { api, type LoadResponse } from "../api/client";
import { useT, t as tRaw } from "../i18n";

function fmtClock(ms: number): string {
  const s = Math.floor(ms / 1000);
  return `${Math.floor(s / 60)}:${String(s % 60).padStart(2, "0")}`;
}

/** Manual Start/Stop stopwatch: measures hands-on annotation time for the scene. */
function SessionTimer({ sessionId }: { sessionId: string }) {
  const t = useT();
  const running = useStore((s) => s.timerRunning);
  const startedAt = useStore((s) => s.timerStartedAt);
  const accumMs = useStore((s) => s.timerAccumMs);
  const startTimer = useStore((s) => s.startTimer);
  const stopTimer = useStore((s) => s.stopTimer);
  const resetTimer = useStore((s) => s.resetTimer);
  const [, force] = useState(0);

  useEffect(() => {
    if (!running) return;
    const id = setInterval(() => force((n) => n + 1), 250);
    return () => clearInterval(id);
  }, [running]);

  const elapsed = accumMs + (running && startedAt ? Date.now() - startedAt : 0);

  async function onStop() {
    const seconds = stopTimer();
    try {
      await api.setSessionTime(sessionId, seconds);
      const e = await api.evaluation(sessionId);
      useStore.getState().setEvalSummary(e.summary);
    } catch {
      /* non-fatal */
    }
    useStore.getState().setStatus(tRaw("timer.stopped", { s: seconds.toFixed(1) }));
  }

  return (
    <span style={{ display: "inline-flex", alignItems: "center", gap: 4, marginRight: 4 }}>
      <span className="hint" style={{ fontVariantNumeric: "tabular-nums", minWidth: 46, textAlign: "right" }}>
        ⏱ {fmtClock(elapsed)}
      </span>
      {running ? (
        <button className="active" onClick={onStop}>{t("timer.stop")}</button>
      ) : (
        <button onClick={startTimer}>{t("timer.start")}</button>
      )}
      {!running && accumMs > 0 && (
        <button onClick={resetTimer} title={t("timer.reset")} style={{ padding: "4px 7px" }}>↺</button>
      )}
    </span>
  );
}

export default function TopBar() {
  const fileRef = useRef<HTMLInputElement>(null);
  const t = useT();
  const lang = useStore((s) => s.lang);
  const setLang = useStore((s) => s.setLang);
  const sessionId = useStore((s) => s.sessionId);
  const setSession = useStore((s) => s.setSession);
  const setStatus = useStore((s) => s.setStatus);
  const setLoading = useStore((s) => s.setLoading);
  const decimated = useStore((s) => s.decimated);
  const numPoints = useStore((s) => s.numPoints);
  const numPointsFull = useStore((s) => s.numPointsFull);
  const evalSummary = useStore((s) => s.evalSummary);

  function applyLoaded(r: LoadResponse, label: string) {
    setSession({
      sessionId: r.session_id, bounds: r.bounds, numPoints: r.num_points,
      numPointsFull: r.num_points_full, decimated: r.decimated,
    });
    const dec = r.decimated ? tRaw("st.loadedDec", { m: r.num_points.toLocaleString() }) : "";
    setStatus(tRaw("st.loaded", { name: label, n: r.num_points_full.toLocaleString(), dec }));
  }

  async function onFile(e: React.ChangeEvent<HTMLInputElement>) {
    const f = e.target.files?.[0];
    if (!f) return;
    setLoading(tRaw("load.uploading"));   // overlay stays until loadScene finishes rendering
    setStatus(tRaw("st.loading", { name: f.name }));
    try {
      applyLoaded(await api.upload(f), f.name);
    } catch (err) {
      setLoading(null);
      setStatus(tRaw("st.loadFail", { msg: (err as Error).message }));
    } finally {
      e.target.value = "";
    }
  }

  async function recomputeGeometry() {
    if (!sessionId) return;
    await api.computeGeometry(sessionId);
    await annotator.refreshInstances();
    setStatus(tRaw("st.geomDone"));
  }

  function download(url: string) {
    if (!sessionId) return;
    const a = document.createElement("a");
    a.href = url;
    a.click();
  }

  return (
    <div className="topbar">
      <span className="title">🌲 {t("app.name")}</span>
      <span className="hint" style={{ marginRight: 8 }}>{t("app.tagline")}</span>
      <input ref={fileRef} type="file" accept=".txt" style={{ display: "none" }} onChange={onFile} />
      <button onClick={() => fileRef.current?.click()}>{t("topbar.open")}</button>
      {sessionId && (
        <>
          <button title={t("topbar.undo")} onClick={() => annotator.undo()}>↶</button>
          <button title={t("topbar.redo")} onClick={() => annotator.redo()}>↷</button>
        </>
      )}
      <div className="spacer" />
      {sessionId && (
        <>
          <span className="hint">
            {t("topbar.pts", { n: numPointsFull.toLocaleString() })}
            {decimated ? t("topbar.rendered", { n: numPoints.toLocaleString() }) : ""}
          </span>
          <SessionTimer sessionId={sessionId} />
          {evalSummary && evalSummary.n_trees > 0 && (
            <span className="hint" style={{ marginLeft: 4 }}>
              {t("topbar.evalChip", {
                k: evalSummary.n_trees,
                iou: evalSummary.mean_iou == null ? "—" : evalSummary.mean_iou.toFixed(3),
                c: evalSummary.mean_clicks == null ? "—" : evalSummary.mean_clicks,
                s: evalSummary.session_time_s == null ? "—" : evalSummary.session_time_s,
              })}
            </span>
          )}
          <button onClick={recomputeGeometry}>{t("topbar.recompute")}</button>
          <button onClick={() => download(api.exportPointsUrl(sessionId))}>{t("topbar.exportPoints")}</button>
          <button onClick={() => download(api.exportAttributesUrl(sessionId))}>{t("topbar.exportAttrs")}</button>
          <button onClick={() => download(api.exportEvaluationUrl(sessionId))}>{t("topbar.exportEval")}</button>
        </>
      )}
      <div style={{ display: "flex", gap: 2, marginLeft: 8 }}>
        <button className={lang === "en" ? "active" : ""} onClick={() => setLang("en")}
                style={{ padding: "4px 7px" }}>EN</button>
        <button className={lang === "zh" ? "active" : ""} onClick={() => setLang("zh")}
                style={{ padding: "4px 7px" }}>中文</button>
      </div>
    </div>
  );
}
