import { useRef } from "react";
import { useStore } from "../state/store";
import { annotator } from "../viewer/annotator";
import { api, type LoadResponse } from "../api/client";
import { useT, t as tRaw } from "../i18n";

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
      <div className="spacer" />
      {sessionId && (
        <>
          <span className="hint">
            {t("topbar.pts", { n: numPointsFull.toLocaleString() })}
            {decimated ? t("topbar.rendered", { n: numPoints.toLocaleString() }) : ""}
          </span>
          <button onClick={recomputeGeometry}>{t("topbar.recompute")}</button>
          <button onClick={() => download(api.exportPointsUrl(sessionId))}>{t("topbar.exportPoints")}</button>
          <button onClick={() => download(api.exportAttributesUrl(sessionId))}>{t("topbar.exportAttrs")}</button>
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
