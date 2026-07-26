import { useStore, type Tool } from "../state/store";
import { annotator } from "../viewer/annotator";
import { useT } from "../i18n";

export default function Toolbar() {
  const t = useT();
  const tool = useStore((s) => s.tool);
  const setTool = useStore((s) => s.setTool);
  const realtime = useStore((s) => s.realtime);
  const toggleRealtime = useStore((s) => s.toggleRealtime);
  const brushRadius = useStore((s) => s.brushRadius);
  const setBrushRadius = useStore((s) => s.setBrushRadius);
  const clicks = useStore((s) => s.clicks);
  const maskCount = useStore((s) => s.maskIndices.length);
  const bbox = useStore((s) => s.bbox);

  const aiTools: { key: Tool; labelKey: string; cls?: string }[] = [
    { key: "bbox", labelKey: "tool.box" },
    { key: "pos", labelKey: "tool.pos", cls: "pos" },
    { key: "neg", labelKey: "tool.neg", cls: "neg" },
  ];
  const manualTools: { key: Tool; labelKey: string }[] = [
    { key: "lasso", labelKey: "tool.lasso" },
    { key: "brush", labelKey: "tool.brush" },
  ];

  return (
    <div className="toolbar">
      <div className="tool-group">
        <div className="label">{t("tool.ai")}</div>
        {aiTools.map((x) => (
          <button key={x.key} className={`${x.cls ?? ""} ${tool === x.key ? "active" : ""}`}
                  onClick={() => setTool(x.key)}>{t(x.labelKey)}</button>
        ))}
        <div className="row" style={{ marginTop: 4 }}>
          <button onClick={() => annotator.runInfer()} disabled={clicks.length === 0}>{t("tool.run")}</button>
          <button className={realtime ? "active" : ""} onClick={toggleRealtime}
                  title={t("tool.realtimeTip")}>{t("tool.realtime")}</button>
        </div>
      </div>

      <div className="tool-group">
        <div className="label">{t("tool.manual")}</div>
        {manualTools.map((x) => (
          <button key={x.key} className={tool === x.key ? "active" : ""}
                  onClick={() => setTool(x.key)}>{t(x.labelKey)}</button>
        ))}
        {tool === "brush" && (
          <div className="field" style={{ marginTop: 4 }}>
            <label>{t("tool.brushRadius", { r: brushRadius })}</label>
            <input type="range" min={6} max={60} value={brushRadius}
                   onChange={(e) => setBrushRadius(+e.target.value)} />
          </div>
        )}
        <div className="hint">{t("tool.manualHint")}</div>
      </div>

      <div className="tool-group">
        <div className="label">{t("tool.view")}</div>
        <button className={tool === "orbit" ? "active" : ""} onClick={() => setTool("orbit")}>{t("tool.orbit")}</button>
        <button onClick={() => annotator.scene?.topView()}>{t("tool.top")}</button>
        <button onClick={() => annotator.scene?.orbitView()}>{t("tool.reset")}</button>
      </div>

      <div className="tool-group">
        <div className="label">{t("tool.current")}</div>
        <div className="hint">
          {t("tool.box2")} {bbox ? t("tool.set") : t("tool.none")}<br />
          {t("tool.clicks")} <span className="badge pos">{clicks.filter((c) => c.positive).length}+</span>{" "}
          <span className="badge neg">{clicks.filter((c) => !c.positive).length}−</span><br />
          {t("tool.mask", { n: maskCount })}
        </div>
        <button style={{ marginTop: 6 }} disabled={maskCount === 0}
                onClick={() => useStore.getState().setDialogOpen(true)}>{t("tool.confirm")}</button>
        <button onClick={() => annotator.clearWorking()}>{t("tool.discard")}</button>
      </div>
    </div>
  );
}
