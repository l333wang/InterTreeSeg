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
  const brushMode = useStore((s) => s.brushMode);
  const setBrushMode = useStore((s) => s.setBrushMode);
  const brushShape = useStore((s) => s.brushShape);
  const setBrushShape = useStore((s) => s.setBrushShape);
  const colorField = useStore((s) => s.colorField);
  const setColorField = useStore((s) => s.setColorField);
  const clicks = useStore((s) => s.clicks);
  const maskCount = useStore((s) => s.maskIndices.length);
  const bbox = useStore((s) => s.bbox);
  const hideMask = useStore((s) => s.hideMask);
  const toggleHideMask = useStore((s) => s.toggleHideMask);

  const aiTools: { key: Tool; labelKey: string; cls?: string }[] = [
    { key: "bbox", labelKey: "tool.box" },
    { key: "pos", labelKey: "tool.pos", cls: "pos" },
    { key: "neg", labelKey: "tool.neg", cls: "neg" },
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
        <div className="row">
          <button className={tool === "brush" ? "active" : ""} onClick={() => setTool("brush")}>{t("tool.brush")}</button>
          <button className={tool === "lasso" ? "active" : ""} onClick={() => setTool("lasso")}>{t("tool.lasso")}</button>
        </div>
        {(tool === "brush" || tool === "lasso") && (
          <div className="row" style={{ marginTop: 4 }}>
            <button className={`pos ${brushMode === "fg" ? "active" : ""}`}
                    onClick={() => setBrushMode("fg")}>{t("tool.brushFg")}</button>
            <button className={`neg ${brushMode === "bg" ? "active" : ""}`}
                    onClick={() => setBrushMode("bg")}>{t("tool.brushBg")}</button>
          </div>
        )}
        {tool === "brush" && (
          <>
            <div className="row" style={{ marginTop: 4 }}>
              <button className={brushShape === "ball" ? "active" : ""}
                      onClick={() => setBrushShape("ball")}>{t("tool.brushBall")}</button>
              <button className={brushShape === "through" ? "active" : ""}
                      onClick={() => setBrushShape("through")}>{t("tool.brushThrough")}</button>
            </div>
            <div className="field" style={{ marginTop: 4 }}>
              <label>{t("tool.brushRadius", { r: brushRadius })}</label>
              <input type="range" min={6} max={60} value={brushRadius}
                     onChange={(e) => setBrushRadius(+e.target.value)} />
            </div>
          </>
        )}
        <div className="hint">{tool === "lasso" ? t("tool.lassoHint") : t("tool.brushHint")}</div>
      </div>

      <div className="tool-group">
        <div className="label">{t("tool.view")}</div>
        <button className={tool === "orbit" ? "active" : ""} onClick={() => setTool("orbit")}>{t("tool.orbit")}</button>
        <button onClick={() => annotator.scene?.topView()}>{t("tool.top")}</button>
        <button onClick={() => annotator.scene?.orbitView()}>{t("tool.reset")}</button>
        <div className="field" style={{ marginTop: 4 }}>
          <label>{t("tool.colorBy")}</label>
          <select value={colorField}
                  onChange={(e) => { setColorField(e.target.value as typeof colorField); annotator.applyColorField(); }}>
            <option value="height">{t("field.height")}</option>
            <option value="intensity">{t("field.intensity")}</option>
            <option value="gt_instance">{t("field.gt_instance")}</option>
            <option value="gt_semantic">{t("field.gt_semantic")}</option>
          </select>
        </div>
      </div>

      <div className="tool-group">
        <div className="label">{t("tool.current")}</div>
        <div className="hint">
          {t("tool.box2")} {bbox ? t("tool.set") : t("tool.none")}<br />
          {t("tool.clicks")} <span className="badge pos">{clicks.filter((c) => c.positive).length}+</span>{" "}
          <span className="badge neg">{clicks.filter((c) => !c.positive).length}−</span><br />
          {t("tool.mask", { n: maskCount })}
        </div>
        <button style={{ marginTop: 6 }} className={hideMask ? "active" : ""}
                onClick={() => { toggleHideMask(); annotator.recolor(); }}>
          {hideMask ? t("tool.showLabel") : t("tool.hideLabel")}
        </button>
        <button style={{ marginTop: 6 }} disabled={maskCount === 0}
                onClick={() => useStore.getState().setDialogOpen(true)}>{t("tool.confirm")}</button>
        <button onClick={() => annotator.clearWorking()}>{t("tool.discard")}</button>
      </div>
    </div>
  );
}
