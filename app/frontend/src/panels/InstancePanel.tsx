import { useStore } from "../state/store";
import { annotator } from "../viewer/annotator";
import { api } from "../api/client";
import type { Instance } from "../api/client";
import { useT } from "../i18n";

export default function InstancePanel() {
  const t = useT();
  const instances = useStore((s) => s.instances);
  const selected = useStore((s) => s.selectedInstance);
  const hidden = useStore((s) => s.hiddenInstances);
  const select = useStore((s) => s.selectInstance);
  const toggleVis = useStore((s) => s.toggleVisibility);
  const config = useStore((s) => s.config);
  const sessionId = useStore((s) => s.sessionId);

  const sel = instances.find((i) => i.id === selected) || null;

  async function patch(attrs: Partial<Instance>) {
    if (!sessionId || sel == null) return;
    await api.updateInstance(sessionId, sel.id, attrs);
    await annotator.refreshInstances();
  }

  return (
    <div className="rightpanel">
      <div className="section-title">{t("inst.title", { n: instances.length })}</div>
      {instances.length === 0 && <div className="hint">{t("inst.empty")}</div>}
      {instances.map((i) => (
        <div key={i.id} className={`instance ${selected === i.id ? "selected" : ""}`}
             onClick={() => select(selected === i.id ? null : i.id)}>
          <span className="swatch" style={{ background: rgb(i.color) }} />
          <span className="name">#{i.id} <span className="meta">{i.species}</span></span>
          <span className="meta">{i.height}m</span>
          <span className="vis" onClick={(e) => { e.stopPropagation(); toggleVis(i.id); }}>
            {hidden.has(i.id) ? "🚫" : "👁"}
          </span>
        </div>
      ))}

      {sel && (
        <div style={{ marginTop: 14, borderTop: "1px solid var(--border)", paddingTop: 12 }}>
          <div className="section-title">{t("inst.attrs", { id: sel.id })}</div>
          <div className="field">
            <label>{t("inst.id")} <span className="derived">{t("inst.idHint")}</span></label>
            <input type="number" min={1} step={1} key={sel.id} defaultValue={sel.id}
                   onKeyDown={(e) => { if (e.key === "Enter") (e.target as HTMLInputElement).blur(); }}
                   onBlur={(e) => {
                     const v = parseInt(e.target.value, 10);
                     if (v && v !== sel.id) annotator.renameInstance(sel.id, v);
                   }} />
          </div>
          <div className="field">
            <label>{t("inst.species")}</label>
            <select value={sel.species} onChange={(e) => patch({ species: e.target.value })}>
              {(config?.species ?? [sel.species]).map((s) => <option key={s}>{s}</option>)}
            </select>
          </div>
          <div className="field">
            <label>{t("inst.health")}</label>
            <select value={sel.health_status} onChange={(e) => patch({ health_status: e.target.value })}>
              {(config?.health_status ?? [sel.health_status]).map((s) => <option key={s}>{s}</option>)}
            </select>
          </div>
          <div className="row">
            <div className="field">
              <label>{t("inst.height")} {sel.manual_fields.includes("height") && <span className="derived">{t("inst.manual")}</span>}</label>
              <input type="number" step="0.01" value={sel.height}
                     onChange={(e) => patch({ height: +e.target.value })} />
            </div>
            <div className="field">
              <label>{t("inst.crown")}</label>
              <input type="number" step="0.01" value={sel.crown_width}
                     onChange={(e) => patch({ crown_width: +e.target.value })} />
            </div>
          </div>
          <div className="row">
            <div className="field">
              <label>{t("inst.dbh")}</label>
              <input type="number" step="0.1" value={sel.dbh ?? ""}
                     onChange={(e) => patch({ dbh: e.target.value === "" ? null : +e.target.value })} />
            </div>
            <div className="field">
              <label>{t("inst.points")}</label>
              <input value={sel.point_count} disabled />
            </div>
          </div>
          <div className="field">
            <label>{t("inst.stem")}</label>
            <input value={`${sel.stem_x}, ${sel.stem_y}`} disabled />
          </div>
          <div className="field">
            <label>{t("inst.notes")}</label>
            <input defaultValue={sel.notes} onBlur={(e) => patch({ notes: e.target.value })} />
          </div>
          <div className="hint" style={{ marginBottom: 6 }}>{t("inst.editHint")}</div>
          <button style={{ width: "100%" }} onClick={() => annotator.deleteInstance(sel.id)}>
            {t("inst.delete")}
          </button>
        </div>
      )}
    </div>
  );
}

function rgb(c: [number, number, number]) {
  return `rgb(${c.map((v) => Math.round(v * 255)).join(",")})`;
}
