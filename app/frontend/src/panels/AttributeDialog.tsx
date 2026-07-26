import { useMemo, useState } from "react";
import { useStore } from "../state/store";
import { annotator } from "../viewer/annotator";
import { useT } from "../i18n";

export default function AttributeDialog() {
  const t = useT();
  const open = useStore((s) => s.dialogOpen);
  const setOpen = useStore((s) => s.setDialogOpen);
  const maskIndices = useStore((s) => s.maskIndices);
  const config = useStore((s) => s.config);
  const nextId = useStore((s) => (s.instances.reduce((m, i) => Math.max(m, i.id), 0) + 1));

  const geom = useMemo(
    () => (open ? annotator.previewGeometry(maskIndices) : null),
    [open, maskIndices]
  );

  const [species, setSpecies] = useState("Unknown");
  const [health, setHealth] = useState("Alive");
  const [dbh, setDbh] = useState("");
  const [notes, setNotes] = useState("");

  if (!open || !geom) return null;

  async function confirm() {
    await annotator.commitCurrent({
      species,
      health_status: health,
      dbh: dbh === "" ? null : +dbh,
      notes,
    });
    setSpecies("Unknown"); setHealth("Alive"); setDbh(""); setNotes("");
    setOpen(false);
  }

  return (
    <div className="modal-backdrop" onClick={() => setOpen(false)}>
      <div className="modal" onClick={(e) => e.stopPropagation()}>
        <h3>{t("dlg.title", { id: nextId })}</h3>
        <div className="hint" style={{ marginBottom: 10 }}>
          {t("dlg.sub", { n: maskIndices.length.toLocaleString() })}
        </div>
        <div className="row">
          <div className="field"><label>{t("inst.height")}</label><input value={geom.height} disabled /></div>
          <div className="field"><label>{t("inst.crown")}</label><input value={geom.crown_width} disabled /></div>
        </div>
        <div className="field">
          <label>{t("inst.species")}</label>
          <select value={species} onChange={(e) => setSpecies(e.target.value)}>
            {(config?.species ?? ["Unknown"]).map((s) => <option key={s}>{s}</option>)}
          </select>
        </div>
        <div className="row">
          <div className="field">
            <label>{t("inst.health")}</label>
            <select value={health} onChange={(e) => setHealth(e.target.value)}>
              {(config?.health_status ?? ["Alive"]).map((s) => <option key={s}>{s}</option>)}
            </select>
          </div>
          <div className="field">
            <label>{t("inst.dbh")}</label>
            <input type="number" step="0.1" value={dbh} onChange={(e) => setDbh(e.target.value)} />
          </div>
        </div>
        <div className="field">
          <label>{t("inst.notes")}</label>
          <input value={notes} onChange={(e) => setNotes(e.target.value)} />
        </div>
        <div className="actions">
          <button onClick={() => setOpen(false)}>{t("dlg.cancel")}</button>
          <button className="active" onClick={confirm}>{t("dlg.confirm")}</button>
        </div>
      </div>
    </div>
  );
}
