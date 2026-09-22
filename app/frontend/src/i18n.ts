// Minimal bilingual (English default / Chinese) i18n. `t(key, params)` reads the
// current language from the store; `useT()` returns a `t` bound to it so React
// components re-render on language change. Placeholders use {name} syntax.
import { useStore, type Lang } from "./state/store";

type Entry = { en: string; zh: string };

const DICT: Record<string, Entry> = {
  // top bar
  "app.name": { en: "InterTreeSeg", zh: "InterTreeSeg" },
  "app.tagline": { en: "Interactive Tree Segmentation", zh: "单木交互分割标注" },
  "topbar.open": { en: "📂 Open File", zh: "📂 打开本地文件" },
  "topbar.undo": { en: "Undo (Ctrl+Z)", zh: "撤销 (Ctrl+Z)" },
  "topbar.redo": { en: "Redo (Ctrl+Shift+Z)", zh: "重做 (Ctrl+Shift+Z)" },
  "topbar.recompute": { en: "Σ Recompute geometry", zh: "Σ 批量算几何" },
  "topbar.exportPoints": { en: "⬇ Point labels", zh: "⬇ 点标签" },
  "topbar.exportAttrs": { en: "⬇ Attributes CSV", zh: "⬇ 属性 CSV" },
  "topbar.exportEval": { en: "⬇ Evaluation CSV", zh: "⬇ 评估 CSV" },
  "topbar.evalChip": {
    en: "Eval: {k} trees · mIoU {iou} · {c} clk/tree · total {s}s",
    zh: "评估: {k} 棵 · 平均IoU {iou} · {c} 点击/棵 · 总用时 {s}s",
  },
  "timer.start": { en: "▶ Start timer", zh: "▶ 开始计时" },
  "timer.stop": { en: "⏹ Stop", zh: "⏹ 停止" },
  "timer.reset": { en: "Reset timer", zh: "重置计时" },
  "timer.stopped": { en: "Timer stopped · total {s}s recorded.", zh: "计时停止 · 已记录总用时 {s} 秒。" },
  "topbar.pts": { en: "{n} pts", zh: "{n} 点" },
  "topbar.rendered": { en: " · rendered {n}", zh: " · 渲染 {n}" },

  // toolbar
  "tool.ai": { en: "AI Segmentation", zh: "AI 分割" },
  "tool.box": { en: "⬛ Top-view bounding box", zh: "⬛ 顶视框选" },
  "tool.pos": { en: "➕ Positive", zh: "➕ 正点击" },
  "tool.neg": { en: "➖ Negative", zh: "➖ 负点击" },
  "tool.run": { en: "▶ Run", zh: "▶ 运行" },
  "tool.realtime": { en: "⚡ Realtime", zh: "⚡ 实时" },
  "tool.realtimeTip": { en: "Realtime: infer on every click", zh: "实时：每次点击即推理" },
  "tool.manual": { en: "Manual refine", zh: "手工精修" },
  "tool.brush": { en: "🖌 Brush", zh: "🖌 刷子" },
  "tool.lasso": { en: "🎯 Lasso", zh: "🎯 套索" },
  "tool.brushFg": { en: "➕ Foreground", zh: "➕ 前景" },
  "tool.brushBg": { en: "➖ Background", zh: "➖ 背景" },
  "tool.brushBall": { en: "Surface ball", zh: "表面球" },
  "tool.brushThrough": { en: "Through-depth", zh: "穿透" },
  "tool.brushRadius": { en: "Brush radius: {r}px", zh: "笔刷半径: {r}px" },
  "tool.brushHint": {
    en: "Pick FG/BG. Ball = precise surface sphere · Through = reaches points behind (all depths).",
    zh: "先选 前景/背景。表面球=精准贴面 · 穿透=够到后面的点(所有深度)。",
  },
  "tool.lassoHint": {
    en: "Pick FG/BG, draw a loop to paint the enclosed points (all depths). Rotate first to isolate.",
    zh: "先选 前景/背景，圈出一片，框内点(所有深度)刷成该类。可先旋转视角隔离目标。",
  },
  "tool.hideLabel": { en: "🙈 Hide current label (H)", zh: "🙈 隐藏当前标注 (H)" },
  "tool.showLabel": { en: "👁 Show current label (H)", zh: "👁 显示当前标注 (H)" },
  "tool.view": { en: "View", zh: "视图" },
  "tool.colorBy": { en: "Color by", zh: "着色字段" },
  "field.height": { en: "Height", zh: "高度" },
  "field.intensity": { en: "Intensity", zh: "强度" },
  "field.gt_instance": { en: "GT tree (label)", zh: "GT 单木 (label)" },
  "field.gt_semantic": { en: "GT class", zh: "GT 类别" },
  "tool.orbit": { en: "🖐 Orbit / Pan", zh: "🖐 旋转 / 平移" },
  "tool.top": { en: "👁 Top view", zh: "👁 顶视" },
  "tool.reset": { en: "⟳ Reset view", zh: "⟳ 复位视角" },
  "tool.current": { en: "Current tree", zh: "当前树" },
  "tool.box2": { en: "Box:", zh: "框选:" },
  "tool.set": { en: "set", zh: "已设" },
  "tool.none": { en: "none", zh: "无" },
  "tool.clicks": { en: "Clicks:", zh: "点击:" },
  "tool.mask": { en: "mask: {n} pts", zh: "mask: {n} 点" },
  "tool.confirm": { en: "✓ Confirm as a tree", zh: "✓ 确认为一棵树" },
  "tool.discard": { en: "✕ Discard current", zh: "✕ 放弃当前" },

  // instance panel
  "inst.title": { en: "Instances ({n})", zh: "实例 ({n})" },
  "inst.empty": { en: "No confirmed trees yet.", zh: "还没有确认的树。" },
  "inst.attrs": { en: "Attributes · tree #{id}", zh: "属性 · 树 #{id}" },
  "inst.id": { en: "ID", zh: "ID" },
  "inst.species": { en: "Species", zh: "树种" },
  "inst.health": { en: "Growth status", zh: "生长状态" },
  "inst.height": { en: "Height (m)", zh: "树高 (m)" },
  "inst.crown": { en: "Crown (m)", zh: "冠幅 (m)" },
  "inst.dbh": { en: "DBH (cm)", zh: "DBH (cm)" },
  "inst.points": { en: "Points", zh: "点数" },
  "inst.stem": { en: "Stem XY", zh: "干基 XY" },
  "inst.notes": { en: "Notes", zh: "备注" },
  "inst.manual": { en: "edited", zh: "手改" },
  "inst.editHint": {
    en: "Select, then use lasso/brush to edit this tree (Left add / Alt remove).",
    zh: "选中后可用套索/刷子编辑该树（左键并入 / Alt 移出）。",
  },
  "inst.delete": { en: "🗑 Delete this tree", zh: "🗑 删除该树" },

  // attribute dialog
  "dlg.title": { en: "Confirm as tree #{id}", zh: "确认为树 #{id}" },
  "dlg.sub": {
    en: "{n} pts · geometry auto-computed (editable later in the panel)",
    zh: "{n} 点 · 几何量已自动计算（可在右侧属性面板再改）",
  },
  "dlg.cancel": { en: "Cancel", zh: "取消" },
  "dlg.confirm": { en: "Confirm", zh: "确认" },

  // bottom bar
  "bottom.progress": { en: "Labeled {k} trees · {pct}% points", zh: "已标注 {k} 棵 · 已标注点 {pct}%" },
  "bottom.infer": { en: " · infer {ms} ms", zh: " · 推理 {ms} ms" },

  // status messages
  "st.ready": { en: "Open a point cloud file to start.", zh: "打开一个点云文件开始标注。" },
  "st.loading": { en: "Uploading and loading {name}…", zh: "正在上传并加载 {name}…" },
  "st.loaded": { en: "Loaded {name} · {n} pts{dec}", zh: "已加载 {name} · {n} 点{dec}" },
  "st.loadedDec": { en: " (rendered decimated to {m})", zh: "（渲染抽稀至 {m}）" },
  "st.loadFail": { en: "Load failed: {msg}", zh: "加载失败: {msg}" },
  "st.geomDone": { en: "Batch geometry computed.", zh: "已批量计算几何属性。" },
  "st.infer": { en: "Inference: {n} pts, {ms} ms ({c} clicks)", zh: "推理: {n} 点, {ms} ms（{c} 次点击）" },
  "st.inferFail": { en: "Inference failed: {msg}", zh: "推理失败: {msg}" },
  "st.committed": { en: "Confirmed tree #{id} ({n} pts, full-res)", zh: "已确认树 #{id}（{n} 点，全分辨率）" },
  "st.committedEval": {
    en: "Confirmed tree #{id} ({n} pts) · IoU {iou} · {c} clicks · {s}s",
    zh: "已确认树 #{id}（{n} 点）· IoU {iou} · {c} 次点击 · {s} 秒",
  },
  "st.box": { en: "Vertical block cut — click positive/negative inside.", zh: "已切出垂直 block，请在框内点击正/负样本。" },
  "st.clickAdded": { en: "Added a {sign} click (batch mode — press Run).", zh: "已添加一个{sign}点击（批量模式，点“运行”推理）。" },
  "st.pos": { en: "positive", zh: "正" },
  "st.neg": { en: "negative", zh: "负" },
  "st.manualAdd": { en: "Manual add (full-res)…", zh: "手工加入（全分辨率）…" },
  "st.manualRemove": { en: "Manual remove (full-res)…", zh: "手工移出（全分辨率）…" },
  "st.manualInst": { en: "Editing tree #{id} (full-res)…", zh: "编辑树 #{id}（全分辨率）…" },
  "st.manualNeed": { en: "Frame a tree or select an instance before manual editing.", zh: "手工编辑前，请先框选一棵树或在右侧选中一个实例。" },
  "st.boxEmpty": { en: "Empty box — draw again.", zh: "框内没有点，请重新框选。" },
  "load.uploading": { en: "Uploading & processing point cloud…", zh: "正在上传并处理点云…" },
  "load.downloading": { en: "Downloading points…", zh: "正在下载点云数据…" },
  "load.rendering": { en: "Building 3D view…", zh: "正在构建三维视图…" },
  "load.wait": { en: "Large clouds may take a few seconds.", zh: "大点云可能需要几秒钟。" },
  "st.undone": { en: "Undone.", zh: "已撤销。" },
  "st.redone": { en: "Redone.", zh: "已重做。" },
  "st.nothingUndo": { en: "Nothing to undo.", zh: "没有可撤销的操作。" },
  "st.nothingRedo": { en: "Nothing to redo.", zh: "没有可重做的操作。" },
  "st.renamed": { en: "Renamed tree #{from} → #{to}.", zh: "已将树 #{from} 改为 #{to}。" },
  "st.renameFail": { en: "Rename failed: {msg}", zh: "改 ID 失败: {msg}" },
  "inst.idHint": { en: "editable — press Enter", zh: "可编辑，回车确认" },
};

export function translate(lang: Lang, key: string, params?: Record<string, unknown>): string {
  const entry = DICT[key];
  let s = entry ? entry[lang] ?? entry.en : key;
  if (params) {
    for (const k of Object.keys(params)) {
      s = s.replace(new RegExp("\\{" + k + "\\}", "g"), String(params[k]));
    }
  }
  return s;
}

/** Non-reactive translator (for use outside React, e.g. the annotator). */
export function t(key: string, params?: Record<string, unknown>): string {
  return translate(useStore.getState().lang, key, params);
}

/** Reactive translator hook — components re-render when the language changes. */
export function useT() {
  const lang = useStore((s) => s.lang);
  return (key: string, params?: Record<string, unknown>) => translate(lang, key, params);
}
