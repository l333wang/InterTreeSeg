// Thin REST/binary client for the annotation backend. All calls go to
// same-origin /api (Vite proxies to FastAPI in dev).

export interface Bounds { min: [number, number, number]; max: [number, number, number]; }

export interface LoadResponse {
  session_id: string;
  num_points: number;
  num_points_full: number;
  decimated: boolean;
  bounds: Bounds;
}

export interface Click { x: number; y: number; z: number; positive: boolean; }
export interface BBox {
  x_min: number; y_min: number; x_max: number; y_max: number;
  z_min?: number | null; z_max?: number | null;
}
export interface InferResponse { mask_indices: number[]; count: number; elapsed_ms: number; }

export interface Instance {
  id: number;
  color: [number, number, number];
  species: string;
  dbh: number | null;
  health_status: string;
  notes: string;
  height: number;
  crown_width: number;
  stem_x: number;
  stem_y: number;
  point_count: number;
  manual_fields: string[];
}

export interface AppConfig { species: string[]; health_status: string[]; }

export interface ScreenSelection {
  view_proj: number[];   // three.js Matrix4.elements (16)
  vw: number;
  vh: number;
  kind: "polygon" | "disc";
  polygon?: number[][];
  cx?: number;
  cy?: number;
  r?: number;
}

async function jpost<T>(url: string, body: unknown): Promise<T> {
  const r = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!r.ok) throw new Error((await r.json().catch(() => ({}))).detail || r.statusText);
  return r.json();
}

export const api = {
  getConfig: (): Promise<AppConfig> => fetch("/api/config").then((r) => r.json()),

  load: (file_path: string) => jpost<LoadResponse>("/api/sessions/load", { file_path }),

  async upload(file: File): Promise<LoadResponse> {
    const fd = new FormData();
    fd.append("file", file);
    const r = await fetch("/api/sessions/upload", { method: "POST", body: fd });
    if (!r.ok) throw new Error((await r.json().catch(() => ({}))).detail || r.statusText);
    return r.json();
  },

  // Binary: Float32 [x,y,z,scalar] * N
  async points(sid: string): Promise<Float32Array> {
    const buf = await fetch(`/api/sessions/${sid}/points`).then((r) => r.arrayBuffer());
    return new Float32Array(buf);
  },

  // Binary: Int32 instance-id * N
  async labels(sid: string): Promise<Int32Array> {
    const buf = await fetch(`/api/sessions/${sid}/labels`).then((r) => r.arrayBuffer());
    return new Int32Array(buf);
  },

  infer: (sid: string, bbox: BBox | null, clicks: Click[]) =>
    jpost<InferResponse>(`/api/sessions/${sid}/infer`, { bbox, clicks }),

  commit: (sid: string, attributes: Partial<Instance>) =>
    jpost<{ instance: Instance; display_indices: number[] }>(
      `/api/sessions/${sid}/commit`, { attributes }),

  editMask: (sid: string, selection: ScreenSelection, add: boolean) =>
    jpost<{ mask_indices: number[]; count: number }>(
      `/api/sessions/${sid}/mask/edit`, { selection, add }),

  clearMask: (sid: string) => jpost(`/api/sessions/${sid}/mask/clear`, {}),

  instances: (sid: string) =>
    fetch(`/api/sessions/${sid}/instances`).then((r) => r.json() as Promise<{ instances: Instance[] }>),

  updateInstance: (sid: string, iid: number, attributes: Partial<Instance>) =>
    fetch(`/api/sessions/${sid}/instances/${iid}`, {
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(attributes),
    }).then((r) => r.json() as Promise<{ instance: Instance }>),

  renameInstance: (sid: string, iid: number, new_id: number) =>
    jpost<{ instance: Instance }>(`/api/sessions/${sid}/instances/${iid}/rename`, { new_id }),

  deleteInstance: (sid: string, iid: number) =>
    fetch(`/api/sessions/${sid}/instances/${iid}`, { method: "DELETE" }).then((r) => r.json()),

  assign: (sid: string, selection: ScreenSelection, target: number) =>
    jpost(`/api/sessions/${sid}/assign`, { selection, target }),

  computeGeometry: (sid: string) =>
    jpost<{ instances: Instance[] }>(`/api/sessions/${sid}/compute_geometry`, {}),

  exportPointsUrl: (sid: string) => `/api/sessions/${sid}/export/points`,
  exportAttributesUrl: (sid: string) => `/api/sessions/${sid}/export/attributes`,
};
