import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
import type { Bounds } from "../api/client";

const HIGHLIGHT: [number, number, number] = [1.0, 0.9, 0.15]; // current tree

/** Imperative three.js manager for one point cloud + its overlays. */
export class PointCloudScene {
  renderer: THREE.WebGLRenderer;
  scene: THREE.Scene;
  camera: THREE.PerspectiveCamera;          // orbit / 3D inspection
  orthoCam: THREE.OrthographicCamera;       // top-view bbox drawing (no perspective distortion)
  activeCamera: THREE.Camera;               // whichever is currently rendering
  controls: OrbitControls;

  private container: HTMLElement;
  private points?: THREE.Points;
  private geom?: THREE.BufferGeometry;
  private baseColor?: Float32Array;   // height colormap, 3N
  private colorAttr?: THREE.BufferAttribute;
  private alphaAttr?: THREE.BufferAttribute;
  private n = 0;
  private center = new THREE.Vector3();
  private sceneSize = 1;
  private zmin = 0;

  private clickMarkers = new THREE.Group();
  private raycaster = new THREE.Raycaster();

  constructor(container: HTMLElement) {
    this.container = container;
    this.renderer = new THREE.WebGLRenderer({ antialias: true });
    this.renderer.setPixelRatio(window.devicePixelRatio);
    this.renderer.setSize(container.clientWidth, container.clientHeight);
    this.renderer.setClearColor(0x101214);
    container.appendChild(this.renderer.domElement);

    this.scene = new THREE.Scene();
    this.scene.add(this.clickMarkers);

    this.camera = new THREE.PerspectiveCamera(
      55, container.clientWidth / container.clientHeight, 0.1, 5000
    );
    this.camera.up.set(0, 0, 1); // Z-up (forestry point clouds)

    this.orthoCam = new THREE.OrthographicCamera(-1, 1, 1, -1, 0.01, 10000);
    this.orthoCam.up.set(0, 0, 1);
    this.activeCamera = this.camera;

    this.controls = new OrbitControls(this.camera, this.renderer.domElement);
    this.controls.enableDamping = true;
    this.controls.dampingFactor = 0.12;

    this.animate = this.animate.bind(this);
    this.animate();
    window.addEventListener("resize", this.onResize);
  }

  private onResize = () => {
    const w = this.container.clientWidth, h = this.container.clientHeight;
    this.camera.aspect = w / h;
    this.camera.updateProjectionMatrix();
    this._updateOrthoFrustum();
    this.renderer.setSize(w, h);
  };

  private _updateOrthoFrustum() {
    const w = this.container.clientWidth, h = this.container.clientHeight;
    const aspect = w / Math.max(1, h);
    const half = this.sceneSize * 0.55;
    this.orthoCam.left = -half * aspect;
    this.orthoCam.right = half * aspect;
    this.orthoCam.top = half;
    this.orthoCam.bottom = -half;
    this.orthoCam.near = 0.01;
    this.orthoCam.far = this.sceneSize * 4;
    this.orthoCam.updateProjectionMatrix();
  }

  private animate() {
    requestAnimationFrame(this.animate);
    this.controls.update();
    this.renderer.render(this.scene, this.activeCamera);
  }

  /** Load interleaved [x,y,z,scalar]*N Float32 buffer. */
  loadPoints(interleaved: Float32Array, bounds: Bounds) {
    if (this.points) {
      this.scene.remove(this.points);
      this.geom?.dispose();
    }
    const n = interleaved.length / 4;
    this.n = n;
    const pos = new Float32Array(n * 3);
    const col = new Float32Array(n * 3);
    const alpha = new Float32Array(n).fill(1);
    // Color by height (z) — the conventional forestry point-cloud visualization.
    const zLo = bounds.min[2];
    const zRange = bounds.max[2] - bounds.min[2] || 1;
    for (let i = 0; i < n; i++) {
      const z = interleaved[i * 4 + 2];
      pos[i * 3] = interleaved[i * 4];
      pos[i * 3 + 1] = interleaved[i * 4 + 1];
      pos[i * 3 + 2] = z;
      const t = (z - zLo) / zRange;
      const [r, g, b] = ramp(t);
      col[i * 3] = r;
      col[i * 3 + 1] = g;
      col[i * 3 + 2] = b;
    }
    this.baseColor = col.slice();   // full-brightness height colors; dimming applied in recolor()

    const geom = new THREE.BufferGeometry();
    geom.setAttribute("position", new THREE.BufferAttribute(pos, 3));
    this.colorAttr = new THREE.BufferAttribute(col, 3);
    this.alphaAttr = new THREE.BufferAttribute(alpha, 1);
    geom.setAttribute("pcolor", this.colorAttr);
    geom.setAttribute("alpha", this.alphaAttr);
    this.geom = geom;

    const mat = new THREE.ShaderMaterial({
      uniforms: { size: { value: 2.0 } },
      vertexShader: VERT,
      fragmentShader: FRAG,
      transparent: true,
    });
    this.points = new THREE.Points(geom, mat);
    this.scene.add(this.points);

    // frame the scene
    const mn = new THREE.Vector3(...bounds.min);
    const mx = new THREE.Vector3(...bounds.max);
    this.center.copy(mn).add(mx).multiplyScalar(0.5);
    this.sceneSize = mn.distanceTo(mx);
    this.zmin = bounds.min[2];
    this.raycaster.params.Points!.threshold = this.sceneSize * 0.004;
    this.controls.target.copy(this.center);
    this.orbitView();
  }

  setPointSize(s: number) {
    if (this.points) (this.points.material as THREE.ShaderMaterial).uniforms.size.value = s;
  }

  /** Recompute the base (unlabeled) point colors from a per-point scalar field.
   *  continuous fields (height/intensity) use the height ramp; categorical fields
   *  (label ids) get a distinct color per value (0 = gray). Caller then recolors. */
  setBaseColorField(values: Float32Array, categorical: boolean) {
    if (!this.baseColor) return;
    const n = Math.min(this.n, values.length);
    const col = this.baseColor;
    if (categorical) {
      for (let i = 0; i < n; i++) {
        const v = values[i];
        const [r, g, b] = v === 0 ? [0.35, 0.35, 0.35] : catColor(v);
        col[i * 3] = r; col[i * 3 + 1] = g; col[i * 3 + 2] = b;
      }
    } else {
      let lo = Infinity, hi = -Infinity;
      for (let i = 0; i < n; i++) { const v = values[i]; if (v < lo) lo = v; if (v > hi) hi = v; }
      const rng = hi - lo || 1;
      for (let i = 0; i < n; i++) {
        const [r, g, b] = ramp((values[i] - lo) / rng);
        col[i * 3] = r; col[i * 3 + 1] = g; col[i * 3 + 2] = b;
      }
    }
  }

  orbitView() {
    const d = this.sceneSize * 0.7;
    this.activeCamera = this.camera;
    this.controls.object = this.camera;
    this.camera.position.set(this.center.x + d, this.center.y - d, this.center.z + d * 0.6);
    this.controls.target.copy(this.center);
    this.controls.enableRotate = true;
    this.controls.update();
  }

  /** Orthographic top-down view: screen rectangle maps linearly to world XY,
   *  so a drawn bbox neither shifts nor expands with point height. Rotation locked. */
  topView() {
    this._updateOrthoFrustum();
    this.orthoCam.position.set(this.center.x, this.center.y, this.center.z + this.sceneSize * 2);
    this.orthoCam.zoom = 1;
    this.orthoCam.updateProjectionMatrix();
    this.activeCamera = this.orthoCam;
    this.controls.object = this.orthoCam;
    this.controls.target.copy(this.center);
    this.controls.enableRotate = false;
    this.controls.update();
  }

  setRotateEnabled(v: boolean) { this.controls.enableRotate = v; }

  /** Nearest point to a screen position → {index, world}. */
  pick(clientX: number, clientY: number): { index: number; world: THREE.Vector3 } | null {
    if (!this.points) return null;
    const ndc = this.toNDC(clientX, clientY);
    this.raycaster.setFromCamera(ndc, this.activeCamera);
    const hits = this.raycaster.intersectObject(this.points);
    if (!hits.length) return null;
    const h = hits[0];
    return { index: h.index ?? 0, world: h.point.clone() };
  }

  /** Intersect the view ray with a horizontal plane at z = zPlane. */
  groundPoint(clientX: number, clientY: number, zPlane?: number): THREE.Vector3 | null {
    const ndc = this.toNDC(clientX, clientY);
    this.raycaster.setFromCamera(ndc, this.activeCamera);
    const plane = new THREE.Plane(new THREE.Vector3(0, 0, 1), -(zPlane ?? this.zmin));
    const out = new THREE.Vector3();
    return this.raycaster.ray.intersectPlane(plane, out) ? out : null;
  }

  private toNDC(clientX: number, clientY: number): THREE.Vector2 {
    const rect = this.renderer.domElement.getBoundingClientRect();
    return new THREE.Vector2(
      ((clientX - rect.left) / rect.width) * 2 - 1,
      -((clientY - rect.top) / rect.height) * 2 + 1
    );
  }

  /** Project a world position to screen (container-relative) pixels. */
  worldToScreen(world: THREE.Vector3): { x: number; y: number } {
    const rect = this.renderer.domElement.getBoundingClientRect();
    const v = world.clone().project(this.activeCamera);
    return { x: ((v.x + 1) / 2) * rect.width, y: ((1 - v.y) / 2) * rect.height };
  }

  positionsRef(): Float32Array | undefined {
    return this.geom?.getAttribute("position").array as Float32Array | undefined;
  }

  /** World-space radius corresponding to a screen pixel radius AT a given point's
   *  depth (works for both perspective & ortho). Lets a pixel-sized brush drive a
   *  true 3D ball query. */
  pixelToWorldRadius(world: THREE.Vector3, pixelRadius: number): number {
    const rect = this.renderer.domElement.getBoundingClientRect();
    const ndc = world.clone().project(this.activeCamera);
    const dxNdc = (2 * pixelRadius) / Math.max(1, rect.width);
    const edge = new THREE.Vector3(ndc.x + dxNdc, ndc.y, ndc.z).unproject(this.activeCamera);
    return world.distanceTo(edge);
  }

  /** Camera view-projection + viewport, for backend full-res screen selection. */
  getViewProj(): { view_proj: number[]; vw: number; vh: number } {
    this.activeCamera.updateMatrixWorld();
    const vp = new THREE.Matrix4().multiplyMatrices(
      this.activeCamera.projectionMatrix, this.activeCamera.matrixWorldInverse
    );
    const rect = this.renderer.domElement.getBoundingClientRect();
    return { view_proj: Array.from(vp.elements), vw: rect.width, vh: rect.height };
  }

  /** Full recolor: base(height) → instance colors → current mask highlight. */
  /** Recolor: current mask → highlight; committed → instance color; unlabeled →
   *  height color × bgDim (dimmed while focusing on one tree). */
  recolor(
    labels: Int32Array,
    maskSet: Set<number> | null,
    hidden: Set<number>,
    instanceColors: Map<number, [number, number, number]>,
    bgDim = 0.9,
    hideMaskPoints = false
  ) {
    if (!this.colorAttr || !this.alphaAttr || !this.baseColor) return;
    const col = this.colorAttr.array as Float32Array;
    const alpha = this.alphaAttr.array as Float32Array;
    for (let i = 0; i < this.n; i++) {
      const lab = labels[i];
      let r: number, g: number, b: number, a = 1;
      if (maskSet && maskSet.has(i)) {
        // hideMaskPoints: hide the current selection entirely (label + its raw
        // points), so the selected tree disappears from view; else highlight it.
        if (hideMaskPoints) { r = g = b = 0; a = 0; }
        else { [r, g, b] = HIGHLIGHT; }
      } else if (lab > 0) {
        if (hidden.has(lab)) { a = 0; }
        const c = instanceColors.get(lab);
        if (c) { [r, g, b] = c; } else { r = 0.6; g = 0.6; b = 0.6; }
      } else {
        r = this.baseColor[i * 3] * bgDim;
        g = this.baseColor[i * 3 + 1] * bgDim;
        b = this.baseColor[i * 3 + 2] * bgDim;
      }
      col[i * 3] = r!; col[i * 3 + 1] = g!; col[i * 3 + 2] = b!;
      alpha[i] = a;
    }
    this.colorAttr.needsUpdate = true;
    this.alphaAttr.needsUpdate = true;
  }

  // ---- bbox helper -------------------------------------------------------
  private bboxHelper?: THREE.Box3Helper;
  setBBoxHelper(bbox: { x_min: number; y_min: number; x_max: number; y_max: number } | null,
               zMin: number, zMax: number) {
    if (this.bboxHelper) { this.scene.remove(this.bboxHelper); this.bboxHelper = undefined; }
    if (!bbox) return;
    const box = new THREE.Box3(
      new THREE.Vector3(bbox.x_min, bbox.y_min, zMin),
      new THREE.Vector3(bbox.x_max, bbox.y_max, zMax)
    );
    this.bboxHelper = new THREE.Box3Helper(box, new THREE.Color(0x4a9eff));
    this.scene.add(this.bboxHelper);
  }
  get zMin() { return this.zmin; }

  /** Unproject a screen pixel (container-relative) to world XY. In the ortho
   *  top-down view this is exact and height-independent — a CloudCompare-style
   *  vertical cut: the drawn rectangle IS the world XY rectangle. */
  screenToWorldXY(px: number, py: number): { x: number; y: number } {
    const rect = this.renderer.domElement.getBoundingClientRect();
    const nx = (px / rect.width) * 2 - 1;
    const ny = -((py / rect.height) * 2 - 1);
    const v = new THREE.Vector3(nx, ny, 0).unproject(this.activeCamera);
    return { x: v.x, y: v.y };
  }

  /** World XY bbox of the exact drawn rectangle (two screen corners unprojected). */
  bboxFromScreenCorners(x0: number, y0: number, x1: number, y1: number) {
    const a = this.screenToWorldXY(x0, y0);
    const b = this.screenToWorldXY(x1, y1);
    return {
      x_min: Math.min(a.x, b.x), x_max: Math.max(a.x, b.x),
      y_min: Math.min(a.y, b.y), y_max: Math.max(a.y, b.y),
    };
  }

  /** World XY bounding box of points whose screen projection is inside a rect.
   *  Using the enclosed points (not a ground-plane raycast) makes the box match
   *  what the user framed, free of perspective parallax between canopy and ground. */
  bboxFromScreenRect(x0: number, y0: number, x1: number, y1: number):
    { x_min: number; y_min: number; x_max: number; y_max: number } | null {
    const pos = this.positionsRef();
    if (!pos) return null;
    const rect = this.renderer.domElement.getBoundingClientRect();
    const minx = Math.min(x0, x1), maxx = Math.max(x0, x1);
    const miny = Math.min(y0, y1), maxy = Math.max(y0, y1);
    const v = new THREE.Vector3();
    let xmin = Infinity, ymin = Infinity, xmax = -Infinity, ymax = -Infinity, found = false;
    for (let i = 0; i < this.n; i++) {
      v.set(pos[i * 3], pos[i * 3 + 1], pos[i * 3 + 2]).project(this.activeCamera);
      if (v.z < -1 || v.z > 1) continue;
      const sx = ((v.x + 1) / 2) * rect.width;
      const sy = ((1 - v.y) / 2) * rect.height;
      if (sx >= minx && sx <= maxx && sy >= miny && sy <= maxy) {
        const wx = pos[i * 3], wy = pos[i * 3 + 1];
        if (wx < xmin) xmin = wx; if (wx > xmax) xmax = wx;
        if (wy < ymin) ymin = wy; if (wy > ymax) ymax = wy;
        found = true;
      }
    }
    return found ? { x_min: xmin, y_min: ymin, x_max: xmax, y_max: ymax } : null;
  }

  // ---- screen-space manual selection ------------------------------------
  /** Indices of points whose screen projection falls inside a polygon (px, container-relative). */
  selectInScreenPolygon(poly: [number, number][]): number[] {
    return this.selectByScreen((sx, sy) => pointInPolygon(sx, sy, poly));
  }
  /** Indices of points within `radius` px of a screen center. */
  selectInScreenDisc(cx: number, cy: number, radius: number): number[] {
    const r2 = radius * radius;
    return this.selectByScreen((sx, sy) => {
      const dx = sx - cx, dy = sy - cy;
      return dx * dx + dy * dy <= r2;
    });
  }
  private selectByScreen(test: (sx: number, sy: number) => boolean): number[] {
    const pos = this.positionsRef();
    if (!pos) return [];
    const rect = this.renderer.domElement.getBoundingClientRect();
    const v = new THREE.Vector3();
    const out: number[] = [];
    for (let i = 0; i < this.n; i++) {
      v.set(pos[i * 3], pos[i * 3 + 1], pos[i * 3 + 2]).project(this.activeCamera);
      if (v.z < -1 || v.z > 1) continue;
      const sx = ((v.x + 1) / 2) * rect.width;
      const sy = ((1 - v.y) / 2) * rect.height;
      if (test(sx, sy)) out.push(i);
    }
    return out;
  }

  // ---- click markers -----------------------------------------------------
  addClickMarker(world: THREE.Vector3, positive: boolean) {
    const geo = new THREE.SphereGeometry(this.sceneSize * 0.004, 12, 12);
    const mat = new THREE.MeshBasicMaterial({ color: positive ? 0x35d07f : 0xff5d5d });
    const m = new THREE.Mesh(geo, mat);
    m.position.copy(world);
    this.clickMarkers.add(m);
  }
  clearClickMarkers() {
    this.clickMarkers.clear();
  }

  dispose() {
    window.removeEventListener("resize", this.onResize);
    this.renderer.dispose();
    this.container.removeChild(this.renderer.domElement);
  }
}

function pointInPolygon(x: number, y: number, poly: [number, number][]): boolean {
  let inside = false;
  for (let i = 0, j = poly.length - 1; i < poly.length; j = i++) {
    const xi = poly[i][0], yi = poly[i][1], xj = poly[j][0], yj = poly[j][1];
    if (((yi > y) !== (yj > y)) && x < ((xj - xi) * (y - yi)) / (yj - yi) + xi) inside = !inside;
  }
  return inside;
}

// distinct color per categorical value (golden-ratio hue), matching instance colors.
function catColor(v: number): [number, number, number] {
  return hsv((Math.abs(Math.round(v)) * 0.61803398875) % 1, 0.6, 0.95);
}
function hsv(h: number, s: number, v: number): [number, number, number] {
  const i = Math.floor(h * 6);
  const f = h * 6 - i;
  const p = v * (1 - s), q = v * (1 - f * s), t = v * (1 - (1 - f) * s);
  const m = [[v, t, p], [q, v, p], [p, v, t], [p, q, v], [t, p, v], [v, p, q]][i % 6];
  return [m[0], m[1], m[2]];
}

// standard height ramp: blue(low) → cyan → green → yellow → red(high)
function ramp(t: number): [number, number, number] {
  t = Math.max(0, Math.min(1, t));
  const stops: [number, number, number][] = [
    [0.13, 0.24, 0.86], // blue
    [0.0, 0.78, 0.9],   // cyan
    [0.2, 0.82, 0.25],  // green
    [0.95, 0.9, 0.15],  // yellow
    [0.92, 0.2, 0.15],  // red
  ];
  const seg = t * (stops.length - 1);
  const i = Math.min(stops.length - 2, Math.floor(seg));
  const f = seg - i;
  const a = stops[i], b = stops[i + 1];
  return [a[0] + (b[0] - a[0]) * f, a[1] + (b[1] - a[1]) * f, a[2] + (b[2] - a[2]) * f];
}

const VERT = `
  attribute vec3 pcolor;
  attribute float alpha;
  varying vec3 vColor;
  varying float vAlpha;
  uniform float size;
  void main() {
    vColor = pcolor;
    vAlpha = alpha;
    vec4 mv = modelViewMatrix * vec4(position, 1.0);
    gl_PointSize = size;
    gl_Position = projectionMatrix * mv;
  }
`;
const FRAG = `
  varying vec3 vColor;
  varying float vAlpha;
  void main() {
    if (vAlpha < 0.5) discard;
    gl_FragColor = vec4(vColor, 1.0);
  }
`;
