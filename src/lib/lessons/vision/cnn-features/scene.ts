import { mat4 } from 'wgpu-matrix';
import { OrbitRig, type Pose } from '../../../gpu/camera.ts';
import { diverging, mixColor, sheetFill, tone } from '../../../gpu/colors.ts';
import { Lines } from '../../../gpu/primitives/lines.ts';
import { Shadows } from '../../../gpu/primitives/shadows.ts';
import { Sheets, type RGBA, type SheetSpec, type SheetTexture } from '../../../gpu/primitives/sheets.ts';
import { easeInOut, intersectPlane, projectToFloor, rayFromCss, sub, dot, type Ray, type Vec3 } from '../../../gpu/space.ts';
import type { Label, Scene, SceneContext, SceneFactory, ScenePointer, Tone } from '../../../gpu/types.ts';
import { C1, C2, CLASSES, forward, receptiveField, type CnnTrace } from '../../../ml/cnn.ts';
import { strokeSegment } from '../../../ml/raster.ts';
import { insetAspect, insetMatrix, lessonInsets } from '../../fit.ts';
import { SceneOrbit } from '../../shared/orbit.ts';
import { CLASS_NAMES, SAMPLES } from './data.ts';
import { WEIGHTS } from './weights.ts';

const IMG = 16;
const M1 = 14;
const P1 = 7;
const M2 = 5;

// World layout (y up). Every grid is an upright sheet facing +z; stages run left to right, the maps of a stage are fanned in depth.
const YC = 1.05; // height of every sheet's center
const EPS = 0.012; // lines on a sheet sit this far toward the camera
const CELL_IN = 0.1;
const CELL_K = 0.14;
const CELL_M = 0.07;
const CELL_P = 0.09;
const CELL_2 = 0.09;
const X_IN = 0;
const X_KER = 1.6;
const X_MAP = 3.4;
const X_POOL = 5.4;
const X_L2 = 7.2;
const X_ANS = 9.4;
const DZ = 0.9; // depth step of the four-map fans
const DZ2 = 0.55; // depth step between the rows of layer-2 maps
const L2_DX = 0.31; // half the distance between the two layer-2 columns
const ANS_W = 2;
const ANS_H = 1.2;
// Answer texture: 4 class bands of 4 rows; bars start at column B0 and use BN columns for 100%.
const SW = 60;
const SH = 16;
const B0 = 24;
const BN = 34;

const FLOOR_X: [number, number] = [-2, 12];
const FLOOR_Z: [number, number] = [-3, 3];

const clamp = (v: number, a: number, b: number) => Math.min(b, Math.max(a, v));
const zFan = (k: number, n: number, dz: number) => ((n - 1) / 2 - k) * dz;
const CLEAR: RGBA = [0, 0, 0, 0];

type Layer = 'a1' | 'p1' | 'a2';
type Kind = 'img' | 'ker' | 'a1' | 'p1' | 'a2' | 'ans';
interface Hover {
	layer: Layer;
	map: number;
	row: number;
	col: number;
}

/** One sheet: a grid of cells standing at (cx, YC, z), w wide and h tall. */
interface Panel {
	kind: Kind;
	k: number;
	/** Index into the stage alphas: 0 drawing, 1 kernels, 2 maps, 3 pooled, 4 layer 2, 5 answer. */
	stage: number;
	cx: number;
	z: number;
	w: number;
	h: number;
	cols: number;
	rows: number;
	tex: SheetTexture;
}

const STAGE_OF: Record<Kind, number> = { img: 0, ker: 1, a1: 2, p1: 3, a2: 4, ans: 5 };

export const createCnnFeaturesScene: SceneFactory = () => {
	let ctx: SceneContext;
	let rig: OrbitRig;
	let orbit: SceneOrbit;
	let sheets: Sheets;
	let shadows: Shadows;
	let floor: Lines;
	/** Connection lines that draw in when their stage appears: 0 drawing > kernels > maps, 1 maps > pooled, 2 pooled > layer 2, 3 rings > answer. */
	const groups: Lines[] = [];
	let rf: Lines;
	let vp: Float32Array = new Float32Array(16);

	let imgP: Panel;
	let ansP: Panel;
	const kerP: Panel[] = [];
	const a1P: Panel[] = [];
	const p1P: Panel[] = [];
	const a2P: Panel[] = [];
	let panels: Panel[] = [];

	// params
	let sample = 1;
	let show = 4;
	let scenario = 1;
	let paramsSeen = false;
	let hl: Tone | null = null;

	// drawing and network
	const image = new Float32Array(IMG * IMG);
	let trace: CnnTrace = forward(WEIGHTS, image);
	let netDirty = true;
	let snap = true;
	let strongest = 0;
	let stroking = false;
	let last: [number, number] = [0, 0];
	let hover: Hover | null = null;
	let linesDirty = true;

	// what is on screen (eased toward the trace)
	const dImg = new Float32Array(IMG * IMG);
	const dA1 = new Float32Array(C1 * M1 * M1);
	const dP1 = new Float32Array(C1 * P1 * P1);
	const dA2 = new Float32Array(C2 * M2 * M2);
	const dProb = new Float32Array(CLASSES);
	const colA = [1, 1, 1, 0, 0, 0]; // per stage alpha
	const revT = [0, 0, 0, 0]; // per connection group clock: draw-in runs after a short delay
	const texDirty: Record<Kind, boolean> = { img: true, ker: true, a1: true, p1: true, a2: true, ans: true };

	let poseKey = '';
	let poseAspect = 0;
	let lastGuess = '';
	let empty = true;

	const mul = (t: Tone) => (hl === null || hl === t ? 1 : 0.3);

	// ---- geometry helpers
	const originOf = (p: Panel): Vec3 => [p.cx - p.w / 2, YC - p.h / 2, p.z];
	const leftOf = (p: Panel): Vec3 => [p.cx - p.w / 2, YC, p.z + EPS];
	const rightOf = (p: Panel): Vec3 => [p.cx + p.w / 2, YC, p.z + EPS];
	const cellCenter3 = (p: Panel, r: number, c: number): Vec3 => [
		p.cx - p.w / 2 + ((c + 0.5) * p.w) / p.cols,
		YC + p.h / 2 - ((r + 0.5) * p.h) / p.rows,
		p.z + EPS
	];
	/** Closed outline of cells r0..r1, c0..c1 lying on a sheet's face (TL, TR, BR, BL, TL). */
	function rectPts(p: Panel, r0: number, c0: number, r1: number, c1: number): Vec3[] {
		const cw = p.w / p.cols;
		const ch = p.h / p.rows;
		const x0 = p.cx - p.w / 2 + c0 * cw;
		const x1 = p.cx - p.w / 2 + (c1 + 1) * cw;
		const y0 = YC + p.h / 2 - r0 * ch;
		const y1 = YC + p.h / 2 - (r1 + 1) * ch;
		const z = p.z + EPS;
		return [[x0, y0, z], [x1, y0, z], [x1, y1, z], [x0, y1, z], [x0, y0, z]];
	}
	const panelOf = (h: Hover): Panel => (h.layer === 'a1' ? a1P[h.map] : h.layer === 'p1' ? p1P[h.map] : a2P[h.map]);
	const stageShown = (stage: number) => [true, true, true, show >= 2, show >= 3, show >= 4][stage];

	function loadSample() {
		image.set(SAMPLES[clamp(sample - 1, 0, SAMPLES.length - 1)]);
		netDirty = true;
	}

	function runNet() {
		netDirty = false;
		trace = forward(WEIGHTS, image);
		let best = 0;
		for (let i = 1; i < trace.a2.length; i++) if (trace.a2[i] > trace.a2[best]) best = i;
		strongest = best;
		let total = 0;
		for (const v of image) total += v;
		let top = 0;
		for (let c = 1; c < CLASSES; c++) if (trace.probs[c] > trace.probs[top]) top = c;
		empty = total < 2;
		linesDirty = true;
		texDirty.ans = true;
		const guess = total < 2 ? '—' : `${CLASS_NAMES[top]} ${Math.round(trace.probs[top] * 100)}%`;
		if (guess !== lastGuess) {
			lastGuess = guess;
			ctx.emit('guess', guess);
		}
	}

	/** Move `d` toward `t`. Returns true while it is still moving. */
	function ease(d: Float32Array, t: ArrayLike<number>, k: number): boolean {
		let moving = false;
		for (let i = 0; i < d.length; i++) {
			const dv = t[i] - d[i];
			if (Math.abs(dv) < 2e-4) d[i] = t[i];
			else {
				d[i] += dv * k;
				moving = true;
			}
		}
		return moving;
	}

	/** Largest value of a layer, but never below `floor` (about half a typical peak), so a nearly empty drawing stays dim. */
	function maxOf(a: ArrayLike<number>, floor: number): number {
		let m = floor;
		for (let i = 0; i < a.length; i++) if (a[i] > m) m = a[i];
		return m;
	}

	/** Map activation as ink, normalised to the map's own peak. */
	const ink = (v: number, m: number, mul: number): RGBA => tone('data', v > 0.002 ? 0.12 + 0.88 * Math.min(1, v / m) : 0.03, mul);

	// ---- textures: one texel per cell, row 0 of the grid is the top of the sheet (texel y runs upward)
	function fillGrid(p: Panel, vals: ArrayLike<number>, off: number, color: (v: number) => RGBA) {
		const { cols, rows } = p;
		p.tex.write((x, y) => color(vals[off + (rows - 1 - y) * cols + x]));
	}

	function writeTextures() {
		const mT = mul('text');
		const mP = mul('param');
		const mD = mul('data');
		if (texDirty.img) fillGrid(imgP, dImg, 0, (v) => tone('text', 0.02 + 0.8 * v, mT));
		if (texDirty.ker) {
			for (let k = 0; k < C1; k++) {
				const w = WEIGHTS.w1.subarray(k * 9, k * 9 + 9);
				let m = 1e-6;
				for (let i = 0; i < 9; i++) m = Math.max(m, Math.abs(w[i]));
				fillGrid(kerP[k], w, 0, (v) => diverging(v / m, 0.85, mP));
			}
		}
		if (texDirty.a1) {
			for (let k = 0; k < C1; k++) {
				const m = maxOf(dA1.subarray(k * M1 * M1, (k + 1) * M1 * M1), 1.5);
				fillGrid(a1P[k], dA1, k * M1 * M1, (v) => ink(v, m, mD));
			}
		}
		if (texDirty.p1) {
			for (let k = 0; k < C1; k++) {
				const m = maxOf(dP1.subarray(k * P1 * P1, (k + 1) * P1 * P1), 1.5);
				fillGrid(p1P[k], dP1, k * P1 * P1, (v) => ink(v, m, mD));
			}
		}
		if (texDirty.a2) {
			for (let k = 0; k < C2; k++) {
				const m = maxOf(dA2.subarray(k * M2 * M2, (k + 1) * M2 * M2), 6);
				fillGrid(a2P[k], dA2, k * M2 * M2, (v) => ink(v, m, mD));
			}
		}
		if (texDirty.ans) {
			let top = 0;
			for (let c = 1; c < CLASSES; c++) if (dProb[c] > dProb[top]) top = c;
			const dim = empty ? 0.3 : 1;
			const track = tone('muted', 0.22, mD * dim);
			const barTop = tone('ok', 0.8, mul('ok') * dim);
			const barOther = tone('data', 0.45, mD * dim);
			ansP.tex.write((x, y) => {
				const r = SH - 1 - y;
				const c = r >> 2;
				const within = r & 3;
				if (within === 0 || within === 3 || x < B0 || c >= CLASSES) return CLEAR;
				const frac = clamp(dProb[c] * BN - (x - B0), 0, 1);
				return frac > 0 ? mixColor(track, c === top ? barTop : barOther, frac) : track;
			});
		}
		for (const k of Object.keys(texDirty) as Kind[]) texDirty[k] = false;
	}

	// ---- connection lines
	function buildStatic() {
		let m = 0;
		const g = groups[0];
		const cMuted = tone('muted', 1, 0.45);
		const link = (l: Lines, at: number, a: Vec3, b: Vec3) => l.set(at, { a, b, width: 1, color: cMuted });
		for (let k = 0; k < C1; k++) {
			link(g, m++, rightOf(imgP), leftOf(kerP[k]));
			link(g, m++, rightOf(kerP[k]), leftOf(a1P[k]));
		}
		g.count = m;
		g.upload();

		const g2 = groups[1];
		m = 0;
		for (let k = 0; k < C1; k++) link(g2, m++, rightOf(a1P[k]), leftOf(p1P[k]));
		g2.count = m;
		g2.upload();

		const g3 = groups[2];
		m = 0;
		const entry: Vec3 = [X_L2 - L2_DX - (CELL_2 * M2) / 2 - 0.05, YC, EPS];
		for (let k = 0; k < C1; k++) link(g3, m++, rightOf(p1P[k]), entry);
		g3.count = m;
		g3.upload();

		m = 0;
		for (let i = FLOOR_X[0]; i <= FLOOR_X[1]; i++) {
			floor.set(m++, { a: [i, 0, FLOOR_Z[0]], b: [i, 0, FLOOR_Z[1]], width: 1, color: tone('text', 0.1) });
		}
		for (let j = FLOOR_Z[0]; j <= FLOOR_Z[1]; j++) {
			floor.set(m++, { a: [FLOOR_X[0], 0, j], b: [FLOOR_X[1], 0, j], width: 1, color: tone('text', 0.1) });
		}
		floor.count = m;
		floor.upload();
	}

	/** The rings on the strongest layer-2 cells and their lines to the answer. Rebuilt whenever the network or the dimming changes. */
	function buildAnswerLines() {
		const g = groups[3];
		let m = 0;
		const mO = mul('ok') * (empty ? 0.3 : 1);
		const gmax = Math.max(1e-6, ...Array.from(trace.g));
		const target = leftOf(ansP);
		for (let k = 0; k < C2; k++) {
			const gv = trace.g[k];
			if (gv < 1e-6) continue;
			const p = a2P[k];
			const idx = trace.gArg[k] - k * M2 * M2;
			const r = Math.floor(idx / M2);
			const c = idx % M2;
			m = g.setPolyline(m, rectPts(p, r, c, r, c), { width: 1.2, color: tone('ok', 1, 0.6 * mO) });
			g.set(m++, { a: cellCenter3(p, r, c), b: target, width: 1, color: tone('ok', 1, (0.08 + 0.42 * (gv / gmax)) * mO) });
		}
		g.count = m;
		g.upload();
	}

	/** Receptive field of the hovered cell on the drawing, or faintly that of the strongest layer-2 cell. */
	function buildReceptive() {
		let m = 0;
		const mOk = mul('ok');
		if (hover) {
			const p = panelOf(hover);
			const cell = rectPts(p, hover.row, hover.col, hover.row, hover.col);
			const f = receptiveField(hover.layer, hover.row, hover.col);
			const field = rectPts(imgP, f.r0, f.c0, f.r1, f.c1);
			m = rf.setPolyline(m, cell, { width: 1.5, color: tone('text', 1, 0.65) });
			m = rf.setPolyline(m, field, { width: 2, color: tone('ok', 1, 0.9 * mOk) });
			for (let j = 0; j < 4; j++) rf.set(m++, { a: field[j], b: cell[j], width: 1, color: tone('ok', 1, 0.35 * mOk) });
		} else if (show >= 3 && colA[4] > 0.5 && !empty) {
			const k = Math.floor(strongest / (M2 * M2));
			const idx = strongest - k * M2 * M2;
			const f = receptiveField('a2', Math.floor(idx / M2), idx % M2);
			m = rf.setPolyline(m, rectPts(imgP, f.r0, f.c0, f.r1, f.c1), { width: 2, color: tone('ok', 1, 0.35 * mOk * colA[4]) });
		}
		rf.count = m;
		rf.upload();
	}

	// ---- camera: an orthographic pose per `show`, fitted to the sheets that are visible
	function fitPose(asp: number): Pose {
		const drawing = scenario >= 6;
		const az = drawing ? 0.36 : [0.5, 0.6, 0.65, 0.7][clamp(show, 1, 4) - 1];
		const el = drawing ? 0.42 : 0.5;
		const sa = Math.sin(az);
		const ca = Math.cos(az);
		const se = Math.sin(el);
		const ce = Math.cos(el);
		let r0 = Infinity;
		let r1 = -Infinity;
		let u0 = Infinity;
		let u1 = -Infinity;
		for (const p of panels) {
			if (!stageShown(p.stage)) continue;
			for (const dx of [-0.5, 0.5]) {
				for (const dy of [-0.5, 0.5]) {
					const sheet: Vec3 = [p.cx + dx * p.w, YC + dy * p.h, p.z];
					for (const q of [sheet, projectToFloor(sheet, 0)]) {
						const r = q[0] * ca - q[2] * sa;
						const u = -q[0] * sa * se + q[1] * ce - q[2] * ca * se;
						r0 = Math.min(r0, r);
						r1 = Math.max(r1, r);
						u0 = Math.min(u0, u);
						u1 = Math.max(u1, u);
					}
				}
			}
		}
		// room for the captions above and the shadows below
		r0 -= 0.1;
		r1 += 0.1;
		u0 -= 0.1;
		u1 += 0.45;
		const cr = (r0 + r1) / 2;
		const cu = (u0 + u1) / 2;
		const target: Vec3 = [cr * ca - cu * sa * se, cu * ce, -cr * sa - cu * ca * se];
		return { target, azimuth: az, elevation: el, distance: Math.max(u1 - u0, (r1 - r0) / Math.max(0.2, asp)), ortho: true };
	}

	function updatePose(asp: number) {
		const key = `${clamp(show, 1, 4)}|${scenario >= 6 ? 1 : 0}`;
		if (key !== poseKey) {
			poseKey = key;
			poseAspect = asp;
			orbit.stepPose(fitPose(asp));
		} else if (!rig.moved && Math.abs(asp / poseAspect - 1) > 0.04) {
			// the stage area changed shape (resize, card folded): refit
			poseAspect = asp;
			orbit.stepPose(fitPose(asp));
		}
	}

	// ---- picking
	const rayAt = (e: { x: number; y: number }): Ray => rayFromCss(vp, ctx.view.width, ctx.view.height, e.x, e.y);

	/** Continuous cell coordinates (col, row) where a ray meets a sheet's plane, and the distance along the ray. */
	function planeHit(p: Panel, ray: Ray): { fx: number; fy: number; t: number } | null {
		const hit = intersectPlane(ray, [0, 0, p.z], [0, 0, 1]);
		if (!hit) return null;
		return {
			fx: ((hit[0] - (p.cx - p.w / 2)) / p.w) * p.cols,
			fy: ((YC + p.h / 2 - hit[1]) / p.h) * p.rows,
			t: dot(sub(hit, ray.origin), ray.dir)
		};
	}

	/** The front-most visible map cell under the pointer. */
	function hit(ray: Ray): Hover | null {
		let best: Hover | null = null;
		let bestT = Infinity;
		const scan = (layer: Layer, list: Panel[], stage: number) => {
			if (colA[stage] <= 0.5 || !stageShown(stage)) return;
			list.forEach((p, map) => {
				const h = planeHit(p, ray);
				if (!h || h.fx < 0 || h.fx >= p.cols || h.fy < 0 || h.fy >= p.rows || h.t >= bestT) return;
				bestT = h.t;
				best = { layer, map, row: Math.floor(h.fy), col: Math.floor(h.fx) };
			});
		};
		scan('a1', a1P, 2);
		scan('p1', p1P, 3);
		scan('a2', a2P, 4);
		return best;
	}

	function paint(fx: number, fy: number, fresh: boolean) {
		if (fresh) last = [fx, fy];
		strokeSegment(image, IMG, last[0], last[1], fx, fy, 0.7);
		last = [fx, fy];
		netDirty = true;
	}

	function sameHover(a: Hover | null, b: Hover | null) {
		return a === b || (a && b && a.layer === b.layer && a.map === b.map && a.row === b.row && a.col === b.col);
	}

	function buildLabels(): Label[] {
		const v = ctx.view;
		const out: Label[] = [];
		const cap = (id: string, text: string, ps: Panel[], a: number) => {
			if (a < 0.02) return;
			let top = Infinity;
			let sx = 0;
			for (const p of ps) {
				const [x, y] = v.project(p.cx, YC + p.h / 2, p.z);
				sx += x;
				top = Math.min(top, y);
			}
			out.push({ id, x: sx / ps.length, y: top - 16, text, tone: 'muted', opacity: a });
		};
		cap('cap1', 'drawing', [imgP], colA[0] * mul('text'));
		cap('cap2', 'kernels', kerP, colA[1] * mul('param'));
		cap('cap3', 'feature maps', a1P, colA[2] * mul('data'));
		cap('cap4', 'pooled', p1P, colA[3] * mul('data'));
		cap('cap5', 'layer 2 maps', a2P, colA[4] * mul('data'));
		cap('cap6', 'answer', [ansP], colA[5] * mul('ok'));
		if (colA[5] > 0.02) {
			const op = colA[5] * mul('ok') * (empty ? 0.3 : 1);
			for (let c = 0; c < CLASSES; c++) {
				const y = YC + ANS_H / 2 - ((c + 0.5) * ANS_H) / 4;
				const [nx, ny] = v.project(X_ANS - ANS_W / 2 + 0.07, y, ansP.z);
				out.push({ id: `name${c}`, x: nx, y: ny, text: CLASS_NAMES[c], tone: 'muted', align: 'left', size: 11, opacity: op });
				const [bx, by] = v.project(X_ANS + ANS_W / 2 + 0.07, y, ansP.z);
				out.push({ id: `pct${c}`, x: bx, y: by, text: `${Math.round(dProb[c] * 100)}%`, tone: 'text', mono: true, align: 'left', size: 11, opacity: op });
			}
		}
		if (hover) {
			const { layer, map, row, col } = hover;
			const p = panelOf(hover);
			const arr = layer === 'a1' ? dA1 : layer === 'p1' ? dP1 : dA2;
			const side = layer === 'a1' ? M1 : layer === 'p1' ? P1 : M2;
			const value = arr[map * side * side + row * side + col];
			const c = cellCenter3(p, row, col);
			const [x, y] = v.project(c[0], c[1] + p.h / p.rows / 2, c[2]);
			out.push({ id: 'hover', x, y: y - 10, text: value.toFixed(2), tone: 'text', mono: true, size: 11 });
		}
		return out;
	}

	const scene: Scene = {
		init(c) {
			ctx = c;
			rig = new OrbitRig();
			orbit = new SceneOrbit(c, rig);
			sheets = new Sheets(c, { capacity: 22 });
			shadows = new Shadows(c, { capacity: 22 });
			floor = new Lines(c, { capacity: 32, depthTest: true });
			groups.push(new Lines(c, { capacity: 8, depthTest: true }));
			groups.push(new Lines(c, { capacity: 4, depthTest: true }));
			groups.push(new Lines(c, { capacity: 4, depthTest: true }));
			groups.push(new Lines(c, { capacity: 48, depthTest: true }));
			rf = new Lines(c, { capacity: 24, depthTest: true });

			const add = (kind: Kind, k: number, cx: number, z: number, cell: number, cols: number, rows: number, w = cell * cols, h = cell * rows) => {
				const p: Panel = { kind, k, stage: STAGE_OF[kind], cx, z, w, h, cols, rows, tex: sheets.createTexture(cols, rows) };
				panels.push(p);
				return p;
			};
			imgP = add('img', 0, X_IN, 0, CELL_IN, IMG, IMG);
			for (let k = 0; k < C1; k++) kerP.push(add('ker', k, X_KER, zFan(k, C1, DZ), CELL_K, 3, 3));
			for (let k = 0; k < C1; k++) a1P.push(add('a1', k, X_MAP, zFan(k, C1, DZ), CELL_M, M1, M1));
			for (let k = 0; k < C1; k++) p1P.push(add('p1', k, X_POOL, zFan(k, C1, DZ), CELL_P, P1, P1));
			for (let k = 0; k < C2; k++) {
				const row = Math.floor(k / 2);
				a2P.push(add('a2', k, X_L2 + ((k % 2) - 0.5) * 2 * L2_DX, zFan(row, 4, DZ2), CELL_2, M2, M2));
			}
			ansP = add('ans', 0, X_ANS, 0, 0, SW, SH, ANS_W, ANS_H);
			buildStatic();
			loadSample();
		},
		setParams(p) {
			const prev = { sample, scenario };
			sample = Math.round(p.sample ?? sample);
			show = Math.round(p.show ?? show);
			scenario = Math.round(p.scenario ?? scenario);
			if (!paramsSeen || prev.sample !== sample || prev.scenario !== scenario) loadSample();
			paramsSeen = true;
		},
		update(fr) {
			const v = ctx.view;
			if (netDirty) runNet();

			const instant = ctx.reducedMotion || snap;
			const k = instant ? 1 : 1 - Math.exp(-fr.dt / 0.04);
			if (ease(dImg, image, k)) texDirty.img = true;
			if (ease(dA1, trace.a1, k)) texDirty.a1 = true;
			if (ease(dP1, trace.p1, k)) texDirty.p1 = true;
			if (ease(dA2, trace.a2, k)) texDirty.a2 = true;
			if (ease(dProb, trace.probs, k)) texDirty.ans = true;

			// stage fade, and the clocks that draw each stage's connection lines in
			const ka = instant ? 1 : 1 - Math.exp(-fr.dt / 0.09);
			const targets = [1, 1, 1, show >= 2 ? 1 : 0, show >= 3 ? 1 : 0, show >= 4 ? 1 : 0];
			for (let i = 0; i < 6; i++) {
				const dv = targets[i] - colA[i];
				if (Math.abs(dv) < 0.002) {
					if (dv !== 0) {
						colA[i] = targets[i];
						linesDirty = true;
					}
				} else {
					colA[i] += dv * ka;
					linesDirty = true;
				}
			}
			const DELAY = 0.25;
			const DRAW = 0.9;
			for (let i = 0; i < 4; i++) {
				const t = targets[i + 2];
				if (instant) revT[i] = t * (DELAY + DRAW);
				else revT[i] = clamp(revT[i] + (t ? fr.dt : -fr.dt * 3), 0, DELAY + DRAW);
				groups[i].setReveal(easeInOut((revT[i] - DELAY) / DRAW));
			}
			snap = false;

			// camera
			const ins = lessonInsets(v);
			const asp = insetAspect(v, ins);
			updatePose(asp);
			orbit.update(fr.dt);
			vp = mat4.multiply(insetMatrix(v, ins), rig.matrix(asp)) as Float32Array;
			v.setViewProj(vp);

			if (texDirty.img || texDirty.ker || texDirty.a1 || texDirty.p1 || texDirty.a2 || texDirty.ans) writeTextures();
			if (linesDirty) {
				linesDirty = false;
				buildAnswerLines();
				buildReceptive();
			}

			// shadows, then the sheets back to front
			const eye = rig.eye();
			const live = panels.filter((p) => colA[p.stage] > 0.01);
			let n = 0;
			for (const p of live) {
				const o = originOf(p);
				const corners = [o, [o[0] + p.w, o[1], o[2]], [o[0] + p.w, o[1] + p.h, o[2]], [o[0], o[1] + p.h, o[2]]].map((q) => projectToFloor(q as Vec3, 0));
				shadows.quad(n++, { corners: corners as [Vec3, Vec3, Vec3, Vec3], blur: 0.1, alpha: colA[p.stage] });
			}
			shadows.count = n;
			const depth = (p: Panel) => Math.hypot(p.cx - eye[0], YC - eye[1], p.z - eye[2]);
			live.sort((a, b) => depth(b) - depth(a));
			live.forEach((p, i) => {
				const spec: SheetSpec = {
					origin: originOf(p),
					u: [p.w, 0, 0],
					v: [0, p.h, 0],
					fill: sheetFill(),
					edge: tone('muted', 0.6),
					edgePx: 1,
					texture: p.tex,
					opacity: colA[p.stage]
				};
				sheets.set(i, spec);
			});
			sheets.count = live.length;

			ctx.labels.set(buildLabels());
		},
		draw(pass) {
			floor.draw(pass);
			shadows.draw(pass);
			sheets.draw(pass);
			for (const g of groups) g.draw(pass);
			rf.draw(pass);
		},
		command(id) {
			if (orbit.command(id)) return;
			if (id === 'clear') {
				image.fill(0);
				netDirty = true;
			}
		},
		highlight(t) {
			hl = t;
			for (const k of Object.keys(texDirty) as Kind[]) texDirty[k] = true;
			linesDirty = true;
		},
		pointer(e: ScenePointer) {
			if (e.type === 'leave' || e.type === 'up') {
				stroking = false;
				orbit.pointer(e);
				if (e.type === 'leave' && hover) {
					hover = null;
					linesDirty = true;
				}
				return;
			}
			const ray = rayAt(e);
			if (e.type === 'down') {
				const h = planeHit(imgP, ray);
				const m = 0.5;
				if (h && h.fx >= -m && h.fx <= IMG + m && h.fy >= -m && h.fy <= IMG + m) {
					stroking = true;
					paint(h.fx, h.fy, true);
					if (hover) {
						hover = null;
						linesDirty = true;
					}
				} else {
					orbit.pointer(e);
				}
				return;
			}
			if (stroking) {
				const h = planeHit(imgP, ray);
				if (h) paint(h.fx, h.fy, false);
				return;
			}
			if (orbit.pointer(e) === 'drag') {
				if (hover) {
					hover = null;
					linesDirty = true;
				}
				return;
			}
			const h = hit(ray);
			if (!sameHover(h, hover)) {
				hover = h;
				linesDirty = true;
			}
		},
		dispose() {
			sheets.dispose();
			shadows.dispose();
			floor.destroy();
			for (const g of groups) g.destroy();
			rf.destroy();
		}
	};
	return scene;
};
