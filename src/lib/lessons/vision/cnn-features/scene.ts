import { Camera2D } from '../../../gpu/camera.ts';
import { diverging, tone } from '../../../gpu/colors.ts';
import { Lines } from '../../../gpu/primitives/lines.ts';
import { Quads } from '../../../gpu/primitives/quads.ts';
import type { Label, Scene, SceneContext, SceneFactory, ScenePointer, Tone } from '../../../gpu/types.ts';
import { C1, C2, CLASSES, forward, receptiveField, type CnnTrace } from '../../../ml/cnn.ts';
import { strokeSegment } from '../../../ml/raster.ts';
import { fitCamera, lessonInsets, type WorldRect } from '../../fit.ts';
import { cellAt, toPixel, writeGrid, type GridLayout } from '../shared/grid.ts';
import { CLASS_NAMES, SAMPLES } from './data.ts';
import { WEIGHTS } from './weights.ts';

// The layout is budgeted in CSS px at the reference stage (about 118 px per world unit) and converted to world units.
const S = 118;
const px = (n: number) => n / S;

const IMG = 16;
const M1 = 14;
const P1 = 7;
const M2 = 5;

// Column left edges and cell sizes in px (x measured from the stage-area center, y up).
const X1 = -314;
const CELL1 = 10; // drawing: 160 px
const X2 = X1 + 16 * CELL1 + 20; // kernels
const CELL2 = 8; // 24 px
const X3 = X2 + 3 * CELL2 + 24; // feature maps
const CELL3 = 5.5; // 77 px each
const MAP_GAP = 22;
const X4 = X3 + M1 * CELL3 + 24; // pooled maps
const CELL4 = 6; // 42 px each
const X5 = X4 + P1 * CELL4 + 24; // layer 2 maps, two columns
const CELL5 = 8; // 40 px each
const GAP5 = 8;
const GAP5Y = 14;
const X6 = X5 + 2 * M2 * CELL5 + GAP5 + 20; // answer
const NAME_W = 46;
const BAR_W = 46;
const BAR_H = 14;
const BAR_PITCH = 40;

const MAP_PITCH = M1 * CELL3 + MAP_GAP;
const mapCenterY = (i: number) => ((C1 - 1) / 2 - i) * MAP_PITCH;
const CAPTION_Y = (C1 * M1 * CELL3 + (C1 - 1) * MAP_GAP) / 2 + 20;

const clamp = (v: number, a: number, b: number) => Math.min(b, Math.max(a, v));

/** Right edge (px) of the last column that is visible at each `show` level. */
const RIGHT_EDGE = [0, X3 + M1 * CELL3, X4 + P1 * CELL4, X5 + 2 * M2 * CELL5 + GAP5, -X1];
/** World rectangle around the visible columns: early steps zoom in and center instead of leaving half the stage empty. */
function viewRect(show: number): WorldRect {
	const half = px(CAPTION_Y + 22);
	return { x0: px(X1) - 0.02, x1: px(RIGHT_EDGE[clamp(show, 1, 4)]) + 0.02, y0: -half, y1: half };
}

type Layer = 'a1' | 'p1' | 'a2';
interface Hover {
	layer: Layer;
	map: number;
	row: number;
	col: number;
}

export const createCnnFeaturesScene: SceneFactory = () => {
	let ctx: SceneContext;
	let camera: Camera2D;
	let quads: Quads;
	let lines: Lines;

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
	let needsDraw = true;
	let rect: WorldRect | null = null;

	// what is on screen (eased toward the trace)
	const dImg = new Float32Array(IMG * IMG);
	const dA1 = new Float32Array(C1 * M1 * M1);
	const dP1 = new Float32Array(C1 * P1 * P1);
	const dA2 = new Float32Array(C2 * M2 * M2);
	const dProb = new Float32Array(CLASSES);
	const colA = [1, 1, 1, 0, 0, 0]; // per column alpha, columns 1..6

	let lastGuess = '';
	let empty = true;

	const mul = (t: Tone) => (hl === null || hl === t ? 1 : 0.3);

	// Grids
	const gImg: GridLayout = { x: px(X1), y: px(8 * CELL1), cell: px(CELL1), cols: IMG, rows: IMG, gap: 0.06 };
	const gKernel = (k: number): GridLayout => ({
		x: px(X2),
		y: px(mapCenterY(k) + 1.5 * CELL2),
		cell: px(CELL2),
		cols: 3,
		rows: 3,
		gap: 0.08
	});
	const gA1 = (k: number): GridLayout => ({
		x: px(X3),
		y: px(mapCenterY(k) + 7 * CELL3),
		cell: px(CELL3),
		cols: M1,
		rows: M1,
		gap: 0.06
	});
	const gP1 = (k: number): GridLayout => ({
		x: px(X4),
		y: px(mapCenterY(k) + 3.5 * CELL4),
		cell: px(CELL4),
		cols: P1,
		rows: P1,
		gap: 0.07
	});
	const block5H = 4 * M2 * CELL5 + 3 * GAP5Y;
	const gA2 = (k: number): GridLayout => ({
		x: px(X5 + (k % 2) * (M2 * CELL5 + GAP5)),
		y: px(block5H / 2 - Math.floor(k / 2) * (M2 * CELL5 + GAP5Y)),
		cell: px(CELL5),
		cols: M2,
		rows: M2,
		gap: 0.07
	});
	const barY = (c: number) => (1.5 - c) * BAR_PITCH;

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
		needsDraw = true;
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

	function hit(wx: number, wy: number): Hover | null {
		if (show >= 1 && colA[2] > 0.5) {
			for (let k = 0; k < C1; k++) {
				const c = cellAt(gA1(k), wx, wy);
				if (c) return { layer: 'a1', map: k, ...c };
			}
		}
		if (show >= 2 && colA[3] > 0.5) {
			for (let k = 0; k < C1; k++) {
				const c = cellAt(gP1(k), wx, wy);
				if (c) return { layer: 'p1', map: k, ...c };
			}
		}
		if (show >= 3 && colA[4] > 0.5) {
			for (let k = 0; k < C2; k++) {
				const c = cellAt(gA2(k), wx, wy);
				if (c) return { layer: 'a2', map: k, ...c };
			}
		}
		return null;
	}

	function paint(wx: number, wy: number, fresh: boolean) {
		const [x, y] = toPixel(gImg, wx, wy);
		if (fresh) last = [x, y];
		strokeSegment(image, IMG, last[0], last[1], x, y, 0.7);
		last = [x, y];
		netDirty = true;
	}

	function sameHover(a: Hover | null, b: Hover | null) {
		return a === b || (a && b && a.layer === b.layer && a.map === b.map && a.row === b.row && a.col === b.col);
	}

	function drawScene() {
		needsDraw = false;
		const mT = mul('text');
		const mP = mul('param');
		const mD = mul('data');
		let n = 0;

		// 1. drawing
		n = writeGrid(quads, n, dImg, gImg, (v) => tone('text', 0.08 + 0.7 * v, mT * colA[0]));

		// 2. kernels
		for (let k = 0; k < C1; k++) {
			let m = 1e-6;
			for (let i = 0; i < 9; i++) m = Math.max(m, Math.abs(WEIGHTS.w1[k * 9 + i]));
			n = writeGrid(quads, n, WEIGHTS.w1.subarray(k * 9, k * 9 + 9), gKernel(k), (v) => diverging(v / m, 0.85, mP * colA[1]));
		}

		// 3. feature maps (a1), pooled (p1), layer 2 (a2): comparable inside a layer
		const m1 = maxOf(dA1, 4);
		for (let k = 0; k < C1; k++) {
			const vals = dA1.subarray(k * M1 * M1, (k + 1) * M1 * M1);
			n = writeGrid(quads, n, vals, gA1(k), (v) => tone('data', 0.06 + 0.72 * Math.min(1, v / m1), mD * colA[2]));
		}
		const mp = maxOf(dP1, 4);
		for (let k = 0; k < C1; k++) {
			const vals = dP1.subarray(k * P1 * P1, (k + 1) * P1 * P1);
			n = writeGrid(quads, n, vals, gP1(k), (v) => tone('data', 0.06 + 0.72 * Math.min(1, v / mp), mD * colA[3]));
		}
		const m2 = maxOf(dA2, 25);
		for (let k = 0; k < C2; k++) {
			const vals = dA2.subarray(k * M2 * M2, (k + 1) * M2 * M2);
			n = writeGrid(quads, n, vals, gA2(k), (v) => tone('data', 0.06 + 0.72 * Math.min(1, v / m2), mD * colA[4]));
		}

		// 4. answer bars
		let top = 0;
		for (let c = 1; c < CLASSES; c++) if (dProb[c] > dProb[top]) top = c;
		const mO = mul('ok') * (empty ? 0.3 : 1);
		const mB = mD * (empty ? 0.3 : 1);
		for (let c = 0; c < CLASSES; c++) {
			const x0 = px(X6 + NAME_W + 4);
			const y = px(barY(c));
			quads.set(n++, {
				x: x0 + px(BAR_W) / 2,
				y,
				w: px(BAR_W),
				h: px(BAR_H),
				color: tone('muted', 0.5, 0.35 * colA[5] * mD),
				radius: 2
			});
			const w = Math.max(px(1), px(BAR_W) * dProb[c]);
			const isTop = c === top;
			quads.set(n++, {
				x: x0 + w / 2,
				y,
				w,
				h: px(BAR_H),
				color: isTop ? tone('ok', 0.75, colA[5] * mO) : tone('data', 0.4, colA[5] * mB),
				radius: 2
			});
		}
		quads.count = n;
		quads.upload();

		// Outlines and connectors
		let m = 0;
		const rect = (g: GridLayout, r0: number, c0: number, r1: number, c1: number, color: [number, number, number, number], width: number) => {
			const x0 = g.x + c0 * g.cell;
			const x1 = g.x + (c1 + 1) * g.cell;
			const y0 = g.y - r0 * g.cell;
			const y1 = g.y - (r1 + 1) * g.cell;
			m = lines.setPolyline(m, [[x0, y0], [x1, y0], [x1, y1], [x0, y1], [x0, y0]], { width, color });
		};
		const link = (ax: number, ay: number, bx: number, by: number, color: [number, number, number, number]) => {
			lines.set(m++, { a: [px(ax), px(ay)], b: [px(bx), px(by)], width: 1, color });
		};

		rect(gImg, 0, 0, IMG - 1, IMG - 1, tone('muted', 1, 0.5 * mT), 1);
		const c1 = (a: number, k: number) => tone('muted', 1, 0.45 * Math.min(a, 1) * k);
		for (let k = 0; k < C1; k++) {
			const y = mapCenterY(k);
			link(X1 + 16 * CELL1, 0, X2, y, c1(colA[1], 1));
			link(X2 + 3 * CELL2, y, X3, y, c1(colA[2], 1));
			link(X3 + M1 * CELL3, y, X4, y, c1(colA[3], 1));
			link(X4 + P1 * CELL4, y, X5, 0, c1(colA[4], 1));
		}

		if (colA[5] > 0.01) {
			const gmax = Math.max(1e-6, ...Array.from(trace.g));
			for (let k = 0; k < C2; k++) {
				const gv = trace.g[k];
				if (gv < 1e-6) continue;
				const g = gA2(k);
				const idx = trace.gArg[k] - k * M2 * M2;
				const r = Math.floor(idx / M2);
				const c = idx % M2;
				const al = colA[5] * mO;
				rect(g, r, c, r, c, tone('ok', 1, 0.6 * al), 1.2);
				lines.set(m++, {
					a: [g.x + (c + 0.5) * g.cell, g.y - (r + 0.5) * g.cell],
					b: [px(X6 - 4), px(0)],
					width: 1,
					color: tone('ok', 1, (0.08 + 0.42 * (gv / gmax)) * al)
				});
			}
		}

		// Receptive field of the hovered cell, or faintly that of the strongest layer-2 cell
		const mOk = mul('ok');
		if (hover) {
			const g = hover.layer === 'a1' ? gA1(hover.map) : hover.layer === 'p1' ? gP1(hover.map) : gA2(hover.map);
			rect(g, hover.row, hover.col, hover.row, hover.col, tone('text', 1, 0.65), 1.5);
			const f = receptiveField(hover.layer, hover.row, hover.col);
			rect(gImg, f.r0, f.c0, f.r1, f.c1, tone('ok', 1, 0.9 * mOk), 2);
		} else if (show >= 3 && colA[4] > 0.5 && !empty) {
			const k = Math.floor(strongest / (M2 * M2));
			const idx = strongest - k * M2 * M2;
			const f = receptiveField('a2', Math.floor(idx / M2), idx % M2);
			rect(gImg, f.r0, f.c0, f.r1, f.c1, tone('ok', 1, 0.35 * mOk * colA[4]), 2);
		}
		lines.count = m;
		lines.upload();
	}

	function buildLabels(): Label[] {
		const v = ctx.view;
		const out: Label[] = [];
		const at = (x: number, y: number) => v.project(px(x), px(y));
		const cap = (id: string, text: string, cx: number, a: number) => {
			const [x, y] = at(cx, CAPTION_Y);
			out.push({ id, x, y, text, tone: 'muted', opacity: a });
		};
		const mT = mul('text');
		const mP = mul('param');
		const mD = mul('data');
		cap('cap1', 'drawing', X1 + 8 * CELL1, colA[0] * mT);
		cap('cap2', 'kernels', X2 + 1.5 * CELL2, colA[1] * mP);
		cap('cap3', 'feature maps', X3 + 7 * CELL3, colA[2] * mD);
		cap('cap4', 'pooled', X4 + 3.5 * CELL4, colA[3] * mD);
		cap('cap5', 'layer 2 maps', X5 + M2 * CELL5 + GAP5 / 2, colA[4] * mD);
		cap('cap6', 'answer', X6 + (NAME_W + BAR_W + 36) / 2, colA[5] * mul('ok'));
		if (colA[5] > 0.01) {
			for (let c = 0; c < CLASSES; c++) {
				const [nx, ny] = at(X6 + NAME_W, barY(c));
				out.push({ id: `name${c}`, x: nx, y: ny, text: CLASS_NAMES[c], tone: 'muted', align: 'right', size: 11, opacity: colA[5] * mul('ok') * (empty ? 0.3 : 1) });
				const [bx, by] = at(X6 + NAME_W + 4 + BAR_W + 4, barY(c));
				out.push({ id: `pct${c}`, x: bx, y: by, text: `${Math.round(dProb[c] * 100)}%`, tone: 'text', mono: true, align: 'left', size: 11, opacity: colA[5] * mul('ok') * (empty ? 0.3 : 1) });
			}
		}
		if (hover) {
			const { layer, map, row, col } = hover;
			const g = layer === 'a1' ? gA1(map) : layer === 'p1' ? gP1(map) : gA2(map);
			const arr = layer === 'a1' ? dA1 : layer === 'p1' ? dP1 : dA2;
			const side = layer === 'a1' ? M1 : layer === 'p1' ? P1 : M2;
			const value = arr[map * side * side + row * side + col];
			const [x, y] = v.project(g.x + (col + 0.5) * g.cell, g.y - side * g.cell);
			out.push({ id: 'hover', x, y: y + 9, text: value.toFixed(2), tone: 'text', mono: true, size: 11 });
		}
		return out;
	}

	const scene: Scene = {
		init(c) {
			ctx = c;
			camera = new Camera2D();
			quads = new Quads(c, { capacity: 1536 });
			lines = new Lines(c, { capacity: 160, blend: 'alpha' });
			loadSample();
		},
		setParams(p) {
			const prev = { sample, scenario };
			sample = Math.round(p.sample ?? sample);
			show = Math.round(p.show ?? show);
			scenario = Math.round(p.scenario ?? scenario);
			if (!paramsSeen || prev.sample !== sample || prev.scenario !== scenario) loadSample();
			paramsSeen = true;
			needsDraw = true;
		},
		update(fr) {
			const v = ctx.view;
			const target = viewRect(show);
			const kr = ctx.reducedMotion || !rect ? 1 : 1 - Math.exp(-fr.dt / 0.18);
			if (!rect) rect = { ...target };
			rect.x0 += (target.x0 - rect.x0) * kr;
			rect.x1 += (target.x1 - rect.x1) * kr;
			rect.y0 += (target.y0 - rect.y0) * kr;
			rect.y1 += (target.y1 - rect.y1) * kr;
			fitCamera(camera, v, rect, lessonInsets(v));
			v.setViewProj(camera.matrix(v.aspect));

			if (netDirty) runNet();

			const instant = ctx.reducedMotion || snap;
			const k = instant ? 1 : 1 - Math.exp(-fr.dt / 0.04);
			let moving = false;
			moving = ease(dImg, image, k) || moving;
			moving = ease(dA1, trace.a1, k) || moving;
			moving = ease(dP1, trace.p1, k) || moving;
			moving = ease(dA2, trace.a2, k) || moving;
			moving = ease(dProb, trace.probs, k) || moving;
			snap = false;
			const ka = instant ? 1 : 1 - Math.exp(-fr.dt / 0.09);
			const targets = [1, 1, 1, show >= 2 ? 1 : 0, show >= 3 ? 1 : 0, show >= 4 ? 1 : 0];
			for (let i = 0; i < 6; i++) {
				const dv = targets[i] - colA[i];
				if (Math.abs(dv) < 0.002) colA[i] = targets[i];
				else {
					colA[i] += dv * ka;
					moving = true;
				}
			}
			if (moving || needsDraw) drawScene();
			ctx.labels.set(buildLabels());
		},
		draw(pass) {
			quads.draw(pass);
			lines.draw(pass);
		},
		command(id) {
			if (id === 'clear') {
				image.fill(0);
				netDirty = true;
				needsDraw = true;
			}
		},
		highlight(t) {
			hl = t;
			needsDraw = true;
		},
		pointer(e: ScenePointer) {
			if (e.type === 'leave' || e.type === 'up') {
				stroking = false;
				if (e.type === 'leave' && hover) {
					hover = null;
					needsDraw = true;
				}
				return;
			}
			const [wx, wy] = ctx.view.unproject(e.x, e.y);
			if (e.type === 'down') {
				const m = 0.5 * gImg.cell;
				const inside =
					wx >= gImg.x - m && wx <= gImg.x + IMG * gImg.cell + m && wy <= gImg.y + m && wy >= gImg.y - IMG * gImg.cell - m;
				if (inside) {
					stroking = true;
					paint(wx, wy, true);
					if (hover) hover = null;
					needsDraw = true;
				}
				return;
			}
			if (stroking) {
				paint(wx, wy, false);
				needsDraw = true;
				return;
			}
			const h = hit(wx, wy);
			if (!sameHover(h, hover)) {
				hover = h;
				needsDraw = true;
			}
		},
		dispose() {
			quads.destroy();
			lines.destroy();
		}
	};
	return scene;
};
