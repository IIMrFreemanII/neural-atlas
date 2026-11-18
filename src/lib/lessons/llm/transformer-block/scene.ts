import { Camera2D } from '../../../gpu/camera.ts';
import { diverging, paperFill, tone } from '../../../gpu/colors.ts';
import { bezier, Lines } from '../../../gpu/primitives/lines.ts';
import { Points } from '../../../gpu/primitives/points.ts';
import { Quads } from '../../../gpu/primitives/quads.ts';
import type { Label, Scene, SceneContext, SceneFactory, Tone } from '../../../gpu/types.ts';
import { fitCamera, lessonInsets, type WorldRect } from '../../fit.ts';
import { pxPerWorld } from '../shared/chips.ts';
import { loadLm } from '../shared/lm.ts';
import { writeGrid } from '../shared/strip.ts';
import { buildBlockModel, TEXT, type BlockModel } from './model.ts';

// World layout, budgeted for a stage area of about 632 x 514 px (one world unit is about 121 px).
const WORLD: WorldRect = { x0: -2.6, x1: 2.6, y0: -2.1, y1: 2.1 };
// Lanes are shifted right so the left edge of every layer box has a clear column for the layer name.
const LANES = [-1.2, -0.45, 0.3, 1.05, 1.8];
const X_MID = 0.3; // horizontal center of the lanes
const N = LANES.length;

const Y_CHIP = -1.94;
const Y_IN = -1.57;
const Y_OUT = 1.54;
const TILE = 0.48; // side of the 8 x 8 tile that draws one 64-number vector
const GRID_GAP = 0.1; // gap between cells, as a share of the cell
const Y_BLOCK0 = -1.25;
const Y_BLOCK1 = 1.25;
const X_BLOCK = 2.4;
const X_LAYER = 2.3;
const X_NAME = -2.23;
const Y_ADD1 = -0.2;
const Y_ADD2 = 0.9;
const BYPASS_DX = 0.14;
const ARC_Y = -0.52;

type LayerKey = 'norm1' | 'attn' | 'norm2' | 'ff';
interface Layer {
	key: LayerKey;
	y: number;
	h: number;
	label: string;
	labelY: number;
	tone: Tone;
}
const LAYERS: Layer[] = [
	{ key: 'norm1', y: -1.02, h: 0.16, label: 'Normalize', labelY: -1.02, tone: 'text' },
	{ key: 'attn', y: -0.58, h: 0.36, label: 'Self-attention', labelY: -0.58, tone: 'param' },
	{ key: 'norm2', y: 0.1, h: 0.16, label: 'Normalize', labelY: 0.1, tone: 'text' },
	{ key: 'ff', y: 0.52, h: 0.36, label: 'Feed-forward', labelY: 0.52, tone: 'param' }
];

// Stack view
const STACK_Y0 = -1.25;
const STACK_STEP = 0.15;
const STACK_MAX_STEP = 0.34;
const STACK_SPAN = 1.8;
const BAR_BASE = 1.35;
const BAR_PITCH = 0.46;
const BAR_X0 = -1.95;
const BAR_W = 0.25;
const BAR_H_MAX = 0.46; // world height of the tallest bar the slider can show

const DIM = 0.35;
const DIM_STREAM = 0.5;

const ORDINALS = ['1st', '2nd', '3rd', '4th', '5th'];

const clamp = (v: number, a: number, b: number) => Math.min(b, Math.max(a, v));
const smooth = (t: number) => t * t * (3 - 2 * t);
/** A share as a whole percent; anything above zero but under half a percent reads as '<1%'. */
const pct = (p: number) => (p > 0 && p < 0.005 ? '<1%' : `${Math.round(p * 100)}%`);

type Rect = { x0: number; x1: number; y0: number; y1: number };

/** Draw a vector as a square-ish tile `size` world units wide: `cols` cells across, row-major, first row on top. */
function tile(q: Quads, start: number, vals: ArrayLike<number>, cols: number, x: number, y: number, size: number, maxAbs: number, alpha: number): number {
	const cell = size / (cols + GRID_GAP * (cols - 1));
	return writeGrid(q, start, vals, cols, { x, y, cell, gap: cell * GRID_GAP, maxAbs, alpha, intensity: 0.85, radius: 1 });
}

export const createTransformerBlockScene: SceneFactory = () => {
	let ctx: SceneContext;
	let model: BlockModel;
	let updMax = 1; // color scale for the updates and normalized vectors, from the data
	let barScale = 1;
	let camera: Camera2D;
	let boxes: Quads; // selection band, chips, layer boxes, stack blocks
	let lines: Lines; // lanes, outline, bypass, arcs
	let dots: Points; // add nodes, attention dots
	let glow: Points; // play markers (additive)
	let strips: Quads; // vectors on the diagram, bars
	let backdrop: Quads; // inspector backdrop
	let insp: Quads; // inspector strips and bars
	let ilines: Lines; // inspector lines
	let marks: Lines; // plus signs in the add nodes
	let ppw = 121;

	// params
	let focus = 1;
	let depth = 1;
	let zoom = 0;
	let column = 4;
	let scenario = 1;
	let paramsSeen = false;
	let hl: Tone | null = null;

	// eased state
	const B = { stream: 1, norm1: 1, attn: 1, add1: 1, norm2: 1, ff: 1, add2: 1 };
	type BKey = keyof typeof B;
	let barA = 0;
	const barVals: number[] = [];
	let barsSeen = false;
	let inspKind = 0; // 3, 4, 5 or 0
	let inspA = 0;

	// play
	let playing = false;
	let playT = 0;
	let playY = -9;
	const PLAY_SECONDS = 2.4;
	const PLAY_Y0 = -1.82; // just above the chips so their text stays readable

	// bookkeeping
	let dirty = true;
	let lastW = 0;
	let lastH = 0;
	let labelKey = '';
	let lastInspect = '';
	let frameNo = 0;
	let stripY = 0;

	const mul = (t: Tone) => (hl === null || hl === t ? 1 : 0.3);
	const neutral = () => (hl === null ? 1 : 0.3);

	function targets(): Record<BKey, number> {
		const t: Record<BKey, number> = { stream: 1, norm1: 1, attn: 1, add1: 1, norm2: 1, ff: 1, add2: 1 };
		if (focus >= 2 && focus <= 5) {
			t.stream = DIM_STREAM;
			t.norm1 = t.attn = t.add1 = t.norm2 = t.ff = t.add2 = DIM;
			if (focus === 2) t.attn = 1;
			if (focus === 3) t.add1 = 1;
			if (focus === 4) t.ff = 1;
			if (focus === 5) t.norm1 = t.norm2 = 1;
		}
		return t;
	}

	function stackStep(d: number): number {
		return clamp(STACK_SPAN / d, STACK_STEP, STACK_MAX_STEP);
	}
	/** Where the output strips want to sit: just above the top of the stack. */
	function stripTarget(d: number): number {
		return STACK_Y0 + stackStep(d) * d + 0.08 + TILE / 2;
	}

	function flash(ya: number, yb: number): number {
		if (!playing) return 0;
		if (playY < ya) return 0;
		if (playY <= yb) return 1;
		return Math.max(0, 1 - (playY - yb) / 0.5);
	}

	function rrPts(cx: number, cy: number, w: number, h: number, r: number): Array<[number, number]> {
		const pts: Array<[number, number]> = [];
		const corners: Array<[number, number, number]> = [
			[cx + w / 2 - r, cy + h / 2 - r, 0],
			[cx - w / 2 + r, cy + h / 2 - r, Math.PI / 2],
			[cx - w / 2 + r, cy - h / 2 + r, Math.PI],
			[cx + w / 2 - r, cy - h / 2 + r, 1.5 * Math.PI]
		];
		for (const [ox, oy, a0] of corners) {
			for (let i = 0; i <= 5; i++) {
				const a = a0 + (i / 5) * (Math.PI / 2);
				pts.push([ox + r * Math.cos(a), oy + r * Math.sin(a)]);
			}
		}
		pts.push(pts[0]);
		return pts;
	}

	function pickLane(x: number, y: number): number {
		const [wx] = ctx.view.unproject(x, y);
		let best = -1;
		let bd = 0.42;
		for (let i = 0; i < N; i++) {
			const d = Math.abs(wx - LANES[i]);
			if (d < bd) {
				bd = d;
				best = i;
			}
		}
		return best;
	}

	/** Lane intervals between yFrom and yTo with the given cut ranges removed. */
	function laneIntervals(yFrom: number, yTo: number, cuts: Array<[number, number]>): Array<[number, number]> {
		const sorted = [...cuts].sort((a, b) => a[0] - b[0]);
		const out: Array<[number, number]> = [];
		let cur = yFrom;
		for (const [a, b] of sorted) {
			if (a > cur) out.push([cur, Math.min(a, yTo)]);
			cur = Math.max(cur, b);
		}
		if (cur < yTo) out.push([cur, yTo]);
		return out.filter(([a, b]) => b - a > 1e-4);
	}

	function rebuild() {
		const v = ctx.view;
		const px = (n: number) => n / ppw;
		const dA = 1 - smooth(clamp(zoom, 0, 1));
		const sA = smooth(clamp(zoom, 0, 1));
		const stack = sA > 0.02;
		const detail = dA > 0.02;
		const nm = neutral();
		const mData = mul('data');
		const mParam = mul('param');
		const mOk = mul('ok');

		let nB = 0;
		let nL = 0;
		let nD = 0;
		let nG = 0;
		let nS = 0;
		let nBd = 0;
		let nI = 0;
		let nIL = 0;
		let nM = 0;
		const labels: Label[] = [];

		// inspector rectangle (world), set below when active
		let mask: Rect | null = null;
		const inspC = inspKind === 3 ? { cx: 0, cy: 0.24, w: 4.6, h: 1.08 } : inspKind === 4 ? { cx: 0, cy: -0.43, w: 4.6, h: 1.38 } : { cx: 0, cy: -0.47, w: 4.6, h: 0.9 };
		const inspOn = inspKind !== 0 && inspA > 0.01 && detail;
		if (inspOn) {
			mask = { x0: inspC.cx - inspC.w / 2, x1: inspC.cx + inspC.w / 2, y0: inspC.cy - inspC.h / 2, y1: inspC.cy + inspC.h / 2 };
		}

		const lab = (id: string, wx: number, wy: number, text: string, o: Partial<Label> = {}) => {
			const [x, y] = v.project(wx, wy);
			let op = o.opacity ?? 1;
			if (mask && wx > mask.x0 && wx < mask.x1 && wy > mask.y0 && wy < mask.y1) op *= 1 - inspA;
			if (op < 0.03) return;
			labels.push({ ...o, id, x, y, text, opacity: op });
		};
		const labI = (id: string, wx: number, wy: number, text: string, o: Partial<Label> = {}) => {
			const [x, y] = v.project(wx, wy);
			const op = (o.opacity ?? 1) * inspA;
			if (op < 0.03) return;
			labels.push({ ...o, id, x, y, text, opacity: op });
		};

		const sel = column;
		const selX = LANES[sel];

		// ---- selection band
		const bandA = Math.max(dA, sA);
		{
			const y0 = -2.04;
			const topStack = focus === 7 ? BAR_BASE : stripY + TILE / 2 + 0.07;
			const top = (Y_OUT + TILE / 2 + 0.07) * dA + topStack * sA;
			boxes.set(nB++, { x: selX, y: (y0 + top) / 2, w: 0.62, h: top - y0, color: tone('data', 0.05, 0.5 * bandA * nm), radius: 8 });
		}

		// ---- chips (both views)
		const chipW = px(3 * 13 * 0.6 + 14);
		const chipH = px(25);
		for (let i = 0; i < N; i++) {
			boxes.set(nB++, { x: LANES[i], y: Y_CHIP, w: chipW, h: chipH, color: tone('text', 0.2, 0.9 * B.stream * nm), radius: 5 });
			lab(`chip${i}`, LANES[i], Y_CHIP, model.tokens[i], { mono: true, size: 13, opacity: B.stream * nm });
		}

		// ---- input strips (both views)
		for (let i = 0; i < N; i++) {
			nS = tile(strips, nS, model.layers[0].x[i], 8, LANES[i], Y_IN, TILE, model.maxStream, B.stream * mData);
		}

		// ---- lanes
		{
			const cutsIn: Array<[number, number]> = [[Y_IN - TILE / 2, Y_IN + TILE / 2]];
			const detailCuts: Array<[number, number]> = [
				...cutsIn,
				[Y_OUT - TILE / 2, Y_OUT + TILE / 2],
				...LAYERS.map((l): [number, number] => [l.y - l.h / 2, l.y + l.h / 2])
			];
			const pulse = ctx.reducedMotion ? 0 : 3;
			const yTop = Y_OUT - TILE / 2;
			const yBot = Y_IN - TILE / 2;
			const range = yTop - yBot;
			for (let i = 0; i < N; i++) {
				const isSel = i === sel;
				const baseA = (isSel ? 0.95 : 0.55) * mData;
				const col = tone('data', 0.7, baseA * B.stream);
				if (detail) {
					for (const [a, b] of laneIntervals(yBot, yTop, detailCuts)) {
						lines.set(nL++, {
							a: [LANES[i], a],
							b: [LANES[i], b],
							width: 1.5,
							color: [col[0], col[1], col[2], col[3] * dA],
							t0: (a - yBot) / range,
							t1: (b - yBot) / range,
							pulseSpeed: 0.12,
							pulseCount: pulse,
							pulseGain: 0.5
						});
					}
				}
				if (stack) {
					const sTop = stripY - TILE / 2;
					const stackTop = i === N - 1 && focus === 7 ? BAR_BASE : sTop;
					const cuts: Array<[number, number]> = [...cutsIn, [stripY - TILE / 2, stripY + TILE / 2]];
					const r2 = stackTop - yBot;
					for (const [a, b] of laneIntervals(yBot, stackTop, cuts)) {
						lines.set(nL++, {
							a: [LANES[i], a],
							b: [LANES[i], b],
							width: 1.5,
							color: [col[0], col[1], col[2], col[3] * sA],
							t0: (a - yBot) / r2,
							t1: (b - yBot) / r2,
							pulseSpeed: 0.12,
							pulseCount: pulse,
							pulseGain: 0.5
						});
					}
				}
			}
		}

		// ---- detail view
		if (detail) {
			// block outline
			nL = lines.setPolyline(nL, rrPts(0, 0, 2 * X_BLOCK, Y_BLOCK1 - Y_BLOCK0, 0.14), {
				width: 1.2,
				color: tone('muted', 1, 0.9 * dA * nm)
			});
			lab('cap-block', -X_BLOCK + 0.02, Y_BLOCK1 + 0.12, 'one block', { tone: 'muted', size: 11, align: 'left', opacity: dA * nm });
			lab('cap-next', X_MID, 1.93, 'to the next block', { tone: 'muted', size: 11, opacity: dA * nm });

			// layer boxes
			for (const l of LAYERS) {
				const bright = B[l.key];
				const f = flash(l.y - l.h / 2 - 0.05, l.y + l.h / 2 + 0.05);
				const base = l.tone === 'param' ? 0.16 : 0.1;
				boxes.set(nB++, {
					x: 0,
					y: l.y,
					w: 2 * X_LAYER,
					h: l.h,
					color: tone(l.tone, (base + 0.4 * f) * 1, (0.9 * bright * mParam) * dA),
					radius: 6
				});
				lab(`name-${l.key}`, X_NAME, l.labelY, l.label, { size: 13, align: 'left', opacity: (0.4 + 0.6 * bright) * dA * nm * (0.5 + 0.5 * mParam) });
			}

			// attention arcs for the selected lane
			const w = model.attention[0][sel];
			// arcs lie under the inspector in steps 4 and 5; fade them out so nothing bleeds through the backdrop
			const bAttn = B.attn * mData * dA * (inspKind === 4 || inspKind === 5 ? 1 - inspA : 1);
			for (let j = 0; j <= sel; j++) {
				const wt = w[j];
				if (j === sel) {
					dots.set(nD++, { x: LANES[j], y: ARC_Y, radius: 3 + 4 * wt, color: tone('data', 1.2, 0.95 * bAttn) });
					continue;
				}
				if (wt < 0.015) continue;
				dots.set(nD++, { x: LANES[j], y: ARC_Y, radius: 2.5 + 4 * wt, color: tone('data', 1.2, (0.4 + 0.55 * wt) * bAttn) });
				const dist = sel - j;
				const peak = 0.04 + 0.03 * (dist - 1);
				const ch = peak / 0.75;
				const pts = bezier([LANES[sel], ARC_Y], [LANES[sel], ARC_Y + ch], [LANES[j], ARC_Y + ch], [LANES[j], ARC_Y], 14);
				nL = lines.setPolyline(nL, pts, { width: 1 + 5 * wt, color: tone('data', 1, (0.2 + 0.7 * wt) * bAttn) });
			}
			if (focus === 2) {
				for (let j = 0; j < N; j++) {
					const t = j <= sel ? `${Math.round(w[j] * 100)}%` : '–';
					lab(`pct${j}`, LANES[j] - 0.06, -0.85, t, { mono: true, size: 11, align: 'right', tone: j <= sel ? 'data' : 'muted', opacity: dA });
				}
			}

			// add nodes and bypass lines
			const adds: Array<{ y: number; k: 'add1' | 'add2'; from: number }> = [
				{ y: Y_ADD1, k: 'add1', from: -1.17 },
				{ y: Y_ADD2, k: 'add2', from: -0.08 }
			];
			for (const a of adds) {
				const under = a.k === 'add1' && (inspKind === 4 || inspKind === 5) ? 1 - inspA : 1;
				const bright = B[a.k] * under;
				const f = flash(a.y - 0.08, a.y + 0.08);
				for (let i = 0; i < N; i++) {
					const lx = LANES[i];
					nL = lines.setPolyline(
						nL,
						[
							[lx, a.from],
							[lx + BYPASS_DX, a.from],
							[lx + BYPASS_DX, a.y],
							[lx + px(8.5), a.y]
						],
						{ width: 1.2, color: tone('data', 0.6, 0.6 * bright * mData * dA) }
					);
					dots.set(nD++, { x: lx, y: a.y, radius: 8, color: paperFill(dA * under) });
					dots.set(nD++, { x: lx, y: a.y, radius: 8, ring: 1.2, color: tone('text', 0.7 + 0.8 * f, 0.9 * bright * nm * dA) });
					const pc = tone('text', 0.8, 0.85 * (0.4 + 0.6 * bright) * nm * dA);
						const arm = px(4.5);
						marks.set(nM++, { a: [lx - arm, a.y], b: [lx + arm, a.y], width: 1.5, color: pc });
						marks.set(nM++, { a: [lx, a.y - arm], b: [lx, a.y + arm], width: 1.5, color: pc });
				}
			}

			// output strips
			for (let i = 0; i < N; i++) {
				nS = tile(strips, nS, model.layers[0].out[i], 8, LANES[i], Y_OUT, TILE, model.maxStream, B.stream * mData * dA);
			}
		}

		// ---- stack view
		if (stack) {
			const d = depth;
			// few blocks are drawn taller so the stack fills its space; twelve blocks use the minimum height
			const step = stackStep(d);
			for (let i = 0; i < d; i++) {
				const y = STACK_Y0 + (i + 0.5) * step;
				const f = flash(y - step / 2, y + step / 2);
				boxes.set(nB++, { x: 0, y, w: 2 * X_LAYER, h: step - 0.02, color: tone('param', 0.12 + 0.4 * f, 0.9 * mParam * sA), radius: 4 });
				lab(`blk${i}`, -X_LAYER + 0.06, y, `${i + 1}`, { mono: true, size: 11, align: 'left', tone: 'muted', opacity: sA * nm });
			}
			const out = model.layers[d - 1].out;
			for (let i = 0; i < N; i++) {
				nS = tile(strips, nS, out[i], 8, LANES[i], stripY, TILE, model.maxStream, mData * sA);
			}
			lab('cap-after', X_MID, stripY + TILE / 2 + 0.12, d === 1 ? 'after 1 block' : `after ${d} blocks`, { tone: 'muted', size: 11, opacity: sA * nm });

			// next-token bars
			if (barA > 0.02) {
				const a = barA * sA;
				nL = lines.setPolyline(nL, [[-2.25, BAR_BASE], [2.25, BAR_BASE]], { width: 1.2, color: tone('muted', 1, 0.9 * a) });
				let top = 0;
				for (let i = 0; i < barVals.length; i++) if (barVals[i] > barVals[top]) top = i;
				// long words would run into their neighbors, so every second label drops one line
				const stagger = model.candidates.some((c) => c.length > 6);
				for (let i = 0; i < barVals.length; i++) {
					const x = BAR_X0 + i * BAR_PITCH;
					const h = Math.max(0.004, barVals[i] * barScale);
					strips.set(nS++, { x, y: BAR_BASE + h / 2, w: BAR_W, h, color: tone('data', i === top ? 1.0 : 0.5, 0.95 * a * mData), radius: 3 });
					lab(`cand${i}`, x, BAR_BASE - (stagger && i % 2 === 1 ? 0.26 : 0.11), model.candidates[i], { mono: true, size: 12, opacity: a * nm });
					lab(`prob${i}`, x, BAR_BASE + h + 0.08, pct(barVals[i]), { mono: true, size: 11, tone: i === top ? 'data' : 'muted', opacity: a * nm });
				}
				const capY = BAR_BASE - (stagger ? 0.46 : 0.32);
				const rest = Math.max(0, 1 - barVals.reduce((p, c) => p + c, 0));
				lab('cap-bars', -2.3, capY, `next word after: ${TEXT}`, { tone: 'muted', size: 11, align: 'left', opacity: a * nm });
				lab('cap-rest', 2.3, capY, `all other words ${pct(rest)}`, { tone: 'muted', size: 11, align: 'right', mono: true, opacity: a * nm });
			}
		}

		// ---- chip outline for the selected lane
		nL = lines.setPolyline(nL, rrPts(selX, Y_CHIP, chipW + px(5), chipH + px(5), px(7)), { width: 1.5, color: tone('text', 0.7, 0.9 * nm) });

		// ---- inspector
		if (inspOn) {
			const { cx, cy, w, h } = inspC;
			backdrop.set(nBd++, { x: cx, y: cy, w, h, color: paperFill(inspA), radius: 8 });
			nIL = ilines.setPolyline(nIL, rrPts(cx, cy, w, h, px(8)), { width: 1, color: tone('muted', 1, 0.7 * inspA) });
			const tok = `${model.tokens[sel]} · ${ORDINALS[sel]}`;
			labI('insp-title', cx, cy + h / 2 - 0.09, tok, { size: 11, tone: 'muted', mono: true });
			const L = model.layers[0];
			const T = (arr: ArrayLike<number>, cols: number, x: number, y: number, size: number, maxAbs: number, a: number) => {
				nI = tile(insp, nI, arr, cols, x, y, size, maxAbs, a * inspA);
			};
			const underline = (x: number, y: number, width: number) => {
				nIL = ilines.setPolyline(nIL, [[x - width / 2, y], [x + width / 2, y]], { width: 2, color: tone('ok', 1, 0.7 * mOk * inspA) });
			};
			if (inspKind === 3) {
				const size = 0.5;
				const sy = cy - 0.07;
				T(L.x[sel], 8, -1.5, sy, size, model.maxStream, mData);
				T(L.attnOut[sel], 8, 0, sy, size, updMax, mOk);
				T(L.mid[sel], 8, 1.5, sy, size, model.maxStream, mData);
				underline(0, sy - size / 2 - 0.07, size);
				const ly = sy + size / 2 + 0.1;
				labI('i3-before', -1.5, ly, 'before', { size: 11, opacity: neutral() });
				labI('i3-update', 0, ly, 'update from attention', { size: 11, tone: 'ok', opacity: mOk });
				labI('i3-after', 1.5, ly, 'after', { size: 11, opacity: neutral() });
				labI('i3-plus', -0.75, sy, '+', { size: 18, opacity: neutral() });
				labI('i3-eq', 0.75, sy, '=', { size: 18, opacity: neutral() });
			} else if (inspKind === 4) {
				const small = 0.5;
				const wide = 0.8;
				const sy = cy - 0.13;
				T(L.norm2[sel], 8, -1.7, sy, small, updMax, mData);
				T(L.hidden[sel], 16, 0, sy, wide, model.maxHidden, mData);
				T(L.mlpOut[sel], 8, 1.7, sy, small, updMax, mOk);
				underline(1.7, sy - small / 2 - 0.07, small);
				const arrow = (x0: number, x1: number) => {
					const col = tone('muted', 1.6, 0.9 * inspA);
					nIL = ilines.setPolyline(nIL, [[x0, sy], [x1, sy]], { width: 1.5, color: col });
					nIL = ilines.setPolyline(nIL, [[x1 - 0.06, sy + 0.05], [x1, sy], [x1 - 0.06, sy - 0.05]], { width: 1.5, color: col });
				};
				arrow(-1.3, -0.56);
				arrow(0.56, 1.3);
				const ly = sy + wide / 2 + 0.1;
				labI('i4-in', -1.7, ly, 'normalized, 64 numbers', { size: 11, opacity: neutral() });
				labI('i4-wide', 0, ly, '4 times wider, 256 numbers', { size: 11, tone: 'data', opacity: mData });
				labI('i4-out', 1.7, ly, 'update, 64 numbers', { size: 11, tone: 'ok', opacity: mOk });
			} else {
				// 64 thin bars per vector. Each chart fills its box, so the shapes can be compared; the spread numbers carry the scale.
				const zy = cy - 0.06;
				const before = L.x[sel];
				const after = L.norm1[sel];
				const half = 0.98;
				const chart = (vals: ArrayLike<number>, x: number, id: string, title: string) => {
					const n0 = vals.length;
					const scale = 0.22 / Math.max(1e-6, ...Array.from(vals, Math.abs));
					const pitch = (2 * half) / n0;
					nIL = ilines.setPolyline(nIL, [[x - half, zy], [x + half, zy]], { width: 1, color: tone('muted', 1.4, 0.9 * inspA) });
					for (let k = 0; k < n0; k++) {
						const val = vals[k];
						const bh = Math.max(0.004, Math.abs(val) * scale);
						insp.set(nI++, {
							x: x - half + (k + 0.5) * pitch,
							y: zy + (val >= 0 ? bh / 2 : -bh / 2),
							w: pitch * 0.72,
							h: bh,
							color: diverging(val >= 0 ? 1 : -1, 0.8, 0.95 * mData * inspA),
							radius: 1
						});
					}
					const m = Array.from(vals).reduce((a, b) => a + b, 0) / n0;
					const sdv = Math.sqrt(Array.from(vals).reduce((a, b) => a + (b - m) ** 2, 0) / n0);
					labI(`i5-${id}`, x, cy + 0.3, title, { size: 11, opacity: neutral() });
					labI(`i5-sd-${id}`, x, cy - 0.37, `spread ${sdv.toFixed(2)}`, { size: 11, mono: true, tone: 'muted' });
				};
				chart(before, -1.2, 'a', 'before');
				chart(after, 1.2, 'b', 'after');
				labI('i5-arrow', 0, zy, '→', { size: 18, tone: 'muted' });
			}
		}

		// ---- play markers
		if (playing) {
			const m = nm;
			for (let i = 0; i < N; i++) {
				glow.set(nG++, { x: LANES[i], y: playY, radius: 12, softness: 1, color: tone('data', 0.9, 0.6 * m) });
				glow.set(nG++, { x: LANES[i], y: playY, radius: 3, color: tone('data', 1.6, 1 * m) });
			}
		}

		boxes.count = nB;
		lines.count = nL;
		dots.count = nD;
		glow.count = nG;
		strips.count = nS;
		backdrop.count = nBd;
		insp.count = nI;
		ilines.count = nIL;
		marks.count = nM;
		marks.upload();
		boxes.upload();
		lines.upload();
		dots.upload();
		glow.upload();
		strips.upload();
		backdrop.upload();
		insp.upload();
		ilines.upload();

		// labels: only set when something changed
		const key = labels.map((l) => `${l.id}|${Math.round(l.x)}|${Math.round(l.y)}|${l.text}|${(l.opacity ?? 1).toFixed(2)}|${l.tone ?? ''}`).join(';');
		if (key !== labelKey || (frameNo < 180 && frameNo % 10 === 0)) {
			labelKey = key;
			ctx.labels.set(labels);
		}
	}

	const scene: Scene = {
		async init(c) {
			ctx = c;
			camera = new Camera2D();
			boxes = new Quads(c, { capacity: 48, blend: 'alpha' });
			lines = new Lines(c, { capacity: 500, blend: 'alpha' });
			dots = new Points(c, { capacity: 80, blend: 'alpha' });
			glow = new Points(c, { capacity: 16, blend: 'additive' });
			strips = new Quads(c, { capacity: 1000, blend: 'alpha' });
			backdrop = new Quads(c, { capacity: 2, blend: 'alpha' });
			insp = new Quads(c, { capacity: 400, blend: 'alpha' });
			ilines = new Lines(c, { capacity: 160, blend: 'alpha' });
			marks = new Lines(c, { capacity: 40, blend: 'alpha' });
			model = buildBlockModel(await loadLm());
			// Colour scales for the updates and the normalized vectors come from the data.
			updMax = 1e-6;
			for (const l of model.layers) {
				for (const m of [l.attnOut, l.mlpOut, l.norm2]) for (const row of m) for (const v of row) updMax = Math.max(updMax, Math.abs(v));
			}
			// One bar scale for every block count, so the bars can be compared as the slider moves.
			barScale = BAR_H_MAX / Math.max(0.1, ...model.probs.flat());
			barVals.length = 0;
			for (let i = 0; i < model.candidates.length; i++) barVals.push(0);
		},
		setParams(p) {
			const prevScenario = scenario;
			focus = clamp(Math.round(p.focus ?? focus), 1, 7);
			depth = clamp(Math.round(p.depth ?? depth), 1, model.depth);
			zoom = clamp(p.zoom ?? zoom, 0, 1);
			column = clamp(Math.round(p.column ?? column), 0, N - 1);
			scenario = Math.round(p.scenario ?? scenario);
			if (paramsSeen && scenario !== prevScenario) playing = false;
			if (!paramsSeen) {
				const t = targets();
				for (const k of Object.keys(B) as BKey[]) B[k] = t[k];
				barA = focus === 7 ? 1 : 0;
				const pr = model.probs[depth - 1];
				for (let i = 0; i < pr.length; i++) barVals[i] = pr[i];
				barsSeen = true;
				stripY = stripTarget(depth);
			}
			paramsSeen = true;
			dirty = true;
		},
		update(fr) {
			const v = ctx.view;
			fitCamera(camera, v, WORLD, lessonInsets(v));
			v.setViewProj(camera.matrix(v.aspect));
			ppw = pxPerWorld(v, camera);
			if (v.width !== lastW || v.height !== lastH) {
				lastW = v.width;
				lastH = v.height;
				dirty = true;
			}

			const dt = fr.dt;
			const snap = ctx.reducedMotion;
			const k = snap ? 1 : 1 - Math.exp(-dt / 0.08);
			let anim = false;

			const t = targets();
			for (const key of Object.keys(B) as BKey[]) {
				const d = t[key] - B[key];
				if (Math.abs(d) > 0.003) {
					B[key] += d * k;
					anim = true;
				} else B[key] = t[key];
			}
			const barT = focus === 7 ? 1 : 0;
			{
				const d = barT - barA;
				if (Math.abs(d) > 0.003) {
					barA += d * k;
					anim = true;
				} else barA = barT;
			}
			if (barsSeen) {
				const pr = model.probs[depth - 1];
				const k2 = snap ? 1 : 1 - Math.exp(-dt / 0.1);
				for (let i = 0; i < pr.length; i++) {
					const d = pr[i] - barVals[i];
					if (Math.abs(d) > 0.0005) {
						barVals[i] += d * k2;
						anim = true;
					} else barVals[i] = pr[i];
				}
			}

			{
				const target = stripTarget(depth);
				const d = target - stripY;
				if (Math.abs(d) > 0.002) {
					stripY += snap ? d : d * (1 - Math.exp(-dt / 0.12));
					anim = true;
				} else stripY = target;
			}

			// inspector kind and fade
			const wanted = zoom < 0.5 && focus >= 3 && focus <= 5 ? focus : 0;
			if (inspKind !== wanted) {
				inspA = snap ? 0 : Math.max(0, inspA - dt * 6);
				anim = true;
				if (inspA <= 0) {
					inspKind = wanted;
				}
			} else if (inspKind !== 0 && inspA < 1) {
				inspA = snap ? 1 : Math.min(1, inspA + dt * 6);
				anim = true;
			}
			if (inspKind === 0) inspA = 0;

			// play
			if (playing) {
				playT += dt;
				const stackView = zoom >= 0.5;
				const y1 = stackView ? (focus === 7 ? BAR_BASE : stripY) : Y_OUT;
				let u = clamp(playT / PLAY_SECONDS, 0, 1);
				// Reduced motion: the markers hop in six jumps (2.5 per second) instead of gliding.
				if (ctx.reducedMotion) u = Math.floor(u * 6) / 6;
				playY = PLAY_Y0 + (y1 - PLAY_Y0) * u;
				anim = true;
				if (playT > PLAY_SECONDS + 0.6) {
					playing = false;
					playY = -9;
				}
			}

			frameNo++;
			if (dirty || anim || frameNo < 90) {
				dirty = false;
				rebuild();
			}

			const inspect = `${model.tokens[column]} · ${ORDINALS[column]}`;
			if (inspect !== lastInspect) {
				lastInspect = inspect;
				ctx.emit('inspect', inspect);
			}
		},
		draw(pass) {
			boxes.draw(pass);
			lines.draw(pass);
			dots.draw(pass);
			marks.draw(pass);
			strips.draw(pass);
			backdrop.draw(pass);
			insp.draw(pass);
			ilines.draw(pass);
			glow.draw(pass);
		},
		command(id) {
			if (id === 'play') {
				playing = true;
				playT = 0;
				dirty = true;
			}
		},
		highlight(t) {
			hl = t;
			dirty = true;
		},
		pointer(e) {
			if (e.type !== 'down') return;
			const i = pickLane(e.x, e.y);
			if (i < 0 || i === column) return;
			column = i;
			dirty = true;
			ctx.emit('param:column', i);
		},
		dispose() {
			boxes.destroy();
			lines.destroy();
			dots.destroy();
			glow.destroy();
			strips.destroy();
			backdrop.destroy();
			insp.destroy();
			ilines.destroy();
			marks.destroy();
		}
	};
	return scene;
};
