import { Camera2D } from '../../../gpu/camera.ts';
import { diverging, mixColor, tone } from '../../../gpu/colors.ts';
import { Lines } from '../../../gpu/primitives/lines.ts';
import { Quads } from '../../../gpu/primitives/quads.ts';
import type { Label, Scene, SceneContext, SceneFactory, ScenePointer, Tone } from '../../../gpu/types.ts';
import { strokeSegment } from '../../../ml/raster.ts';
import { mulberry32 } from '../../../ml/rng.ts';
import { CLASSES, DIM, GRID, IMAGE, PATCH, PATCHES, TOKENS, vitForward, type VitTrace } from '../../../ml/vit.ts';
import { fitCamera, type Insets, type WorldRect } from '../../fit.ts';
import { CLASS_NAMES, SAMPLES } from '../cnn-features/data.ts';
import { WEIGHTS } from './weights.ts';

// The layout is budgeted in CSS px at the reference stage (about 118 px per world unit) and converted to world units.
// Design space: x from -320 to 320, y up, 0 near the token row.
const S = 118;
const px = (n: number) => n / S;

/** The chips row matters here (legend and one readout), so the insets are fixed like the decision trees lesson. */
const INSETS: Insets = { top: 64, right: 24, bottom: 110, left: 24 };

// Drawing: 16 x 16 cells of 10 px, top left corner at (GX, GY). One patch is 40 px.
const CELL = 13;
const GX = -310;
const GY = 278;
const PATCH_PX = PATCH * CELL;
const SIDE = IMAGE * CELL;

// Token row: 17 tokens, class token first.
const ROW_X = (i: number) => -296 + 37 * i;
const TOK_HALF = 15;
const THUMB_Y = 22;
const THUMB_CELL = 6;
const FRAME_TOP = 38;
const FRAME_BOT_SHORT = 8;
const FRAME_BOT_LONG = -127;
const STRIP_TOP = 4;
const STRIP_CELL = 4;
const NUM_Y = 45;
const ARROW_Y = 56;

// Probability bars, right of the drawing.
const BAR_X = -2;
const BAR_W = 120;
const BAR_PITCH = 30;
const BAR_CY = 174;
const NAME_X = -10;
const ARROW_X = -85;
const BAR_H = 14;
const barY = (c: number) => BAR_CY + (1.5 - c) * BAR_PITCH;
const INFO_X = 245;

const clamp = (v: number, a: number, b: number) => Math.min(b, Math.max(a, v));
const lerp = (a: number, b: number, t: number) => a + (b - a) * t;
const smooth = (t: number) => t * t * (3 - 2 * t);

/** World rectangle (from px) around everything that is visible at a given `show`. */
function viewRect(show: number): WorldRect {
	return { x0: px(-320), x1: px(320), y0: px(show >= 2 ? -160 : -20), y1: px(298) };
}

type Rgba = [number, number, number, number];

/** A random order of the 16 slots that moves every patch. */
function randomPerm(rand: () => number): number[] {
	for (;;) {
		const p = Array.from({ length: PATCHES }, (_, i) => i);
		for (let i = p.length - 1; i > 0; i--) {
			const j = Math.floor(rand() * (i + 1));
			[p[i], p[j]] = [p[j], p[i]];
		}
		if (p.filter((v, i) => v !== i).length >= 14) return p;
	}
}

export const createVitScene: SceneFactory = () => {
	let ctx: SceneContext;
	let camera: Camera2D;
	let quads: Quads;
	let lines: Lines;

	// params
	let sample = 4;
	let show = 1;
	let layer = 2;
	let head = 1;
	let positions = 1;
	let focus = 0;
	let scenario = 1;
	let paramsSeen = false;
	let hl: Tone | null = null;

	// drawing, shuffle, network
	const base = new Float32Array(IMAGE * IMAGE); // the drawing as made
	const shown = new Float32Array(IMAGE * IMAGE); // what the model sees: base with the patches moved
	/** perm[s] = which patch of `base` sits in slot s. */
	let perm: number[] = Array.from({ length: PATCHES }, (_, i) => i);
	const shuffleRand = mulberry32(2024);
	let trace: VitTrace = vitForward(WEIGHTS, shown);
	let netDirty = true;
	let snap = true;
	let empty = true;
	let lastAnswer = '';
	let stroking = false;
	let last: [number, number] = [0, 0];
	let needsDraw = true;
	let rect: WorldRect | null = null;
	let lastW = 0;
	let lastH = 0;
	let flight = 1;

	// where each original patch is drawn (slot column and row, eased) and where it is going
	const curCol = new Float32Array(PATCHES);
	const curRow = new Float32Array(PATCHES);
	const tgtCol = new Float32Array(PATCHES);
	const tgtRow = new Float32Array(PATCHES);

	// vector strips: 17 tokens x 32 numbers, eased toward their target
	const dStrip = new Float32Array(TOKENS * DIM);
	const tgtStrip = new Float32Array(TOKENS * DIM);
	let stripScale = 1;
	const dProb = new Float32Array(CLASSES);
	// alphas: strips, position marks, bars, attention, bars emphasis
	const colA = [0, 0, 0, 0, 0];

	/** The class token is the focus on the pages about it, whatever patch was clicked before. */
	const eff = () => (show === 5 || show === 6 ? 0 : clamp(focus, 0, TOKENS - 1));

	/**
	 * Tint for an attention weight. An even spread over the 17 tokens (about 0.06 each) stays nearly invisible,
	 * so only the tokens a head singles out show; 0.3 or more is full strength.
	 */
	const attnAlpha = (w: number) => 0.85 * clamp((w - 0.04) / 0.26, 0, 1);

	const mul = (t: Tone) => (hl === null || hl === t ? 1 : 0.3);

	function patchOrigin(o: number): [number, number] {
		return [Math.floor(o / GRID), o % GRID];
	}

	function aimPatches() {
		for (let s = 0; s < PATCHES; s++) {
			const o = perm[s];
			tgtRow[o] = Math.floor(s / GRID);
			tgtCol[o] = s % GRID;
		}
	}

	function copyPatch(from: Float32Array, fromPatch: number, to: Float32Array, toPatch: number) {
		const [fr, fc] = patchOrigin(fromPatch);
		const [tr, tc] = patchOrigin(toPatch);
		for (let r = 0; r < PATCH; r++) {
			for (let c = 0; c < PATCH; c++) {
				to[(tr * PATCH + r) * IMAGE + tc * PATCH + c] = from[(fr * PATCH + r) * IMAGE + fc * PATCH + c];
			}
		}
	}

	function reshuffle() {
		for (let s = 0; s < PATCHES; s++) copyPatch(base, perm[s], shown, s);
	}

	function unshuffle() {
		for (let s = 0; s < PATCHES; s++) copyPatch(shown, s, base, perm[s]);
	}

	function loadSample(resetPerm: boolean) {
		base.set(SAMPLES[clamp(sample - 1, 0, SAMPLES.length - 1)]);
		if (resetPerm) {
			perm = Array.from({ length: PATCHES }, (_, i) => i);
			aimPatches();
			curCol.set(tgtCol);
			curRow.set(tgtRow);
		}
		netDirty = true;
	}

	function runNet() {
		netDirty = false;
		reshuffle();
		trace = vitForward(WEIGHTS, shown, { noPositions: positions === 0 });

		// Strip values: the patch vectors (and the class token) before block 1 at show 2, the tokens entering it after.
		const src: ArrayLike<number>[] = show >= 3 ? trace.tokens : [WEIGHTS.cls, ...trace.patchEmbed];
		const abs: number[] = [];
		for (let i = 0; i < TOKENS; i++) {
			for (let d = 0; d < DIM; d++) {
				tgtStrip[i * DIM + d] = src[i][d];
				abs.push(Math.abs(src[i][d]));
			}
		}
		abs.sort((a, b) => a - b);
		stripScale = Math.max(0.05, abs[Math.floor(abs.length * 0.95)]);

		let total = 0;
		for (const v of shown) total += v;
		empty = total < 2;
		let top = 0;
		for (let c = 1; c < CLASSES; c++) if (trace.probs[c] > trace.probs[top]) top = c;
		const answer = empty ? '—' : `${CLASS_NAMES[top]} ${Math.round(trace.probs[top] * 100)}%`;
		if (answer !== lastAnswer) {
			lastAnswer = answer;
			ctx.emit('answer', answer);
		}
		needsDraw = true;
	}

	/** Move `d` toward `t`. Returns true while it is still moving. */
	function ease(d: Float32Array, t: ArrayLike<number>, k: number, eps = 2e-4): boolean {
		let moving = false;
		for (let i = 0; i < d.length; i++) {
			const dv = t[i] - d[i];
			if (Math.abs(dv) < eps) d[i] = t[i];
			else {
				d[i] += dv * k;
				moving = true;
			}
		}
		return moving;
	}

	/** 0..1 progress of patch p flying from the drawing into the row. */
	const flightOf = (p: number) => smooth(clamp(flight * 1.5 - 0.5 * (p / (PATCHES - 1)), 0, 1));
	const frameBottom = () => lerp(FRAME_BOT_SHORT, FRAME_BOT_LONG, colA[0]);

	/** Attention of the focused token for the chosen block and head, and the value that counts as full strength. */
	function attention(): number[] {
		return trace.layers[clamp(layer, 1, 2) - 1].attn[clamp(head, 1, 2) - 1][eff()];
	}

	function cellColor(v: number, a: number): Rgba {
		const boost = hl === 'data' ? 1.3 : 1;
		return mixColor(tone('text', 0.07, a), tone('data', 0.78 * boost, a), clamp(v, 0, 1));
	}

	function drawScene() {
		needsDraw = false;
		const mD = mul('data');
		const mP = mul('param');
		const mG = mul('grad');
		const mO = mul('ok');
		const mT = mul('text');
		const attn = attention();
		let n = 0;
		const q = (x: number, y: number, w: number, h: number, color: Rgba, radius = 0) => {
			quads.set(n++, { x: px(x), y: px(y), w: px(w), h: px(h), color, radius });
		};

		// 1. the drawing, as 16 patches that can move: empty cells, then the attention tint, then the ink on top
		for (let o = 0; o < PATCHES; o++) {
			const x0 = GX + curCol[o] * PATCH_PX;
			const y0 = GY - curRow[o] * PATCH_PX;
			for (let r = 0; r < PATCH; r++) {
				for (let c = 0; c < PATCH; c++) {
					q(x0 + (c + 0.5) * CELL, y0 - (r + 0.5) * CELL, CELL * 0.92, CELL * 0.92, tone('text', 0.07, mD * mT));
				}
			}
		}
		if (colA[3] > 0.01) {
			for (let s = 0; s < PATCHES; s++) {
				const o = perm[s];
				const w = attn[1 + s];
				const x0 = GX + curCol[o] * PATCH_PX;
				const y0 = GY - curRow[o] * PATCH_PX;
				q(x0 + PATCH_PX / 2, y0 - PATCH_PX / 2, PATCH_PX - 1, PATCH_PX - 1, tone('grad', 0.9, attnAlpha(w) * colA[3] * mG));
			}
		}
		const boost = hl === 'data' ? 1.3 : 1;
		for (let o = 0; o < PATCHES; o++) {
			const [pr, pc] = patchOrigin(o);
			const x0 = GX + curCol[o] * PATCH_PX;
			const y0 = GY - curRow[o] * PATCH_PX;
			for (let r = 0; r < PATCH; r++) {
				for (let c = 0; c < PATCH; c++) {
					const v = base[(pr * PATCH + r) * IMAGE + pc * PATCH + c];
					if (v < 0.02) continue;
					q(x0 + (c + 0.5) * CELL, y0 - (r + 0.5) * CELL, CELL * 0.92, CELL * 0.92, tone('data', 0.78 * boost, Math.min(1, v * 1.5) * mD * mT));
				}
			}
		}

		// 3. attention on the tokens of the row
		const fb = frameBottom();
		if (colA[3] > 0.01) {
			for (let i = 0; i < TOKENS; i++) {
				const w = attn[i];
				q(ROW_X(i), (FRAME_TOP + fb) / 2, 2 * TOK_HALF, FRAME_TOP - fb, tone('grad', 0.9, attnAlpha(w) * colA[3] * mG));
			}
		}

		// 4. patch thumbnails, flying from the drawing into the row
		for (let p = 0; p < PATCHES; p++) {
			const t = flightOf(p);
			if (t <= 0) continue;
			const sx = GX + (p % GRID) * PATCH_PX + PATCH_PX / 2;
			const sy = GY - Math.floor(p / GRID) * PATCH_PX - PATCH_PX / 2;
			const cx = lerp(sx, ROW_X(p + 1), t);
			const cy = lerp(sy, THUMB_Y, t);
			const cell = lerp(CELL, THUMB_CELL, t);
			const a = mD * mT * Math.min(1, t * 4);
			const pr = Math.floor(p / GRID);
			const pc = p % GRID;
			for (let r = 0; r < PATCH; r++) {
				for (let c = 0; c < PATCH; c++) {
					const v = shown[(pr * PATCH + r) * IMAGE + pc * PATCH + c];
					q(cx + (c - 1.5) * cell, cy - (r - 1.5) * cell, cell * 0.9, cell * 0.9, cellColor(v, a));
				}
			}
		}

		// 5. vector strips
		if (colA[0] > 0.01) {
			for (let i = 0; i < TOKENS; i++) {
				const a = colA[0] * mD * (i === 0 ? 1 : flightOf(i - 1) >= 1 ? 1 : 0);
				if (a <= 0) continue;
				for (let d = 0; d < DIM; d++) {
					q(ROW_X(i), STRIP_TOP - (d + 0.5) * STRIP_CELL, 2 * TOK_HALF - 6, STRIP_CELL * 0.85, diverging(dStrip[i * DIM + d] / stripScale, 0.85, a));
				}
			}
		}

		// 6. position marks
		if (colA[1] > 0.01) {
			for (let i = 0; i < TOKENS; i++) {
				q(ROW_X(i), 7, 2 * TOK_HALF - 6, 2, tone('param', 0.95, colA[1] * mP));
			}
		}

		// 7. probability bars
		if (colA[2] > 0.01) {
			let top = 0;
			for (let c = 1; c < CLASSES; c++) if (dProb[c] > dProb[top]) top = c;
			const dim = empty ? 0.3 : 1;
			const h = BAR_H + 6 * colA[4];
			for (let c = 0; c < CLASSES; c++) {
				const y = barY(c);
				q(BAR_X + BAR_W / 2, y, BAR_W, h, tone('muted', 0.5, 0.35 * colA[2] * mO));
				const w = Math.max(1, BAR_W * dProb[c]);
				const isTop = c === top;
				q(BAR_X + w / 2, y, w, h, isTop ? tone('ok', 0.75, colA[2] * mO * dim) : tone('muted', 0.9, colA[2] * mO * dim * 0.8));
			}
		}
		quads.count = n;
		quads.upload();

		// Lines
		let m = 0;
		const seg = (ax: number, ay: number, bx: number, by: number, color: Rgba, width: number) => {
			lines.set(m++, { a: [px(ax), px(ay)], b: [px(bx), px(by)], width, color });
		};
		const box = (cx: number, cy: number, w: number, h: number, color: Rgba, width: number) => {
			const x0 = px(cx - w / 2);
			const x1 = px(cx + w / 2);
			const y0 = px(cy + h / 2);
			const y1 = px(cy - h / 2);
			m = lines.setPolyline(m, [[x0, y0], [x1, y0], [x1, y1], [x0, y1], [x0, y0]], { width, color });
		};

		// patch grid over the drawing
		const grid = tone('text', 0.55, 0.7 * mT);
		for (let k = 0; k <= GRID; k++) {
			seg(GX + k * PATCH_PX, GY, GX + k * PATCH_PX, GY - SIDE, grid, 1.5);
			seg(GX, GY - k * PATCH_PX, GX + SIDE, GY - k * PATCH_PX, grid, 1.5);
		}
		// token frames
		for (let i = 0; i < TOKENS; i++) {
			const cls = i === 0;
			box(ROW_X(i), (FRAME_TOP + fb) / 2, 2 * TOK_HALF, FRAME_TOP - fb, cls ? tone('param', 1, 0.9 * mP) : tone('muted', 1, 0.6 * mT), cls ? 1.5 : 1);
		}
		// the focused token
		if (colA[3] > 0.01) {
			const a = colA[3];
			if (eff() === 0) {
				box(ROW_X(0), (FRAME_TOP + fb) / 2, 2 * TOK_HALF + 4, FRAME_TOP - fb + 4, tone('param', 1, a * mP), 2.5);
			} else {
				const f = eff();
				const s = f - 1;
				const o = perm[s];
				box(ROW_X(f), (FRAME_TOP + fb) / 2, 2 * TOK_HALF + 4, FRAME_TOP - fb + 4, tone('text', 1, 0.9 * a * mT), 2);
				box(GX + curCol[o] * PATCH_PX + PATCH_PX / 2, GY - curRow[o] * PATCH_PX - PATCH_PX / 2, PATCH_PX, PATCH_PX, tone('text', 1, 0.95 * a * mT), 2.5);
			}
		}
		// class token to the answer
		if (colA[4] > 0.01) {
			const c = tone('ok', 1, 0.75 * colA[4] * mO);
			const pts: Array<[number, number]> = [
				[px(ROW_X(0)), px(FRAME_TOP)],
				[px(ROW_X(0)), px(ARROW_Y)],
				[px(ARROW_X), px(ARROW_Y)],
				[px(ARROW_X), px(BAR_CY)],
				[px(ARROW_X + 20), px(BAR_CY)]
			];
			m = lines.setPolyline(m, pts, { width: 2, color: c });
			seg(ARROW_X + 20, BAR_CY, ARROW_X + 12, BAR_CY + 5, c, 2);
			seg(ARROW_X + 20, BAR_CY, ARROW_X + 12, BAR_CY - 5, c, 2);
		}
		lines.count = m;
		lines.upload();
	}

	function caption(): string {
		if (show <= 1) return 'the patches, row by row, become a row of tokens';
		if (show === 2) return 'each patch is weighed into 32 numbers';
		return positions === 1 ? 'token = patch vector + position vector (violet mark)' : 'position vectors are off: the model sees a bag of patches';
	}

	function buildLabels(): Label[] {
		const v = ctx.view;
		const out: Label[] = [];
		const at = (x: number, y: number) => v.project(px(x), px(y));
		const mD = mul('data');
		const mP = mul('param');
		const mG = mul('grad');
		const mO = mul('ok');
		const mT = mul('text');

		{
			const [x, y] = at(GX + SIDE / 2, GY + 14);
			out.push({ id: 'cap-draw', x, y, text: 'drawing', tone: 'muted', opacity: mT });
		}
		{
			const [x, y] = at(ROW_X(0), THUMB_Y);
			out.push({ id: 'cls', x, y, text: 'CLS', tone: 'param', size: 11, mono: true, opacity: mP });
		}
		for (let p = 0; p < PATCHES; p++) {
			const [x, y] = at(ROW_X(p + 1), NUM_Y);
			out.push({ id: `n${p}`, x, y, text: String(p + 1), tone: 'muted', size: 10, mono: true, opacity: flightOf(p) * mT });
		}
		{
			const [x, y] = at(0, show >= 2 ? -148 : -6);
			out.push({ id: 'cap-row', x, y, text: caption(), tone: positions === 0 && show >= 3 ? 'neg' : 'muted', size: 12, opacity: show >= 3 ? mP : mT });
		}
		if (colA[2] > 0.01) {
			const dim = empty ? 0.3 : 1;
			for (let c = 0; c < CLASSES; c++) {
				const [nx, ny] = at(NAME_X, barY(c));
				out.push({ id: `name${c}`, x: nx, y: ny, text: CLASS_NAMES[c], tone: 'muted', align: 'right', size: 11, opacity: colA[2] * mO * dim });
				const [bx, by] = at(BAR_X + BAR_W + 6, barY(c));
				out.push({ id: `pct${c}`, x: bx, y: by, text: `${Math.round(dProb[c] * 100)}%`, tone: 'text', mono: true, align: 'left', size: 11, opacity: colA[2] * mO * dim });
			}
			const [cx, cy] = at(BAR_X + BAR_W / 2, BAR_CY + 2 * BAR_PITCH + 6);
			out.push({ id: 'cap-bars', x: cx, y: cy, text: 'answer', tone: 'muted', opacity: colA[2] * mO });
		}
		if (colA[3] > 0.01) {
			const [x, y] = at(INFO_X, 205);
			out.push({ id: 'blockhead', x, y, text: `Block ${clamp(layer, 1, 2)} · head ${clamp(head, 1, 2)}`, tone: 'grad', size: 13, opacity: colA[3] * mG });
			const [x2, y2] = at(INFO_X, 186);
			out.push({ id: 'focus', x: x2, y: y2, text: eff() === 0 ? 'looking from the class token' : `looking from patch ${eff()}`, tone: 'muted', size: 11, opacity: colA[3] * mG });
		}
		return out;
	}

	function paint(wx: number, wy: number, fresh: boolean) {
		const x = (wx * S - GX) / CELL;
		const y = (GY - wy * S) / CELL;
		if (fresh) last = [x, y];
		strokeSegment(shown, IMAGE, last[0], last[1], x, y, 0.7);
		last = [x, y];
		unshuffle();
		netDirty = true;
		needsDraw = true;
	}

	const scene: Scene = {
		init(c) {
			ctx = c;
			camera = new Camera2D();
			quads = new Quads(c, { capacity: 1800 });
			lines = new Lines(c, { capacity: 220, blend: 'alpha' });
			aimPatches();
			curCol.set(tgtCol);
			curRow.set(tgtRow);
			loadSample(true);
		},
		setParams(p) {
			const prev = { sample, scenario, show, positions };
			sample = Math.round(p.sample ?? sample);
			show = Math.round(p.show ?? show);
			layer = Math.round(p.layer ?? layer);
			head = Math.round(p.head ?? head);
			positions = Math.round(p.positions ?? positions);
			focus = Math.round(p.focus ?? focus);
			scenario = Math.round(p.scenario ?? scenario);
			if (!paramsSeen || prev.scenario !== scenario) {
				loadSample(true);
				flight = show === 1 && !ctx.reducedMotion ? 0 : 1;
			} else if (prev.sample !== sample) {
				loadSample(false);
			}
			if (prev.show !== show || prev.positions !== positions) netDirty = true;
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
			fitCamera(camera, v, rect, INSETS);
			v.setViewProj(camera.matrix(v.aspect));

			if (netDirty) runNet();

			const instant = ctx.reducedMotion || snap;
			let moving = false;
			const k = instant ? 1 : 1 - Math.exp(-fr.dt / 0.04);
			moving = ease(dStrip, tgtStrip, k, 1e-3) || moving;
			moving = ease(dProb, trace.probs, k) || moving;
			const kp = instant ? 1 : 1 - Math.exp(-fr.dt / 0.12);
			moving = ease(curCol, tgtCol, kp, 2e-3) || moving;
			moving = ease(curRow, tgtRow, kp, 2e-3) || moving;
			snap = false;

			if (flight < 1) {
				flight = ctx.reducedMotion ? 1 : Math.min(1, flight + fr.dt / 1.0);
				moving = true;
			}
			const ka = instant ? 1 : 1 - Math.exp(-fr.dt / 0.09);
			const targets = [show >= 2 ? 1 : 0, show >= 3 && positions === 1 ? 1 : 0, show >= 3 ? 1 : 0, show >= 4 && show !== 6 ? 1 : 0, show >= 6 ? 1 : 0];
			for (let i = 0; i < colA.length; i++) {
				const dv = targets[i] - colA[i];
				if (Math.abs(dv) < 0.002) {
					// Draw once more at the exact target, or a faint remnant stays on screen.
					if (dv !== 0) moving = true;
					colA[i] = targets[i];
				} else {
					colA[i] += dv * ka;
					moving = true;
				}
			}
			const resized = v.width !== lastW || v.height !== lastH;
			// drawScene clears needsDraw, so decide about the labels first.
			const relabel = moving || resized || needsDraw;
			if (moving || needsDraw) drawScene();
			if (relabel) {
				lastW = v.width;
				lastH = v.height;
				ctx.labels.set(buildLabels());
			}
		},
		draw(pass) {
			quads.draw(pass);
			lines.draw(pass);
		},
		command(id) {
			if (id === 'clear') {
				base.fill(0);
				netDirty = true;
			} else if (id === 'shuffle') {
				perm = randomPerm(shuffleRand);
				aimPatches();
				netDirty = true;
			} else if (id === 'restore') {
				perm = Array.from({ length: PATCHES }, (_, i) => i);
				aimPatches();
				netDirty = true;
			}
			needsDraw = true;
		},
		highlight(t) {
			hl = t;
			needsDraw = true;
		},
		pointer(e: ScenePointer) {
			if (e.type === 'leave' || e.type === 'up') {
				stroking = false;
				return;
			}
			const [wx, wy] = ctx.view.unproject(e.x, e.y);
			if (e.type === 'down') {
				const X = wx * S;
				const Y = wy * S;
				if (show >= 4) {
					const i = Math.round((X + 296) / 37);
					if (i >= 0 && i < TOKENS && Math.abs(X - ROW_X(i)) <= TOK_HALF && Y <= FRAME_TOP && Y >= frameBottom()) {
						ctx.emit('param:focus', i);
						return;
					}
				}
				const m = 5;
				const inside = X >= GX - m && X <= GX + SIDE + m && Y <= GY + m && Y >= GY - SIDE - m;
				if (!inside) return;
				if (show === 7) {
					stroking = true;
					paint(wx, wy, true);
				} else if (show >= 4) {
					const col = clamp(Math.floor((X - GX) / PATCH_PX), 0, GRID - 1);
					const rowIdx = clamp(Math.floor((GY - Y) / PATCH_PX), 0, GRID - 1);
					ctx.emit('param:focus', 1 + rowIdx * GRID + col);
				}
				return;
			}
			if (stroking) paint(wx, wy, false);
		},
		dispose() {
			quads.destroy();
			lines.destroy();
		}
	};
	return scene;
};
