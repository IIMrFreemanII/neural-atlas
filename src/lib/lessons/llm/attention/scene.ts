import { Camera2D } from '../../../gpu/camera.ts';
import { sequential, tone } from '../../../gpu/colors.ts';
import { bezier, Lines } from '../../../gpu/primitives/lines.ts';
import { Quads } from '../../../gpu/primitives/quads.ts';
import type { Label, Scene, SceneContext, SceneFactory, ScenePointer, Tone } from '../../../gpu/types.ts';
import type { Mat } from '../../../ml/vec.ts';
import { fitCamera, lessonInsets, type WorldRect } from '../../fit.ts';
import { layoutChips, pxPerWorld, type ChipBox } from '../shared/chips.ts';
import { stripLength, writeStrip } from '../shared/strip.ts';
import { loadLm } from '../shared/lm.ts';
import { FEATURES, HEADS, SENTENCES, runHead, type HeadRun } from './model.ts';
import { buildTrained, type TrainedSentence } from './trained.ts';

const WORLD: WorldRect = { x0: -2.6, x1: 2.6, y0: -2.1, y1: 2.1 };
const N = 9;
/** Most tokens any shown sentence can have; the eased arrays are this big. */
const NMAX = 12;
const CX = 0.35;
const FONT = 13;
const FONT_T = 12; // narrower chips so 11 tokens fit where 9 do
const CHIP_H_PX = 25;
const ROW_LOW = -0.35;
const ROW_HIGH = 0.75;

// Mechanics rows (world y). Chips sit at ROW_HIGH (0.65 to 0.85).
const KEY_CAP_Y = 0.42;
const KEY_Y = 0.25;
const MATCH_Y = -0.12; // zero line of the score bars
const MATCH_H = 0.16;
const MATCH_LAB_Y = -0.4;
const SHARE_BASE = -0.95;
const SHARE_H = 0.36;
const SHARE_LAB_Y = -1.05;
const VAL_Y = -1.36;
const VAL_CELL = 0.055;
const VAL_THICK = 0.1;
const OUT_Y = -1.78;
const OUT_CELL = 0.1;
const OUT_THICK = 0.2;
const OUT_CAP_Y = -2.0;
const QUERY_X = -2.25;
const CAPTION_X = -2.55;

// Matrix
const MAT_Y0 = 0.3;
const MAT_PITCH = 0.27;
const MAT_CELL_H = 0.245;
// The trained sentences have up to 11 tokens, so their rows are packed closer.
const MAT_PITCH_T = 0.215;
const MAT_CELL_H_T = 0.19;
const MAT_GAP_PX = 5;

const clamp = (v: number, a: number, b: number) => Math.min(b, Math.max(a, v));
const zeros = (r: number, c: number): Mat => Array.from({ length: r }, () => new Array<number>(c).fill(0));
/** Copies `m` into a zero matrix of NMAX rows and `cols` columns, so a target always has the shape of the eased copy. */
const padTo = (m: Mat, cols: number): Mat => {
	const out = zeros(NMAX, cols);
	for (let i = 0; i < m.length; i++) for (let j = 0; j < m[i].length; j++) out[i][j] = m[i][j];
	return out;
};

function easeMat(cur: Mat, tgt: Mat, k: number): void {
	for (let i = 0; i < cur.length; i++) {
		const a = cur[i];
		const b = tgt[i];
		for (let j = 0; j < a.length; j++) a[j] += (b[j] - a[j]) * k;
	}
}

function copyMat(cur: Mat, tgt: Mat): void {
	for (let i = 0; i < cur.length; i++) for (let j = 0; j < cur[i].length; j++) cur[i][j] = tgt[i][j];
}

export const createAttentionScene: SceneFactory = () => {
	let ctx: SceneContext;
	let camera: Camera2D;
	let quads: Quads;
	let lines: Lines;

	// params
	let sentence = 0;
	let head = 0;
	let causal = false;
	let sharp = 1;
	let arcsOn = true;
	let mech = 0;
	let matrix = false;
	let tokenParam = 8;
	let scenario = 1;
	let trainedOn = false;
	let tsent = 0;
	let tblock = 0;
	let thead = 0;
	let trainedData: TrainedSentence[] = [];
	let paramsSeen = false;
	let hl: Tone | null = null;

	// selection
	let locked = 8;
	let hoverSel = -1;
	let hoverCol = -1; // matrix column under the pointer
	let hoverCell: { kind: 'value' | 'out'; tok: number; i: number } | null = null;

	// model state: targets and displayed (eased) copies
	let run: HeadRun = runHead(SENTENCES[0], HEADS[0], { causal: false, sharp: 1 });
	let wT: Mat = zeros(NMAX, NMAX);
	let maskT: Mat = zeros(NMAX, NMAX);
	let scoresT: Mat = zeros(NMAX, NMAX);
	let qT: Mat = zeros(NMAX, 4);
	let kT: Mat = zeros(NMAX, 4);
	let vT: Mat = zeros(NMAX, 6);
	let oT: Mat = zeros(NMAX, 6);
	const W = zeros(NMAX, NMAX);
	const S = zeros(NMAX, NMAX);
	const M = zeros(NMAX, NMAX);
	const Q = zeros(NMAX, 4);
	const K = zeros(NMAX, 4);
	const V = zeros(NMAX, 6);
	const O = zeros(NMAX, 6);
	let needSnap = true;
	let modelDirty = true;

	// layout
	let boxes: ChipBox[] = [];
	let boxesScale = 0;
	let boxesTokens: readonly string[] = [];
	let n = N; // token count of what is shown
	let matPitch = MAT_PITCH;
	let matCellH = MAT_CELL_H;
	let rowY = ROW_LOW;
	let rowSnap = true;
	const fade = { keys: 0, match: 0, share: 0, values: 0, matrix: 0, arcs: 0 };

	// emitted
	let lastFrom = '';
	let lastTo = '';

	const mul = (t: Tone) => (hl === null || hl === t ? 1 : 0.3);
	const sel = () => (hoverSel >= 0 ? hoverSel : locked);
	const showTrained = () => trainedOn && trainedData.length > 0;
	/** Tokens the chips are laid out from. The toy lays out its first sentence for both, as before. */
	const layoutTokens = (): readonly string[] => (showTrained() ? trainedData[tsent].tokens : SENTENCES[0].tokens);
	/** Tokens whose text is shown. */
	const shownTokens = (): readonly string[] => (showTrained() ? trainedData[tsent].tokens : SENTENCES[sentence].tokens);

	function recompute() {
		modelDirty = false;
		if (showTrained()) {
			const w = trainedData[tsent].weights[tblock][thead];
			n = w.length;
			matPitch = MAT_PITCH_T;
			matCellH = MAT_CELL_H_T;
			run = { q: zeros(n, 4), k: zeros(n, 4), v: zeros(n, 6), scores: zeros(n, n), weights: w, out: zeros(n, 6) };
			maskT = padTo(
				w.map((row, i) => row.map((_, j) => (j > i ? 1 : 0))),
				NMAX
			);
			scoresT = zeros(NMAX, NMAX);
		} else {
			n = N;
			matPitch = MAT_PITCH;
			matCellH = MAT_CELL_H;
			run = runHead(SENTENCES[sentence], HEADS[head], { causal, sharp });
			maskT = padTo(
				run.weights.map((row, i) => row.map((_, j) => (causal && j > i ? 1 : 0))),
				NMAX
			);
			scoresT = padTo(
				run.scores.map((row) => row.map((s) => (Number.isFinite(s) ? s : 0))),
				NMAX
			);
		}
		wT = padTo(run.weights, NMAX);
		qT = padTo(run.q, 4);
		kT = padTo(run.k, 4);
		vT = padTo(run.v, 6);
		oT = padTo(run.out, 6);
		if (needSnap || ctx.reducedMotion) {
			copyMat(W, wT);
			copyMat(S, scoresT);
			copyMat(M, maskT);
			copyMat(Q, qT);
			copyMat(K, kT);
			copyMat(V, vT);
			copyMat(O, oT);
			needSnap = false;
		}
	}

	function ensureLayout() {
		const scale = pxPerWorld(ctx.view, camera);
		const tokens = layoutTokens();
		if (Math.abs(scale - boxesScale) < 1e-6 && boxesTokens === tokens) return;
		boxesScale = scale;
		boxesTokens = tokens;
		const trained = showTrained();
		boxes = layoutChips(tokens, {
			x0: CX - 6,
			x1: CX + 6,
			y: 0,
			scale,
			fontPx: trained ? FONT_T : FONT,
			padPx: trained ? 5 : 6,
			gapPx: 5,
			heightPx: CHIP_H_PX,
			minChars: trained ? 3 : 5
		});
	}

	function hitChip(wx: number, wy: number): number {
		const slack = 3 / boxesScale;
		for (const b of boxes) {
			if (Math.abs(wx - b.x) <= b.w / 2 + slack / 2 && Math.abs(wy - (rowY + b.y)) <= b.h / 2 + slack) return b.index;
		}
		return -1;
	}

	function matRowY(i: number): number {
		return MAT_Y0 - i * matPitch;
	}

	function hitMatrix(wx: number, wy: number): { row: number; col: number } | null {
		if (!matrix || fade.matrix < 0.5 || boxes.length < n) return null;
		const row = Math.round((MAT_Y0 - wy) / matPitch);
		if (row < 0 || row >= n || Math.abs(wy - matRowY(row)) > matPitch / 2) return null;
		for (const b of boxes) if (Math.abs(wx - b.x) <= b.w / 2 + 2.5 / boxesScale) return { row, col: b.index };
		return null;
	}

	function hitFeature(wx: number, wy: number): typeof hoverCell {
		if (matrix || mech < 4 || fade.values < 0.5 || boxes.length < n) return null;
		const g = VAL_CELL * 0.12;
		const len = stripLength(6, VAL_CELL, g);
		for (const b of boxes) {
			for (let i = 0; i < 6; i++) {
				const x = b.x - len / 2 + VAL_CELL / 2 + i * (VAL_CELL + g);
				if (Math.abs(wx - x) <= (VAL_CELL + g) / 2 && Math.abs(wy - VAL_Y) <= VAL_THICK / 2 + 0.02) {
					return { kind: 'value', tok: b.index, i };
				}
			}
		}
		const og = OUT_CELL * 0.12;
		const olen = stripLength(6, OUT_CELL, og);
		const ox = outX();
		for (let i = 0; i < 6; i++) {
			const x = ox - olen / 2 + OUT_CELL / 2 + i * (OUT_CELL + og);
			if (Math.abs(wx - x) <= (OUT_CELL + og) / 2 && Math.abs(wy - OUT_Y) <= OUT_THICK / 2 + 0.02) {
				return { kind: 'out', tok: sel(), i };
			}
		}
		return null;
	}

	function outX(): number {
		const b = boxes[sel()];
		return clamp(b ? b.x : 0, -2.0, 2.0);
	}

	function bestTarget(i: number): { j: number; w: number } {
		let j = 0;
		for (let k = 1; k < n; k++) if (run.weights[i][k] > run.weights[i][j]) j = k;
		return { j, w: run.weights[i][j] };
	}

	function writeAll(labels: Label[]) {
		const v = ctx.view;
		const s = sel();
		const toks = shownTokens();
		const sc = boxesScale;
		const mData = mul('data');
		const mOk = mul('ok');
		const mParam = mul('param');
		const mText = mul('text');
		let nq = 0;
		let nl = 0;
		const q = quads;
		const l = lines;
		const chipH = CHIP_H_PX / sc;

		// chips
		for (const b of boxes) {
			const w = arcsOn ? W[s][b.index] : 0;
			const mask = arcsOn ? M[s][b.index] : 0;
			const dim = 1 - 0.6 * mask;
			q.set(nq++, {
				x: b.x,
				y: rowY,
				w: b.w,
				h: chipH,
				color: tone('data', (0.14 + 0.5 * w) * dim, 0.9 * (hl === null || hl === 'data' || b.index === s ? 1 : 0.5)),
				radius: 6
			});
			labels.push({
				id: `chip-${b.index}`,
				x: v.project(b.x, rowY)[0],
				y: v.project(b.x, rowY)[1],
				text: toks[b.index],
				tone: 'text',
				mono: true,
				size: showTrained() ? FONT_T : FONT,
				opacity: dim * (hl === null ? 1 : Math.max(mText, b.index === s ? mOk : 0))
			});
		}

		// selection outline
		const sb = boxes[s];
		if (sb) {
			const pad = 2.5 / sc;
			const x0 = sb.x - sb.w / 2 - pad;
			const x1 = sb.x + sb.w / 2 + pad;
			const y0 = rowY - chipH / 2 - pad;
			const y1 = rowY + chipH / 2 + pad;
			nl = l.setPolyline(
				nl,
				[
					[x0, y0],
					[x1, y0],
					[x1, y1],
					[x0, y1],
					[x0, y0]
				],
				{ width: 2, color: tone('ok', 1, 0.8 * mOk) }
			);
		}

		// arcs and percentages
		if (fade.arcs > 0.01 && sb) {
			const ya = rowY + chipH / 2 + 0.28;
			const yl = rowY + chipH / 2 + 0.13;
			const order = boxes.map((b) => b.index).filter((j) => j !== s);
			order.sort((a, b) => W[s][a] - W[s][b]);
			for (const j of order) {
				const w = clamp(W[s][j], 0, 1);
				const vis = fade.arcs * (1 - M[s][j]);
				if (vis < 0.02) continue;
				const bj = boxes[j];
				const d = Math.abs(j - s);
				const peak = 0.15 + 0.09 * d;
				const hh = peak / 0.75;
				const pts = bezier([sb.x, ya], [sb.x, ya + hh], [bj.x, ya + hh], [bj.x, ya], 28);
				const pulse = !ctx.reducedMotion && w > 0.08;
				const a = (0.15 + 0.85 * w) * vis * mData;
				if (w > 0.15) {
					nl = l.setPolyline(nl, pts, { width: 4 + 8 * w, color: tone('data', 1, 0.035 * w * vis * mData) });
				}
				nl = l.setPolyline(nl, pts, {
					width: 1 + 5 * w,
					color: tone('data', w < 0.2 ? 1 : 0.85, a),
					pulseSpeed: pulse ? 0.35 : 0,
					pulseCount: pulse ? 1 : 0,
					pulseGain: pulse ? 0.6 * w : 0
				});
			}
			for (const b of boxes) {
				const w = clamp(W[s][b.index], 0, 1);
				const vis = fade.arcs * (1 - M[s][b.index]);
				if (vis < 0.05) continue;
				labels.push({
					id: `pct-${b.index}`,
					x: v.project(b.x, yl)[0],
					y: v.project(b.x, yl)[1],
					text: `${Math.round(w * 100)}%`,
					tone: 'data',
					mono: true,
					size: 11,
					opacity: (0.35 + 0.65 * w) * vis * mData
				});
			}
		}

		// ---- mechanics rows ----
		const wsel = W[s];
		const maxW = Math.max(1e-6, ...wsel);
		if (fade.keys > 0.01) {
			const a = fade.keys;
			const KC = 0.075;
			for (const b of boxes) {
				nq = writeStrip(q, nq, K[b.index], { x: b.x, y: KEY_Y, cell: KC, thickness: 0.075, maxAbs: 1.2, alpha: a * mParam, radius: 1.5 });
				const len = stripLength(4, KC);
				nl = l.setPolyline(
					nl,
					[
						[b.x - len / 2, KEY_Y - 0.065],
						[b.x + len / 2, KEY_Y - 0.065]
					],
					{ width: 1.5, color: tone('param', 0.9, 0.6 * a * mParam) }
				);
			}
			// query of the selected token
			const qmax = Math.max(1.2, ...Q[s].map(Math.abs));
			nq = writeStrip(q, nq, Q[s], { x: QUERY_X, y: KEY_Y, cell: 0.075, thickness: 0.075, maxAbs: qmax, alpha: a * mOk, radius: 1.5 });
			const qlen = stripLength(4, 0.075);
			nl = l.setPolyline(
				nl,
				[
					[QUERY_X - qlen / 2, KEY_Y - 0.065],
					[QUERY_X + qlen / 2, KEY_Y - 0.065]
				],
				{ width: 1.5, color: tone('ok', 0.9, 0.6 * a * mOk) }
			);
			const [qx, qy] = v.project(QUERY_X, KEY_CAP_Y);
			labels.push({ id: 'cap-query', x: qx, y: qy, text: `query of ${toks[s]}`, tone: 'ok', size: 11, mono: true, opacity: a * mOk });
			const [kx, ky] = v.project(boxes[0].x - boxes[0].w / 2, KEY_CAP_Y);
			labels.push({ id: 'cap-key', x: kx, y: ky, text: 'key of each word', tone: 'param', size: 11, mono: true, align: 'left', opacity: a * mParam });
		}

		if (fade.match > 0.01) {
			const a = fade.match;
			const row = S[s];
			let top = 1;
			for (let j = 0; j < n; j++) if (M[s][j] < 0.5) top = Math.max(top, Math.abs(row[j]));
			const xa = boxes[0].x - 0.27;
			const xb = boxes[n - 1].x + 0.27;
			nl = l.setPolyline(
				nl,
				[
					[xa, MATCH_Y],
					[xb, MATCH_Y]
				],
				{ width: 1, color: tone('muted', 1, 0.45 * a * mText) }
			);
			for (const b of boxes) {
				const j = b.index;
				const vis = a * (1 - M[s][j]);
				if (vis < 0.02) continue;
				const h = (row[j] / top) * MATCH_H;
				const hh = Math.max(Math.abs(h), 0.004);
				q.set(nq++, { x: b.x, y: MATCH_Y + h / 2, w: 0.1, h: hh, color: tone('text', 0.6, vis * mText), radius: 1.5 });
				labels.push({
					id: `score-${j}`,
					x: v.project(b.x, MATCH_LAB_Y)[0],
					y: v.project(b.x, MATCH_LAB_Y)[1],
					text: row[j].toFixed(1),
					tone: 'text',
					mono: true,
					size: 11,
					opacity: vis * mText
				});
			}
			const [cx, cy] = v.project(CAPTION_X, MATCH_Y);
			labels.push({ id: 'cap-score', x: cx, y: cy, text: 'score', tone: 'text', size: 11, mono: true, align: 'left', opacity: a * mText });
		}

		if (fade.share > 0.01) {
			const a = fade.share;
			for (const b of boxes) {
				const j = b.index;
				const w = clamp(wsel[j], 0, 1);
				const vis = a * (1 - M[s][j]);
				if (vis < 0.02) continue;
				const h = Math.max(w * SHARE_H, 0.004);
				q.set(nq++, { x: b.x, y: SHARE_BASE + h / 2, w: 0.1, h, color: tone('data', 0.85, vis * mData), radius: 1.5 });
				labels.push({
					id: `share-${j}`,
					x: v.project(b.x, SHARE_LAB_Y)[0],
					y: v.project(b.x, SHARE_LAB_Y)[1],
					text: `${Math.round(w * 100)}%`,
					tone: 'data',
					mono: true,
					size: 11,
					opacity: vis * mData
				});
			}
			const [cx, cy] = v.project(CAPTION_X, SHARE_BASE + 0.15);
			labels.push({ id: 'cap-share', x: cx, y: cy, text: 'share', tone: 'data', size: 11, mono: true, align: 'left', opacity: a * mData });
		}

		if (fade.values > 0.01) {
			const a = fade.values;
			for (const b of boxes) {
				const al = (0.2 + 0.8 * (clamp(wsel[b.index], 0, 1) / maxW)) * a * (1 - M[s][b.index] * 0.8) * mData;
				nq = writeStrip(q, nq, V[b.index], { x: b.x, y: VAL_Y, cell: VAL_CELL, thickness: VAL_THICK, maxAbs: 1.2, alpha: al, radius: 1.5 });
			}
			const ox = outX();
			nq = writeStrip(q, nq, O[s], { x: ox, y: OUT_Y, cell: OUT_CELL, thickness: OUT_THICK, maxAbs: 1, alpha: a * mData, radius: 2 });
			for (const b of boxes) {
				const w = clamp(wsel[b.index], 0, 1);
				if (w <= 0.1 || M[s][b.index] > 0.5) continue;
				nl = l.setPolyline(
					nl,
					[
						[b.x, VAL_Y - VAL_THICK / 2 - 0.01],
						[ox + (b.x - ox) * 0.08, OUT_Y + OUT_THICK / 2 + 0.01]
					],
					{ width: 1 + 3 * w, color: tone('data', 0.9, (0.2 + 0.45 * w) * a * mData) }
				);
			}
			const [cx, cy] = v.project(CAPTION_X, VAL_Y);
			labels.push({ id: 'cap-values', x: cx, y: cy, text: 'values', tone: 'data', size: 11, mono: true, align: 'left', opacity: a * mData });
			const [ocx, ocy] = v.project(ox, OUT_CAP_Y);
			labels.push({ id: 'cap-out', x: ocx, y: ocy, text: `new vector for ${toks[s]}`, tone: 'data', size: 11, mono: true, opacity: a * mData });
			if (hoverCell) {
				const arr = hoverCell.kind === 'value' ? V[hoverCell.tok] : O[hoverCell.tok];
				const cell = hoverCell.kind === 'value' ? VAL_CELL : OUT_CELL;
				const g = cell * 0.12;
				const len = stripLength(6, cell, g);
				const cxw = (hoverCell.kind === 'value' ? boxes[hoverCell.tok].x : ox) - len / 2 + cell / 2 + hoverCell.i * (cell + g);
				const cyw = hoverCell.kind === 'value' ? VAL_Y + 0.16 : OUT_Y + 0.3;
				const text = `${FEATURES[hoverCell.i]} ${arr[hoverCell.i].toFixed(2)}`;
				const half = (text.length * 11 * 0.6) / 2 / sc;
				const [fx, fy] = v.project(clamp(cxw, WORLD.x0 + half, WORLD.x1 - half), cyw);
				labels.push({ id: 'feature-tip', x: fx, y: fy, text, tone: 'text', mono: true, size: 11 });
			}
		}

		// ---- matrix ----
		if (fade.matrix > 0.01) {
			const a = fade.matrix;
			const cellGap = MAT_GAP_PX / sc;
			for (let i = 0; i < n; i++) {
				for (const b of boxes) {
					const j = b.index;
					const mk = M[i][j];
					const base = sequential(clamp(W[i][j], 0, 1), 0.85, 0.95 * a * mData);
					const dimc = tone('muted', 0.2, 0.5 * a * mData);
					q.set(nq++, {
						x: b.x,
						y: matRowY(i),
						w: b.w,
						h: matCellH - cellGap * 0.2,
						color: [
							base[0] + (dimc[0] - base[0]) * mk,
							base[1] + (dimc[1] - base[1]) * mk,
							base[2] + (dimc[2] - base[2]) * mk,
							base[3] + (dimc[3] - base[3]) * mk
						],
						radius: 3
					});
				}
				const lx = boxes[0].x - boxes[0].w / 2 - 6 / sc;
				const [px, py] = v.project(lx, matRowY(i));
				labels.push({
					id: `row-${i}`,
					x: px,
					y: py,
					text: toks[i],
					tone: i === s ? 'ok' : 'text',
					mono: true,
					size: 12,
					align: 'right',
					opacity: a * (i === s ? mOk : mText)
				});
			}
			const rx0 = boxes[0].x - boxes[0].w / 2 - 2 / sc;
			const rx1 = boxes[n - 1].x + boxes[n - 1].w / 2 + 2 / sc;
			const ry0 = matRowY(s) - matPitch / 2 + 0.01;
			const ry1 = matRowY(s) + matPitch / 2 - 0.01;
			nl = l.setPolyline(
				nl,
				[
					[rx0, ry0],
					[rx1, ry0],
					[rx1, ry1],
					[rx0, ry1],
					[rx0, ry0]
				],
				{ width: 2, color: tone('ok', 1, 0.8 * a * mOk) }
			);
			if (hoverCol >= 0 && hoverSel >= 0) {
				const w = run.weights[hoverSel][hoverCol];
				const text = `${toks[hoverSel]} → ${toks[hoverCol]} ${Math.round(w * 100)}%`;
				const half = (text.length * 11 * 0.6) / 2 / sc;
				const [tx, ty] = v.project(clamp(boxes[hoverCol].x, WORLD.x0 + half, WORLD.x1 - half), 0.54);
				labels.push({ id: 'matrix-tip', x: tx, y: ty, text, tone: 'text', mono: true, size: 11, opacity: a });
			}
		}

		q.count = nq;
		l.count = nl;
		q.upload();
		l.upload();
	}

	const scene: Scene = {
		async init(c) {
			ctx = c;
			camera = new Camera2D();
			quads = new Quads(c, { capacity: 320, blend: 'alpha' });
			lines = new Lines(c, { capacity: 900, blend: 'alpha' });
			trainedData = buildTrained(await loadLm());
		},
		setParams(p) {
			const prev = { tokenParam, scenario };
			const nSentence = clamp(Math.round(p.sentence ?? sentence), 0, 1);
			const nHead = clamp(Math.round(p.head ?? head), 0, 2);
			const nCausal = (p.causal ?? 0) >= 0.5;
			const nSharp = p.sharp ?? sharp;
			const nTrained = (p.trained ?? 0) >= 0.5;
			const nTsent = clamp(Math.round(p.tsentence ?? tsent), 0, Math.max(0, trainedData.length - 1));
			const nTblock = clamp(Math.round(p.tblock ?? tblock), 0, 3);
			const nThead = clamp(Math.round(p.thead ?? thead), 0, 3);
			if (nTrained !== trainedOn || nTsent !== tsent || nTblock !== tblock || nThead !== thead) modelDirty = true;
			const sentenceChanged = nTrained && nTrained === trainedOn && nTsent !== tsent;
			trainedOn = nTrained;
			tsent = nTsent;
			tblock = nTblock;
			thead = nThead;
			if (nSentence !== sentence || nHead !== head || nCausal !== causal || Math.abs(nSharp - sharp) > 1e-6) modelDirty = true;
			sentence = nSentence;
			head = nHead;
			causal = nCausal;
			sharp = nSharp;
			arcsOn = (p.arcs ?? 1) >= 0.5;
			mech = trainedOn ? 0 : Math.round(p.mech ?? 0);
			matrix = (p.matrix ?? 0) >= 0.5 || trainedOn; // the trained view is always the grid
			tokenParam = clamp(Math.round(p.token ?? tokenParam), 0, NMAX - 1);
			scenario = Math.round(p.scenario ?? scenario);
			if (!paramsSeen || prev.tokenParam !== tokenParam || prev.scenario !== scenario) {
				locked = tokenParam;
				hoverSel = -1;
				hoverCol = -1;
			}
			if (sentenceChanged) {
				// A new sentence: follow its last word, the one that predicts what comes next.
				locked = trainedData[tsent].tokens.length - 1;
				hoverSel = -1;
				hoverCol = -1;
			}
			if (!paramsSeen) {
				needSnap = true;
				rowSnap = true;
				for (const k of Object.keys(fade) as Array<keyof typeof fade>) fade[k] = -1;
			}
			paramsSeen = true;
		},
		update(fr) {
			const v = ctx.view;
			fitCamera(camera, v, WORLD, lessonInsets(v));
			v.setViewProj(camera.matrix(v.aspect));
			ensureLayout();
			if (modelDirty) recompute();

			const instant = ctx.reducedMotion;
			const k = instant ? 1 : 1 - Math.exp(-fr.dt / 0.15);
			if (k > 0 && k < 1) {
				easeMat(W, wT, k);
				easeMat(S, scoresT, k);
				easeMat(M, maskT, k);
				easeMat(Q, qT, k);
				easeMat(K, kT, k);
				easeMat(V, vT, k);
				easeMat(O, oT, k);
			}

			// row position and fades
			const targetRow = mech === 0 && !matrix ? ROW_LOW : ROW_HIGH;
			const kr = instant || rowSnap ? 1 : 1 - Math.exp(-fr.dt / 0.1);
			rowY += (targetRow - rowY) * (instant || rowSnap ? 1 : kr);
			if (Math.abs(targetRow - rowY) < 1e-4) rowY = targetRow;
			rowSnap = false;
			const kf = instant ? 1 : 1 - Math.exp(-fr.dt / 0.08);
			const tg = {
				keys: mech >= 1 && !matrix ? 1 : 0,
				match: mech >= 2 && !matrix ? 1 : 0,
				share: mech >= 3 && !matrix ? 1 : 0,
				values: mech >= 4 && !matrix ? 1 : 0,
				matrix: matrix ? 1 : 0,
				arcs: arcsOn ? 1 : 0
			};
			for (const key of Object.keys(fade) as Array<keyof typeof fade>) {
				if (fade[key] < 0) fade[key] = tg[key];
				else {
					fade[key] += (tg[key] - fade[key]) * kf;
					if (Math.abs(tg[key] - fade[key]) < 0.003) fade[key] = tg[key];
				}
			}

			if (locked >= n) locked = n - 1;
			if (hoverSel >= n) hoverSel = -1;
			if (hoverCol >= n) hoverCol = -1;
			const labels: Label[] = [];
			writeAll(labels);
			ctx.labels.set(labels);

			const s = sel();
			const toks = shownTokens();
			if (toks[s] !== lastFrom) {
				lastFrom = toks[s];
				ctx.emit('from', toks[s]);
			}
			const best = bestTarget(s);
			// Before the arcs are introduced (step 1) the answer stays hidden.
			const toText = arcsOn ? `${toks[best.j]} ${Math.round(best.w * 100)}%` : '—';
			if (toText !== lastTo) {
				lastTo = toText;
				ctx.emit('to', toText);
			}
		},
		draw(pass) {
			quads.draw(pass);
			lines.draw(pass);
		},
		highlight(t) {
			hl = t;
		},
		pointer(e: ScenePointer) {
			if (e.type === 'leave') {
				hoverSel = -1;
				hoverCol = -1;
				hoverCell = null;
				return;
			}
			const [wx, wy] = ctx.view.unproject(e.x, e.y);
			const chip = hitChip(wx, wy);
			const cell = hitMatrix(wx, wy);
			hoverCell = hitFeature(wx, wy);
			if (chip >= 0) {
				hoverSel = chip;
				hoverCol = -1;
			} else if (cell) {
				hoverSel = cell.row;
				hoverCol = cell.col;
			} else {
				hoverSel = -1;
				hoverCol = -1;
			}
			if (e.type === 'down' && hoverSel >= 0) locked = hoverSel;
		},
		dispose() {
			quads.destroy();
			lines.destroy();
		}
	};
	return scene;
};
