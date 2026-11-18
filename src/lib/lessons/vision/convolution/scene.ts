import { Camera2D } from '../../../gpu/camera.ts';
import { diverging, tone } from '../../../gpu/colors.ts';
import { Lines } from '../../../gpu/primitives/lines.ts';
import { Quads } from '../../../gpu/primitives/quads.ts';
import type { Label, Scene, SceneContext, SceneFactory, ScenePointer, Tone } from '../../../gpu/types.ts';
import { conv2d, makeGrid, patchAt, type Grid } from '../../../ml/conv.ts';
import { strokeSegment } from '../../../ml/raster.ts';
import { fitCamera, lessonInsets, type WorldRect } from '../../fit.ts';
import { cellAt, cellCenter, toPixel, writeGrid, type GridLayout } from '../shared/grid.ts';
import { IMAGES, KERNELS, N } from './data.ts';

const WORLD: WorldRect = { x0: -2.6, x1: 2.6, y0: -1.65, y1: 2.2 };
const CELL = 0.07;
const TOP = 1.95;
const IMG_X = -2.5;
const OUT_X = 2.5 - N * CELL;
const MID_Y = TOP - (N * CELL) / 2;
const K_CELL = 0.24;
const P_CELL = 0.3;
const P_TOP = -0.4;
const P_MID = P_TOP - 1.5 * P_CELL;
const SCAN_SECONDS = 5;
const STROKE_HALF_WIDTH = 0.8;

const IMG_L: GridLayout = { x: IMG_X, y: TOP, cell: CELL, cols: N, rows: N };
const K_L: GridLayout = { x: -1.5 * K_CELL, y: MID_Y + 1.5 * K_CELL, cell: K_CELL, cols: 3, rows: 3, gap: 0.1, radius: 2 };
const PATCH_L: GridLayout = { x: -2.55, y: P_TOP, cell: P_CELL, cols: 3, rows: 3, gap: 0.1, radius: 2 };
const PK_L: GridLayout = { ...PATCH_L, x: -1.15 };
const PROD_L: GridLayout = { ...PATCH_L, x: 0.25 };
const SUM_X = 2.05;

const clamp = (v: number, a: number, b: number) => Math.min(b, Math.max(a, v));
const fmt = (v: number, d: number) => {
	const s = Math.abs(v).toFixed(d);
	return v < 0 && Number(s) !== 0 ? '−' + s : s;
};
const signTone = (v: number): Tone => (v > 0.05 ? 'data' : v < -0.05 ? 'neg' : 'muted');

export const createConvolutionScene: SceneFactory = () => {
	let ctx: SceneContext;
	let camera: Camera2D;
	let quads: Quads;
	let lines: Lines;

	// params
	let imageId = 0;
	let kernelId = 3;
	let stride = 1;
	let pad = 0;
	let calc = 1;
	let scenario = 0;
	let paramsSeen = false;
	let hl: Tone | null = null;

	// state
	const image: Grid = makeGrid(N, N);
	const kernel: Grid = makeGrid(3, 3);
	let out: Grid = makeGrid(1, 1);
	let norm = 1;
	let scan = 1;
	let playing = false;
	let curRow = 4;
	let curCol = 1;
	let hoverImg: { col: number; row: number } | null = null;
	let hoverOut: { col: number; row: number } | null = null;
	let drawing = false;
	let last: [number, number] = [0, 0];
	let calcOp = 1;
	let outDirty = true;
	let dirty = true;
	let viewSig = '';
	let lastSize = '';
	let lastSum = NaN;

	const mul = (t: Tone) => (hl === null || hl === t ? 1 : 0.3);
	const mulOut = () => (hl === null || hl === 'data' || hl === 'neg' ? 1 : 0.3);

	function loadImage() {
		image.data.set(IMAGES[clamp(imageId, 0, IMAGES.length - 1)].make());
		outDirty = true;
	}

	function loadKernel() {
		kernel.data.set(KERNELS[clamp(kernelId, 0, KERNELS.length - 1)].k);
		outDirty = true;
	}

	function outLayout(): GridLayout {
		return {
			x: OUT_X + ((N - out.w) / 2) * CELL,
			y: TOP - ((N - out.h) / 2) * CELL,
			cell: CELL,
			cols: out.w,
			rows: out.h
		};
	}

	function reachedCount(): number {
		const total = out.w * out.h;
		return scan >= 1 ? total : Math.min(total, Math.round(scan * total));
	}

	function recompute() {
		outDirty = false;
		out = conv2d(image, kernel, { stride, pad });
		let m = 0;
		for (let i = 0; i < out.data.length; i++) m = Math.max(m, Math.abs(out.data[i]));
		norm = Math.max(0.5, m);
		curRow = clamp(curRow, 0, out.h - 1);
		curCol = clamp(curCol, 0, out.w - 1);
		const size = `${out.w} × ${out.h}`;
		if (size !== lastSize) {
			lastSize = size;
			ctx.emit('size', size);
		}
		dirty = true;
	}

	function setPosition(row: number, col: number) {
		row = clamp(row, 0, out.h - 1);
		col = clamp(col, 0, out.w - 1);
		if (row !== curRow || col !== curCol) {
			curRow = row;
			curCol = col;
			dirty = true;
		}
	}

	/** Window position centered nearest to image pixel (px, py). */
	function positionFromPixel(col: number, row: number) {
		setPosition(Math.round((row - 1 + pad) / stride), Math.round((col - 1 + pad) / stride));
	}

	function rect(n: number, x0: number, y0: number, x1: number, y1: number, width: number, color: [number, number, number, number]) {
		return lines.setPolyline(
			n,
			[
				[x0, y0],
				[x1, y0],
				[x1, y1],
				[x0, y1],
				[x0, y0]
			],
			{ width, color }
		);
	}

	function arrow(n: number, x0: number, x1: number, y: number, color: [number, number, number, number]) {
		lines.set(n++, { a: [x0, y], b: [x1, y], width: 1.2, color });
		lines.set(n++, { a: [x1 - 0.05, y + 0.04], b: [x1, y], width: 1.2, color });
		lines.set(n++, { a: [x1 - 0.05, y - 0.04], b: [x1, y], width: 1.2, color });
		return n;
	}

	function rebuild() {
		dirty = false;
		const v = ctx.view;
		const labels: Label[] = [];
		const mImg = mul('text');
		const mK = mul('param');
		const mO = mulOut();
		const outL = outLayout();
		const count = reachedCount();
		const cop = calcOp;

		// image
		let q = writeGrid(quads, 0, image.data, IMG_L, (val) => tone('text', 0.08 + 0.7 * val, mImg));
		if (pad === 1) {
			const padL: GridLayout = { x: IMG_X - CELL, y: TOP + CELL, cell: CELL, cols: N + 2, rows: N + 2 };
			const s = CELL * 0.92;
			for (let r = 0; r < N + 2; r++) {
				for (let c = 0; c < N + 2; c++) {
					if (r > 0 && r < N + 1 && c > 0 && c < N + 1) continue;
					const [x, y] = cellCenter(padL, c, r);
					quads.set(q++, { x, y, w: s, h: s, color: tone('muted', 0.3, 0.7 * mImg) });
				}
			}
		}
		// output
		q = writeGrid(quads, q, out.data, outL, (val, i) =>
			i < count ? diverging(val / norm, 0.6, mO) : tone('text', 0.05, mO)
		);
		// kernel
		const kmax = Math.max(1, ...Array.from(kernel.data, Math.abs));
		q = writeGrid(quads, q, kernel.data, K_L, (w) => tone('param', 0.08 + 0.45 * (Math.abs(w) / kmax), mK));

		// the current patch and its arithmetic
		const opts = { stride, pad };
		const patch = patchAt(image, kernel, curRow, curCol, opts);
		const prod = new Float32Array(9);
		let sum = 0;
		let pmax = 0.5;
		for (let i = 0; i < 9; i++) {
			prod[i] = patch.data[i] * kernel.data[i];
			sum += prod[i];
			pmax = Math.max(pmax, Math.abs(prod[i]));
		}
		if (cop > 0.01) {
			q = writeGrid(quads, q, patch.data, PATCH_L, (val) => tone('text', 0.04 + 0.18 * val, cop * mImg));
			q = writeGrid(quads, q, kernel.data, PK_L, (w) => tone('param', 0.08 + 0.45 * (Math.abs(w) / kmax), cop * mK));
			q = writeGrid(quads, q, prod, PROD_L, (val) => diverging(val / pmax, 0.35, cop * mO));
		}
		quads.count = q;
		quads.upload();

		// lines
		let n = 0;
		const frameC = tone('muted', 1, 0.6);
		n = rect(n, IMG_X - 0.02, TOP + 0.02, IMG_X + N * CELL + 0.02, TOP - N * CELL - 0.02, 1, frameC);
		n = rect(n, OUT_X - 0.02, TOP + 0.02, OUT_X + N * CELL + 0.02, TOP - N * CELL - 0.02, 1, frameC);
		const arrowC = tone('muted', 1, 0.7);
		n = arrow(n, IMG_X + N * CELL + 0.08, K_L.x - 0.08, MID_Y, arrowC);
		n = arrow(n, 1.5 * K_CELL + 0.08, OUT_X - 0.08, MID_Y, arrowC);
		if (cop > 0.01) {
			n = lines.setPolyline(
				n,
				[
					[-2.55, 0.02],
					[2.55, 0.02]
				],
				{ width: 1, color: tone('muted', 1, 0.35 * cop) }
			);
		}
		// window in the image and matching output cell
		if (calc >= 0.5) {
			const wc = tone('param', 1.4, 0.95 * mK);
			const wx = IMG_X + (curCol * stride - pad) * CELL;
			const wy = TOP - (curRow * stride - pad) * CELL;
			const halo: [number, number, number, number] = [0, 0, 0, 0.85 * mK];
			n = rect(n, wx, wy, wx + 3 * CELL, wy - 3 * CELL, 5, halo);
			n = rect(n, wx, wy, wx + 3 * CELL, wy - 3 * CELL, 2, wc);
			const ox = outL.x + curCol * CELL;
			const oy = outL.y - curRow * CELL;
			n = rect(n, ox, oy, ox + CELL, oy - CELL, 5, halo);
			n = rect(n, ox, oy, ox + CELL, oy - CELL, 2, wc);
		}
		// hovered pixel (step 1 reads values off the image)
		if (hoverImg && calc < 0.5) {
			const px = IMG_X + hoverImg.col * CELL;
			const py = TOP - hoverImg.row * CELL;
			n = rect(n, px, py, px + CELL, py - CELL, 1.5, tone('text', 0.9, 0.7 * mImg));
		}
		lines.count = n;
		lines.upload();

		// labels
		const L = (id: string, wxp: number, wyp: number, text: string, o: Partial<Label> = {}) => {
			const [x, y] = v.project(wxp, wyp);
			labels.push({ id, x, y, text, ...o });
		};
		L('cap-image', IMG_X + (N * CELL) / 2, TOP + 0.14, 'image', { tone: 'muted', opacity: mImg });
		L('cap-out', OUT_X + (N * CELL) / 2, TOP + 0.14, 'output', { tone: 'muted', opacity: mO });
		L('cap-kernel', 0, K_L.y + 0.14, 'kernel', { tone: 'muted', opacity: mK });
		L('hint', 0, K_L.y - 3 * K_CELL - 0.14, 'click a weight to edit', { tone: 'muted', size: 11, opacity: mK });
		for (let i = 0; i < 9; i++) {
			const w = kernel.data[i];
			const [cx, cy] = cellCenter(K_L, i % 3, Math.floor(i / 3));
			L(`kw${i}`, cx, cy, String(w), { tone: signTone(w), mono: true, size: 13, opacity: mK });
		}
		if (hoverImg && calc < 0.5) {
			const val = image.data[hoverImg.row * N + hoverImg.col];
			L('hover-px', IMG_X + (N * CELL) / 2, TOP - N * CELL - 0.15, `row ${hoverImg.row + 1}, column ${hoverImg.col + 1}: ${val.toFixed(2)}`, {
				tone: 'text',
				mono: true,
				size: 12,
				opacity: mImg
			});
		}
		if (cop > 0.01) {
			const co = cop;
			L('cap-patch', PATCH_L.x + 1.5 * P_CELL, -0.25, 'patch', { tone: 'muted', opacity: co * mImg });
			L('cap-pk', PK_L.x + 1.5 * P_CELL, -0.25, 'kernel', { tone: 'muted', opacity: co * mK });
			L('cap-prod', PROD_L.x + 1.5 * P_CELL, -0.25, 'products', { tone: 'muted', opacity: co * mO });
			L('cap-sum', SUM_X, -0.25, 'sum', { tone: 'muted', opacity: co * mO });
			L('op-x', -1.4, P_MID, '×', { tone: 'muted', size: 16, opacity: co });
			L('op-eq', 0, P_MID, '=', { tone: 'muted', size: 16, opacity: co });
			L('op-sigma', 1.43, P_MID, 'Σ', { tone: 'muted', size: 16, opacity: co });
			for (let i = 0; i < 9; i++) {
				const c = i % 3;
				const r = Math.floor(i / 3);
				const [ax, ay] = cellCenter(PATCH_L, c, r);
				L(`pv${i}`, ax, ay, patch.data[i].toFixed(1), { tone: 'text', mono: true, size: 11, opacity: co * mImg });
				const [bx, by] = cellCenter(PK_L, c, r);
				L(`pw${i}`, bx, by, String(kernel.data[i]), { tone: signTone(kernel.data[i]), mono: true, size: 11, opacity: co * mK });
				const [dx, dy] = cellCenter(PROD_L, c, r);
				L(`pp${i}`, dx, dy, fmt(prod[i], 1), { tone: 'text', mono: true, size: 11, opacity: co * mO });
			}
			L('sum-val', SUM_X, P_MID, fmt(sum, 1), { tone: signTone(sum), mono: true, size: 16, opacity: co * mO });
			L('sum-note', SUM_X, P_MID - 0.3, 'goes into the output', { tone: 'muted', size: 11, opacity: co * mO });
		}
		ctx.labels.set(labels);

		const rounded = Math.round(sum * 10) / 10;
		if (rounded !== lastSum) {
			lastSum = rounded;
			ctx.emit('sum', rounded);
		}
	}

	const scene: Scene = {
		init(c) {
			ctx = c;
			camera = new Camera2D();
			quads = new Quads(c, { capacity: 1500, blend: 'alpha' });
			lines = new Lines(c, { capacity: 80, blend: 'alpha' });
			loadImage();
			loadKernel();
			recompute();
		},
		setParams(p) {
			const prev = { imageId, kernelId, stride, pad, scenario };
			imageId = Math.round(p.image ?? imageId);
			kernelId = Math.round(p.kernel ?? kernelId);
			stride = clamp(Math.round(p.stride ?? stride), 1, 2);
			pad = (p.pad ?? pad) >= 0.5 ? 1 : 0;
			const newCalc = (p.calc ?? calc) >= 0.5 ? 1 : 0;
			scenario = Math.round(p.scenario ?? scenario);
			const scenarioChanged = prev.scenario !== scenario;
			if (!paramsSeen || scenarioChanged || prev.imageId !== imageId) {
				loadImage();
				drawing = false;
			}
			if (!paramsSeen || scenarioChanged || prev.kernelId !== kernelId) loadKernel();
			if (newCalc !== calc || !paramsSeen) {
				calc = newCalc;
				if (!paramsSeen || ctx.reducedMotion) calcOp = calc;
				dirty = true;
			}
			if (!paramsSeen || scenarioChanged) {
				playing = false;
				scan = clamp(p.scan ?? scan, 0, 1);
				dirty = true;
			} else if (!playing && p.scan !== undefined && p.scan !== scan) {
				scan = clamp(p.scan, 0, 1);
				dirty = true;
			}
			if (prev.stride !== stride || prev.pad !== pad) outDirty = true;
			paramsSeen = true;
		},
		update(fr) {
			const v = ctx.view;
			fitCamera(camera, v, WORLD, lessonInsets(v));
			v.setViewProj(camera.matrix(v.aspect));

			if (outDirty) recompute();
			if (playing) {
				scan += fr.dt / SCAN_SECONDS;
				const total = out.w * out.h;
				if (scan >= 1) {
					scan = 1;
					playing = false;
					ctx.emit('param:scan', 1);
				}
				const idx = Math.max(0, reachedCount() - 1);
				curRow = Math.min(out.h - 1, Math.floor(idx / out.w));
				curCol = idx % out.w;
				if (total > 0) dirty = true;
			}
			if (Math.abs(calcOp - calc) > 0.001) {
				calcOp += (calc - calcOp) * Math.min(1, fr.dt * 8 + 0.02);
				if (Math.abs(calcOp - calc) <= 0.01) calcOp = calc;
				dirty = true;
			}
			const sig = `${v.width}x${v.height}`;
			if (sig !== viewSig) {
				viewSig = sig;
				dirty = true;
			}
			if (dirty) rebuild();
		},
		draw(pass) {
			quads.draw(pass);
			lines.draw(pass);
		},
		command(id) {
			if (id === 'play') {
				if (ctx.reducedMotion) {
					scan = 1;
					playing = false;
					ctx.emit('param:scan', 1);
					// The window ends on the last output cell, as it does when the scan plays out.
					curRow = Math.max(0, out.h - 1);
					curCol = Math.max(0, out.w - 1);
				} else {
					scan = 0;
					playing = true;
					ctx.emit('param:scan', 0);
				}
				dirty = true;
			} else if (id === 'clear') {
				image.data.fill(0);
				outDirty = true;
			}
		},
		highlight(t) {
			hl = t;
			dirty = true;
		},
		pointer(e: ScenePointer) {
			if (e.type === 'leave') {
				drawing = false;
				if (hoverImg || hoverOut) dirty = true;
				hoverImg = null;
				hoverOut = null;
				return;
			}
			if (e.type === 'up') {
				drawing = false;
				return;
			}
			const [wx, wy] = ctx.view.unproject(e.x, e.y);
			if (e.type === 'down') {
				const kc = cellAt(K_L, wx, wy);
				if (kc) {
					const i = kc.row * 3 + kc.col;
					const w = kernel.data[i];
					kernel.data[i] = w >= 2 ? -2 : w + 1;
					outDirty = true;
					return;
				}
				if (cellAt(IMG_L, wx, wy)) {
					drawing = true;
					last = toPixel(IMG_L, wx, wy);
					strokeSegment(image.data, N, last[0], last[1], last[0], last[1], STROKE_HALF_WIDTH);
					outDirty = true;
				}
			} else if (drawing) {
				const cur = toPixel(IMG_L, wx, wy);
				strokeSegment(image.data, N, last[0], last[1], cur[0], cur[1], STROKE_HALF_WIDTH);
				last = cur;
				outDirty = true;
			}
			const hi = cellAt(IMG_L, wx, wy);
			const ho = cellAt(outLayout(), wx, wy);
			if (hi?.col !== hoverImg?.col || hi?.row !== hoverImg?.row || ho?.col !== hoverOut?.col || ho?.row !== hoverOut?.row) {
				dirty = true;
			}
			hoverImg = hi;
			hoverOut = ho;
			if (!playing) {
				if (ho) setPosition(ho.row, ho.col);
				else if (hi) positionFromPixel(hi.col, hi.row);
			}
		},
		dispose() {
			quads.destroy();
			lines.destroy();
		}
	};
	return scene;
};
