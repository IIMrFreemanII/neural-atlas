import { Camera2D } from '../../../gpu/camera.ts';
import { tone } from '../../../gpu/colors.ts';
import { Lines } from '../../../gpu/primitives/lines.ts';
import { Points } from '../../../gpu/primitives/points.ts';
import { Quads } from '../../../gpu/primitives/quads.ts';
import type { Label, Scene, SceneContext, SceneFactory, ScenePointer, Tone } from '../../../gpu/types.ts';
import { decode, encode } from '../../../ml/autoencoder.ts';
import { strokeSegment } from '../../../ml/raster.ts';
import { fitCamera, lessonInsets, type WorldRect } from '../../fit.ts';
import { CLASS_NAMES, SAMPLES } from '../cnn-features/data.ts';
import { toPixel, writeGrid, type GridLayout } from '../shared/grid.ts';
import { mapPoints, nearestLabel } from './data.ts';
import { WEIGHTS } from './weights.ts';

// The layout is budgeted in CSS px at the reference stage (about 118 px per world unit) and converted to world units.
const S = 118;
const px = (n: number) => n / S;

const IMG = 16;
const CELL_WIDE = 10; // drawing and rebuilt with the code box: 160 px
const CELL_NARROW = 7; // with the map open: 112 px
const FUNNEL_WIDE = 96; // length of encoder and decoder
const FUNNEL_NARROW = 56;
const TIP = 12; // half height of a funnel's narrow end
const GAP = 8;
const BOX_W = 70;
const BOX_H = 56;
const MAP_W = 300;
const TILES = 9;
const TILE_GAP = 3;

/** Left edge of the whole layout, fixed so the left side never moves. */
const X0 = -(2 * IMG * CELL_NARROW + 2 * FUNNEL_NARROW + 4 * GAP + MAP_W) / 2 - 30;

const TONES: Tone[] = ['data', 'param', 'grad', 'neg'];
const PLURAL = ['circles', 'squares', 'triangles', 'crosses'];

const clamp = (v: number, a: number, b: number) => Math.min(b, Math.max(a, v));
const fmt = (v: number) => {
	const s = v.toFixed(2);
	return s === '-0.00' ? '0.00' : s.replace('-', '−');
};

type Rgba = [number, number, number, number];

/** Rounded rectangle outline as a closed polyline (world units from px inputs). */
function roundedRect(cx: number, cy: number, w: number, h: number, r: number): Array<[number, number]> {
	const pts: Array<[number, number]> = [];
	const corners: Array<[number, number, number]> = [
		[cx + w / 2 - r, cy + h / 2 - r, 0],
		[cx - w / 2 + r, cy + h / 2 - r, 1],
		[cx - w / 2 + r, cy - h / 2 + r, 2],
		[cx + w / 2 - r, cy - h / 2 + r, 3]
	];
	for (const [x, y, q] of corners) {
		for (let i = 0; i <= 4; i++) {
			const a = ((q + i / 4) * Math.PI) / 2;
			pts.push([px(x + r * Math.cos(a)), px(y + r * Math.sin(a))]);
		}
	}
	pts.push(pts[0]);
	return pts;
}

export const createAutoencoderScene: SceneFactory = () => {
	let ctx: SceneContext;
	let camera: Camera2D;
	let quads: Quads;
	let lines: Lines;
	let points: Points;

	// params
	let sample = 1;
	let show = 3;
	let scenario = 1;
	let paramsSeen = false;
	let hl: Tone | null = null;

	// state
	const image = new Float32Array(IMG * IMG);
	let source: 'drawing' | 'point' = 'drawing';
	let code: [number, number] = [0, 0];
	const rebuilt = new Float32Array(IMG * IMG);
	let empty = false;
	let stroking = false;
	let dragging = false;
	let last: [number, number] = [0, 0];
	let lastNearest = '';
	let needsDraw = true;
	let snap = true;

	// what is on screen
	const dImg = new Float32Array(IMG * IMG);
	const dReb = new Float32Array(IMG * IMG);
	let dCode: [number, number] = [0, 0];
	let cw = BOX_W; // eased width of the center block
	let srcA = 1;
	let codeA = 1;
	let dotsA = 0;
	let tilesA = 0;
	let labA = 0;

	const dataPts = mapPoints(WEIGHTS);
	const centroids = (() => {
		const sum = CLASS_NAMES.map(() => [0, 0, 0]);
		for (const p of dataPts) {
			sum[p.label][0] += p.x;
			sum[p.label][1] += p.y;
			sum[p.label][2]++;
		}
		return sum.map(([x, y, n]) => [x / Math.max(1, n), y / Math.max(1, n)] as [number, number]);
	})();
	/** Where each class name goes: above the cluster (or below it for clusters in the lower half), in code units. */
	const labelAt = centroids.map(([cx, cy], c) => {
		const ys = dataPts
			.filter((p) => p.label === c)
			.map((p) => p.y)
			.sort((a, b) => a - b);
		const edge = cy >= 0 ? ys[Math.floor(ys.length * 0.95)] : ys[Math.floor(ys.length * 0.05)];
		return { x: cx, y: edge ?? cy, dir: cy >= 0 ? 1 : -1 };
	});
	const tiles: Float32Array[] = [];

	const mul = (t: Tone) => (hl === null || hl === t ? 1 : 0.3);

	/** Sizes (px) that shrink as the map opens. */
	function dims(o = openness()) {
		const cell = CELL_WIDE + (CELL_NARROW - CELL_WIDE) * o;
		return { cell, side: IMG * cell, funnel: FUNNEL_WIDE + (FUNNEL_NARROW - FUNNEL_WIDE) * o };
	}
	const captionY = (o = openness()) => -(dims(o).side / 2 + 18);
	const gridImg = (o = openness()): GridLayout => ({
		x: px(X0),
		y: px(dims(o).side / 2),
		cell: px(dims(o).cell),
		cols: IMG,
		rows: IMG,
		gap: 0.06
	});

	/** Column positions (px) for a center width. */
	function layout(w: number, o = openness()) {
		const { side, funnel } = dims(o);
		const encX0 = X0 + side + GAP;
		const encX1 = encX0 + funnel;
		const mapL = encX1 + GAP;
		const mapR = mapL + w;
		const decX0 = mapR + GAP;
		const decX1 = decX0 + funnel;
		const rebL = decX1 + GAP;
		return { encX0, encX1, mapL, mapR, decX0, decX1, rebL, mapCx: (mapL + mapR) / 2 };
	}
	const gridReb = (): GridLayout => ({ ...gridImg(), x: px(layout(cw).rebL) });

	const targetWidth = () => (show >= 2 ? MAP_W : BOX_W);
	const openness = () => clamp((cw - BOX_W) / (MAP_W - BOX_W), 0, 1);

	function viewRect(): WorldRect {
		const l = layout(cw);
		const half = px(128 + 68 * openness());
		return { x0: px(X0) - 0.02, x1: px(l.rebL + dims().side) + 0.02, y0: -half, y1: half };
	}

	function loadSample() {
		image.set(SAMPLES[clamp(sample - 1, 0, SAMPLES.length - 1)]);
	}

	function setCode(c: [number, number], emit: boolean) {
		code = [clamp(c[0], -1, 1), clamp(c[1], -1, 1)];
		decode(WEIGHTS, code, rebuilt);
		if (emit) {
			ctx.emit('param:zx', code[0]);
			ctx.emit('param:zy', code[1]);
		}
		needsDraw = true;
		publish();
	}

	/** Read the code off the drawing (source becomes 'drawing') and tell the sliders. */
	function fromDrawing() {
		source = 'drawing';
		let total = 0;
		for (const v of image) total += v;
		empty = total < 2;
		setCode(encode(WEIGHTS, image), true);
	}

	function publish() {
		const name = source === 'drawing' && empty ? '—' : CLASS_NAMES[nearestLabel(dataPts, code[0], code[1])];
		if (name !== lastNearest) {
			lastNearest = name;
			ctx.emit('nearest', name);
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

	function easeNum(v: number, target: number, k: number, eps: number): [number, boolean] {
		const dv = target - v;
		return Math.abs(dv) < eps ? [target, false] : [v + dv * k, true];
	}

	function paint(wx: number, wy: number, fresh: boolean) {
		const [x, y] = toPixel(gridImg(), wx, wy);
		if (fresh) last = [x, y];
		strokeSegment(image, IMG, last[0], last[1], x, y, 0.7);
		last = [x, y];
		fromDrawing();
	}

	/** Pointer (px, stage-centered, y up) to a code, clamped to the map. */
	function codeAt(wx: number, wy: number): [number, number] {
		const l = layout(MAP_W, 1);
		const half = MAP_W / 2;
		return [clamp((wx * S - l.mapCx) / half, -1, 1), clamp((wy * S) / half, -1, 1)];
	}

	function drawScene() {
		needsDraw = false;
		const mT = mul('text');
		const mP = mul('param');
		const mD = mul('data');
		const l = layout(cw);
		const open = openness();
		const { side } = dims();
		const sh = side / 2;
		const boxA = 1 - open;
		const mapA = open;
		const half = cw / 2;
		const codeAll = mD * codeA;
		const imgAlpha = mT * srcA;
		const encAlpha = mP * srcA;

		let n = 0;
		n = writeGrid(quads, n, dImg, gridImg(), (v) => tone('text', 0.08 + 0.7 * v, imgAlpha));
		n = writeGrid(quads, n, dReb, gridReb(), (v) => tone('data', 0.08 + 0.72 * v, codeAll));

		// decoded tiles over the map
		if (tilesA > 0.01 && mapA > 0.01) {
			const pitch = cw / TILES;
			const cell = (pitch - TILE_GAP) / IMG;
			const alpha = tilesA * mD * mapA;
			for (let j = 0; j < TILES; j++) {
				for (let i = 0; i < TILES; i++) {
					const g: GridLayout = {
						x: px(l.mapL + i * pitch + TILE_GAP / 2),
						y: px(half - j * pitch - TILE_GAP / 2),
						cell: px(cell),
						cols: IMG,
						rows: IMG,
						gap: 0.05
					};
					n = writeGrid(quads, n, tiles[j * TILES + i], g, (v) => tone('data', 0.06 + 0.6 * v, alpha));
				}
			}
		}
		quads.count = n;
		quads.upload();

		// lines
		let m = 0;
		const seg = (ax: number, ay: number, bx: number, by: number, width: number, color: Rgba) => {
			lines.set(m++, { a: [px(ax), px(ay)], b: [px(bx), px(by)], width, color });
		};
		m = lines.setPolyline(
			m,
			[[px(X0), px(sh)], [px(X0 + side), px(sh)], [px(X0 + side), -px(sh)], [px(X0), -px(sh)], [px(X0), px(sh)]],
			{ width: 1, color: tone('muted', 1, 0.5 * imgAlpha) }
		);
		const reb = gridReb();
		m = lines.setPolyline(
			m,
			[[reb.x, px(sh)], [reb.x + px(side), px(sh)], [reb.x + px(side), -px(sh)], [reb.x, -px(sh)], [reb.x, px(sh)]],
			{ width: 1, color: tone('muted', 1, 0.5 * codeAll) }
		);
		const funnel = (xa: number, ha: number, xb: number, hb: number, color: Rgba) => {
			m = lines.setPolyline(m, [[px(xa), px(ha)], [px(xb), px(hb)], [px(xb), px(-hb)], [px(xa), px(-ha)], [px(xa), px(ha)]], {
				width: 1.5,
				color
			});
		};
		funnel(l.encX0, sh, l.encX1, TIP, tone('param', 0.9, 0.8 * encAlpha));
		funnel(l.decX1, sh, l.decX0, TIP, tone('param', 0.9, 0.8 * mP));

		if (boxA > 0.01) {
			m = lines.setPolyline(m, roundedRect(l.mapCx, 0, BOX_W, BOX_H, 10), {
				width: 1.5,
				color: tone('data', 0.9, 0.7 * boxA * codeAll)
			});
		}
		const dx = l.mapCx + dCode[0] * half;
		const dy = dCode[1] * half;
		if (mapA > 0.01) {
			m = lines.setPolyline(
				m,
				[[px(l.mapL), px(half)], [px(l.mapR), px(half)], [px(l.mapR), -px(half)], [px(l.mapL), -px(half)], [px(l.mapL), px(half)]],
				{ width: 1, color: tone('muted', 1, 0.7 * mapA) }
			);
			seg(l.mapL, 0, l.mapR, 0, 1, tone('muted', 1, 0.25 * mapA));
			seg(l.mapCx, -half, l.mapCx, half, 1, tone('muted', 1, 0.25 * mapA));
			seg(l.encX1, 0, dx, dy, 1, tone('data', 1, 0.5 * mapA * codeAll));
			seg(dx, dy, l.decX0, 0, 1, tone('data', 1, 0.5 * mapA * codeAll));
		}
		lines.count = m;
		lines.upload();

		// points: map dots, then the code
		let k = 0;
		if (dotsA > 0.01 && mapA > 0.01) {
			const dim = hl === null ? 1 : 0.3;
			const a = dotsA * mapA * dim;
			for (const p of dataPts) {
				points.set(k++, {
					x: px(l.mapCx + p.x * half),
					y: px(p.y * half),
					radius: 2.5,
					color: tone(TONES[p.label], 1, a)
				});
			}
		}
		if (mapA > 0.01) {
			points.set(k++, { x: px(dx), y: px(dy), radius: 8, ring: 2, color: tone('text', 1, 0.95 * mapA * codeAll) });
			points.set(k++, { x: px(dx), y: px(dy), radius: 4, softness: 1, color: tone('data', 1, mapA * codeAll) });
		}
		points.count = k;
		points.upload();
	}

	function buildLabels(): Label[] {
		const v = ctx.view;
		const out: Label[] = [];
		const l = layout(cw);
		const open = openness();
		const boxA = 1 - open;
		const mapA = open;
		const mT = mul('text');
		const mP = mul('param');
		const mD = mul('data');
		const at = (x: number, y: number) => v.project(px(x), px(y));
		const cap = (id: string, text: string, cx: number, y: number, a: number, size = 12) => {
			if (a < 0.01) return;
			const [x, sy] = at(cx, y);
			out.push({ id, x, y: sy, text, tone: 'muted', opacity: a, size });
		};
		const codeAll = mD * codeA;
		cap('capImg', source === 'point' ? 'not used' : 'drawing', X0 + dims().side / 2, captionY(), mT * srcA);
		cap('capEnc', 'encoder', (l.encX0 + l.encX1) / 2, captionY(), mP * srcA);
		cap('subEnc', '256 → 32 → 2', (l.encX0 + l.encX1) / 2, captionY() - 16, mP * srcA * 0.8 * boxA, 10);
		cap('capDec', 'decoder', (l.decX0 + l.decX1) / 2, captionY(), mP);
		cap('subDec', '2 → 32 → 256', (l.decX0 + l.decX1) / 2, captionY() - 16, mP * 0.8 * boxA, 10);
		cap('capReb', 'rebuilt', l.rebL + dims().side / 2, captionY(), mD);
		cap('capCode', 'code', l.mapCx, captionY(), boxA * mD);
		cap('capMap', 'latent space', l.mapCx, -cw / 2 - 28, mapA * mD);

		if (boxA > 0.01) {
			const a = boxA * codeAll;
			const [x, y] = at(l.mapCx, 0);
			out.push({ id: 'bx', x, y: y - 11, text: `x ${fmt(code[0])}`, tone: 'data', mono: true, opacity: a });
			out.push({ id: 'by', x, y: y + 11, text: `y ${fmt(code[1])}`, tone: 'data', mono: true, opacity: a });
		}
		if (mapA > 0.01) {
			const half = cw / 2;
			const [px0, py0] = at(l.mapCx + dCode[0] * half, dCode[1] * half);
			out.push({
				id: 'codeLbl',
				x: px0,
				y: py0 + 19,
				text: `(${fmt(code[0])}, ${fmt(code[1])})`,
				tone: 'text',
				mono: true,
				size: 11,
				opacity: mapA * codeAll * (1 - labA)
			});
			if (labA > 0.01) {
				const dim = hl === null ? 1 : 0.3;
				for (let c = 0; c < CLASS_NAMES.length; c++) {
					// just beyond the cluster, above it (below for the lower ones), kept near the map
					const { x: gx, y: gy, dir } = labelAt[c];
					const lx = clamp(gx * half, -half + 34, half - 34);
					const near = gy * half + dir * 12;
					const ly = Math.abs(near) > half - 14 ? dir * (half + 11) : near;
					const [x, y] = at(l.mapCx + lx, ly);
					out.push({ id: `cls${c}`, x, y, text: PLURAL[c], tone: TONES[c], size: 11, opacity: labA * mapA * dim });
				}
			}
		}
		return out;
	}

	const scene: Scene = {
		init(c) {
			ctx = c;
			camera = new Camera2D();
			quads = new Quads(c, { capacity: 22528 });
			lines = new Lines(c, { capacity: 300, blend: 'alpha' });
			points = new Points(c, { capacity: 820 });
			for (let j = 0; j < TILES; j++) {
				for (let i = 0; i < TILES; i++) {
					const zx = -1 + ((i + 0.5) * 2) / TILES;
					const zy = 1 - ((j + 0.5) * 2) / TILES;
					tiles.push(decode(WEIGHTS, [zx, zy]));
				}
			}
			loadSample();
		},
		setParams(p) {
			const prevSample = sample;
			const prevScenario = scenario;
			sample = Math.round(p.sample ?? sample);
			show = Math.round(p.show ?? show);
			scenario = Math.round(p.scenario ?? scenario);
			const zx = p.zx ?? code[0];
			const zy = p.zy ?? code[1];
			if (!paramsSeen || prevScenario !== scenario) {
				loadSample();
				if (Math.round(p.pin ?? 0) === 1) {
					source = 'point';
					empty = false;
					setCode([zx, zy], false);
				} else fromDrawing();
			} else if (prevSample !== sample) {
				loadSample();
				fromDrawing();
			} else if (Math.abs(zx - code[0]) > 1e-6 || Math.abs(zy - code[1]) > 1e-6) {
				source = 'point';
				setCode([zx, zy], false);
			}
			paramsSeen = true;
			needsDraw = true;
		},
		update(fr) {
			const v = ctx.view;
			const instant = ctx.reducedMotion || snap;
			let moving = false;

			const kw = instant ? 1 : 1 - Math.exp(-fr.dt / 0.18);
			const [w, wMoving] = easeNum(cw, targetWidth(), kw, 0.05);
			cw = w;
			moving = wMoving;

			fitCamera(camera, v, viewRect(), lessonInsets(v));
			v.setViewProj(camera.matrix(v.aspect));

			const kImg = instant ? 1 : 1 - Math.exp(-fr.dt / 0.04);
			const kReb = instant ? 1 : 1 - Math.exp(-fr.dt / 0.06);
			moving = ease(dImg, image, kImg) || moving;
			moving = ease(dReb, rebuilt, kReb) || moving;

			if (dragging || instant) dCode = [code[0], code[1]];
			else {
				for (let i = 0; i < 2; i++) {
					const dv = code[i] - dCode[i];
					if (Math.abs(dv) < 1e-4) dCode[i] = code[i];
					else {
						dCode[i] += dv * kReb;
						moving = true;
					}
				}
			}

			const ka = instant ? 1 : 1 - Math.exp(-fr.dt / 0.09);
			const dead = source === 'drawing' && empty;
			let mv: boolean;
			[srcA, mv] = easeNum(srcA, source === 'point' ? 0.25 : 1, ka, 0.002);
			moving = mv || moving;
			[codeA, mv] = easeNum(codeA, dead ? 0.3 : 1, ka, 0.002);
			moving = mv || moving;
			[dotsA, mv] = easeNum(dotsA, show >= 4 ? 0.15 : show >= 3 ? 0.75 : 0, ka, 0.002);
			moving = mv || moving;
			[tilesA, mv] = easeNum(tilesA, show >= 4 ? 1 : 0, ka, 0.002);
			moving = mv || moving;
			[labA, mv] = easeNum(labA, show >= 3 ? 1 : 0, ka, 0.002);
			moving = mv || moving;
			snap = false;

			if (moving || needsDraw) drawScene();
			ctx.labels.set(buildLabels());
		},
		draw(pass) {
			quads.draw(pass);
			lines.draw(pass);
			points.draw(pass);
		},
		command(id) {
			if (id === 'clear') {
				image.fill(0);
				fromDrawing();
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
				dragging = false;
				return;
			}
			const [wx, wy] = ctx.view.unproject(e.x, e.y);
			if (e.type === 'down') {
				const gImg = gridImg();
				const m = 0.5 * gImg.cell;
				const inGrid =
					wx >= gImg.x - m && wx <= gImg.x + IMG * gImg.cell + m && wy <= gImg.y + m && wy >= gImg.y - IMG * gImg.cell - m;
				if (inGrid) {
					stroking = true;
					paint(wx, wy, true);
					return;
				}
				if (show >= 2 && cw > MAP_W * 0.9) {
					const l = layout(cw);
					const sx = wx * S;
					const sy = wy * S;
					if (sx >= l.mapL && sx <= l.mapR && Math.abs(sy) <= cw / 2) {
						dragging = true;
						source = 'point';
						setCode(codeAt(wx, wy), true);
					}
				}
				return;
			}
			if (stroking) paint(wx, wy, false);
			else if (dragging) setCode(codeAt(wx, wy), true);
		},
		dispose() {
			quads.destroy();
			lines.destroy();
			points.destroy();
		}
	};
	return scene;
};
