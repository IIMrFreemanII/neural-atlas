import { Camera2D } from '../../../gpu/camera.ts';
import { paperFill, tone } from '../../../gpu/colors.ts';
import { Lines } from '../../../gpu/primitives/lines.ts';
import { Points } from '../../../gpu/primitives/points.ts';
import type { Label, Scene, SceneContext, SceneFactory, Tone } from '../../../gpu/types.ts';
import { evalPoly, fitPoly, polyMse } from '../../../ml/polyfit.ts';
import { fitCamera, lessonInsets, type WorldRect } from '../../fit.ts';
import { makeSample, truth, type Sample } from './data.ts';

const MAX_DEGREE = 12;
const N_TEST = 60;
const N_GHOSTS = 20;
const CURVE_SAMPLES = 240;
const Y_LIM = 1.15;
const ERR_LO = 0.001;
const ERR_HI = 100;

// World layout. The plot maps data x in [-1, 1] to [0, pw] and y to y * yScale.
// Alone, the plot is wide. With the chart beside it the pair is narrower and taller instead, which fills a
// stage that is wider than the pair but not tall enough for it: the plot gets narrower, both stretch in y.
const PW_ALONE = 3.4;
const PW_PAIR = 2.6;
const YS_PAIR = 1.5; // y stretch of the pair
const GAP = 0.85; // plot right edge to chart left, room for the error labels
const CW = 2.1 - 0.3;
const CHART_TAIL = 0.1;
const HALF_H = 1.38;

// Current layout, eased between the two above by the chart mix. Set at the top of every update.
let pw = PW_ALONE;
let yScale = 1;
let c0 = pw + GAP; // chart left

function fmtErr(e: number): string {
	if (!Number.isFinite(e) || e >= 1000) return 'over 1000';
	return e < 10 ? e.toFixed(3) : e.toFixed(1);
}
const clamp = (v: number, a: number, b: number) => Math.min(b, Math.max(a, v));
const px = (x: number) => ((x + 1) / 2) * pw;
const logT = (e: number) => clamp(Math.log10(Math.max(e, 1e-12) / ERR_LO) / Math.log10(ERR_HI / ERR_LO), 0, 1);
const cx = (d: number) => c0 + ((d - 1) / (MAX_DEGREE - 1)) * CW;
const cy = (e: number) => (-Y_LIM + 2 * Y_LIM * logT(e)) * yScale;

type Pt = [number, number];

export const createOverfittingScene: SceneFactory = () => {
	let ctx: SceneContext;
	let camera: Camera2D;
	let base: Lines;
	let ghostLines: Lines;
	let curveLines: Lines;
	let chartLines: Lines;
	let dots: Points;
	let chartDots: Points;

	// params
	let degree = 4;
	let n = 14;
	let noise = 0.15;
	let reg = 0;
	let fit = 1;
	let truthLine = 0;
	let showTest = 0;
	let chart = 0;
	let ghosts = 0;
	let scenario = 1;
	let seed = 1;
	let hl: Tone | null = null;

	// derived
	let lambda = 0;
	let train: Sample = makeSample(n, noise, seed);
	let test: Sample = makeSample(N_TEST, noise, 9001);
	let curveY = new Float64Array(CURVE_SAMPLES);
	const trainErrs = new Float64Array(MAX_DEGREE);
	const testErrs = new Float64Array(MAX_DEGREE);
	let ghostY: Float64Array<ArrayBuffer>[] = [];
	let fitKey = '';
	let ghostKey = '';
	let chartMix = 0;
	let dirty = true;
	let lastTrain = '';
	let lastTest = '';
	let lastLabels: Label[] = [];
	let lastW = 0;
	let lastH = 0;

	const xs = new Float64Array(CURVE_SAMPLES);
	for (let i = 0; i < CURVE_SAMPLES; i++) xs[i] = -1 + (2 * i) / (CURVE_SAMPLES - 1);

	const mul = (t: Tone) => (hl === null || hl === t ? 1 : 0.3);

	function curveOf(sample: Sample, deg: number): Float64Array<ArrayBuffer> {
		const m = fitPoly(sample.x, sample.y, deg, lambda);
		const ys = new Float64Array(CURVE_SAMPLES);
		for (let i = 0; i < CURVE_SAMPLES; i++) ys[i] = evalPoly(m, xs[i]);
		return ys;
	}

	function refit() {
		const key = `${degree}|${lambda}|${n}|${noise}|${seed}`;
		if (key !== fitKey) {
			fitKey = key;
			train = makeSample(n, noise, seed);
			test = makeSample(N_TEST, noise, 9001);
			for (let d = 1; d <= MAX_DEGREE; d++) {
				const m = fitPoly(train.x, train.y, d, lambda);
				trainErrs[d - 1] = polyMse(m, train.x, train.y);
				testErrs[d - 1] = polyMse(m, test.x, test.y);
			}
			curveY = curveOf(train, degree);
		}
		const gkey = `${degree}|${lambda}|${n}|${noise}`;
		if (ghosts >= 0.5 && gkey !== ghostKey) {
			ghostKey = gkey;
			ghostY = [];
			for (let g = 0; g < N_GHOSTS; g++) ghostY.push(curveOf(makeSample(n, noise, 100 + g), degree));
		}
	}

	/** Write a polyline of (data x, data y) samples clipped to |y| <= Y_LIM. Returns the next free segment index. */
	function clippedCurve(lines: Lines, start: number, curve: Float64Array, width: number, color: [number, number, number, number]) {
		let k = start;
		for (let i = 0; i + 1 < CURVE_SAMPLES; i++) {
			let x0 = xs[i];
			let y0 = curve[i];
			let x1 = xs[i + 1];
			let y1 = curve[i + 1];
			if (!Number.isFinite(y0) || !Number.isFinite(y1)) continue;
			if ((y0 > Y_LIM && y1 > Y_LIM) || (y0 < -Y_LIM && y1 < -Y_LIM)) continue;
			// Liang-Barsky against the y band
			let t0 = 0;
			let t1 = 1;
			const dy = y1 - y0;
			const dx = x1 - x0;
			if (dy !== 0) {
				const ta = (-Y_LIM - y0) / dy;
				const tb = (Y_LIM - y0) / dy;
				t0 = Math.max(t0, Math.min(ta, tb));
				t1 = Math.min(t1, Math.max(ta, tb));
			}
			if (t0 >= t1) continue;
			const nx0 = x0 + dx * t0;
			const ny0 = y0 + dy * t0;
			x1 = x0 + dx * t1;
			y1 = y0 + dy * t1;
			x0 = nx0;
			y0 = ny0;
			lines.set(k++, { a: [px(x0), y0 * yScale], b: [px(x1), y1 * yScale], width, color });
		}
		return k;
	}

	function rebuild(labels: Label[]) {
		const v = ctx.view;
		const mP = mul('param');
		const mD = mul('data');
		const mO = mul('ok');
		const mM = mul('muted');

		// plot frame, axes, true pattern
		let k = 0;
		const frameC = tone('muted', 1, 0.5);
		const axisC = tone('muted', 1, 0.25);
		const yl = Y_LIM * yScale;
		const corners: Pt[] = [
			[0, -yl],
			[pw, -yl],
			[pw, yl],
			[0, yl],
			[0, -yl]
		];
		k = base.setPolyline(k, corners, { width: 1, color: frameC });
		base.set(k++, { a: [pw / 2, -yl], b: [pw / 2, yl], width: 1, color: axisC });
		base.set(k++, { a: [0, 0], b: [pw, 0], width: 1, color: axisC });
		if (truthLine >= 0.5) {
			const pts: Pt[] = [];
			for (let i = 0; i < CURVE_SAMPLES; i++) pts.push([px(xs[i]), truth(xs[i]) * yScale]);
			k = base.setPolyline(k, pts, { width: 1.5, color: tone('muted', 1, 0.55 * mM) });
		}
		base.count = k;
		base.upload();

		// ghosts and the fitted curve
		let g = 0;
		if (ghosts >= 0.5) {
			const c = tone('param', 1, 0.15 * mP);
			for (const ys of ghostY) g = clippedCurve(ghostLines, g, ys, 1, c);
		}
		ghostLines.count = g;
		ghostLines.upload();

		let c = 0;
		if (fit >= 0.5) {
			c = clippedCurve(curveLines, 0, curveY, 2.5, tone('param', 1, mP));
		}
		curveLines.count = c;
		curveLines.upload();

		// points
		let d = 0;
		const inside = (y: number) => Math.abs(y) <= Y_LIM;
		for (let pass = 0; pass < 2; pass++) {
			for (let i = 0; i < train.x.length; i++) {
				if (!inside(train.y[i])) continue;
				const p = { x: px(train.x[i]), y: train.y[i] * yScale, radius: 4.5 };
				if (pass === 0) dots.set(d++, { ...p, radius: 6, color: paperFill(mD) });
				else dots.set(d++, { ...p, color: tone('data', mD) });
			}
		}
		if (showTest >= 0.5) {
			for (let i = 0; i < test.x.length; i++) {
				if (!inside(test.y[i])) continue;
				dots.set(d++, { x: px(test.x[i]), y: test.y[i] * yScale, radius: 3.2, ring: 1.4, color: tone('ok', mO) });
			}
		}
		dots.count = d;
		dots.upload();

		// error chart
		let q = 0;
		let cd = 0;
		const a = chartMix;
		if (a > 0.01) {
			const axis = tone('muted', 1, 0.6 * a);
			const grid = tone('muted', 1, 0.18 * a);
			chartLines.set(q++, { a: [c0, -yl], b: [c0, yl], width: 1, color: axis });
			chartLines.set(q++, { a: [c0, -yl], b: [c0 + CW, -yl], width: 1, color: axis });
			for (const e of [0.01, 0.1, 1, 10]) {
				chartLines.set(q++, { a: [c0, cy(e)], b: [c0 + CW, cy(e)], width: 1, color: grid });
				const [lx, ly] = v.project(c0, cy(e));
				labels.push({ id: `ey-${e}`, x: lx - 6, y: ly, text: String(e), tone: 'muted', mono: true, align: 'right', opacity: a });
			}
			const trainPts: Pt[] = [];
			const testPts: Pt[] = [];
			for (let dd = 1; dd <= MAX_DEGREE; dd++) {
				trainPts.push([cx(dd), cy(trainErrs[dd - 1])]);
				testPts.push([cx(dd), cy(testErrs[dd - 1])]);
			}
			q = chartLines.setPolyline(q, testPts, { width: 2, color: tone('ok', 1, 0.9 * mO * a) });
			q = chartLines.setPolyline(q, trainPts, { width: 2, color: tone('data', 1, 0.9 * mD * a) });
			chartLines.set(q++, { a: [cx(degree), -yl], b: [cx(degree), yl], width: 1.5, color: tone('param', 1, 0.6 * mP * a) });
			for (let dd = 1; dd <= MAX_DEGREE; dd++) {
				chartDots.set(cd++, { x: testPts[dd - 1][0], y: testPts[dd - 1][1], radius: 2.8, color: tone('ok', 1, 0.95 * mO * a) });
				chartDots.set(cd++, { x: trainPts[dd - 1][0], y: trainPts[dd - 1][1], radius: 2.8, color: tone('data', 1, 0.95 * mD * a) });
			}
			for (let dd = 2; dd <= MAX_DEGREE; dd += 2) {
				const [lx, ly] = v.project(cx(dd), -yl);
				labels.push({ id: `ex-${dd}`, x: lx, y: ly + 12, text: String(dd), tone: 'muted', mono: true, opacity: a });
			}
			const [l0x, l0y] = v.project(c0, -yl);
			const [l1x] = v.project(c0 + CW, -yl);
			labels.push({ id: 'cap-simple', x: l0x, y: l0y + 32, text: 'too simple', tone: 'muted', align: 'left', opacity: a });
			labels.push({ id: 'cap-flex', x: l1x, y: l0y + 32, text: 'too flexible', tone: 'muted', align: 'right', opacity: a });
			labels.push({ id: 'cap-degree', x: (l0x + l1x) / 2, y: l0y + 32, text: 'degree', tone: 'muted', mono: true, opacity: a });
			const [tx, ty] = v.project(c0, yl);
			labels.push({ id: 'cap-error', x: tx, y: ty - 12, text: 'error', tone: 'muted', mono: true, align: 'left', opacity: a });
		}
		chartLines.count = q;
		chartLines.upload();
		chartDots.count = cd;
		chartDots.upload();
		dirty = false;
	}

	const scene: Scene = {
		init(c) {
			ctx = c;
			camera = new Camera2D();
			base = new Lines(c, { capacity: 400, blend: 'alpha' });
			ghostLines = new Lines(c, { capacity: N_GHOSTS * CURVE_SAMPLES, blend: 'alpha' });
			curveLines = new Lines(c, { capacity: 2 * CURVE_SAMPLES + 8, blend: 'alpha' });
			chartLines = new Lines(c, { capacity: 100, blend: 'alpha' });
			dots = new Points(c, { capacity: 2 * 60 + N_TEST + 8, blend: 'alpha' });
			chartDots = new Points(c, { capacity: 2 * MAX_DEGREE + 4, blend: 'alpha' });
		},
		setParams(p) {
			const prev = `${degree}|${n}|${noise}|${reg}|${fit}|${truthLine}|${showTest}|${chart}|${ghosts}|${scenario}`;
			degree = clamp(Math.round(p.degree ?? degree), 1, MAX_DEGREE);
			n = clamp(Math.round(p.n ?? n), 8, 60);
			noise = p.noise ?? noise;
			reg = p.reg ?? reg;
			fit = p.fit ?? fit;
			truthLine = p.truthLine ?? truthLine;
			showTest = p.test ?? showTest;
			chart = p.chart ?? chart;
			ghosts = p.ghosts ?? ghosts;
			const sc = Math.round(p.scenario ?? scenario);
			if (sc !== scenario) {
				scenario = sc;
				seed = 1;
			}
			lambda = reg <= 0 ? 0 : Math.pow(10, -5 + 5 * reg);
			const now = `${degree}|${n}|${noise}|${reg}|${fit}|${truthLine}|${showTest}|${chart}|${ghosts}|${scenario}`;
			if (now !== prev || fitKey === '') {
				refit();
				dirty = true;
			}
		},
		update(fr) {
			const v = ctx.view;
			const target = chart >= 0.5 ? 1 : 0;
			if (chartMix !== target) {
				if (ctx.reducedMotion) chartMix = target;
				else {
					const step = fr.dt / 0.35;
					chartMix = target > chartMix ? Math.min(target, chartMix + step) : Math.max(target, chartMix - step);
				}
				if (fr.dt === 0 && !ctx.reducedMotion) chartMix = target;
				dirty = true;
			}
			const s = chartMix * chartMix * (3 - 2 * chartMix);
			pw = PW_ALONE + (PW_PAIR - PW_ALONE) * s;
			yScale = 1 + (YS_PAIR - 1) * s;
			c0 = pw + GAP;
			const world: WorldRect = { x0: 0, x1: pw + (GAP + CW + CHART_TAIL) * s, y0: -HALF_H * yScale, y1: HALF_H * yScale };
			fitCamera(camera, v, world, lessonInsets(v));
			v.setViewProj(camera.matrix(v.aspect));

			const sized = v.width !== lastW || v.height !== lastH;
			if (dirty || chartMix !== target || sized) {
				lastW = v.width;
				lastH = v.height;
				// labels depend on the camera, so rebuild while the chart is visible
				lastLabels = [];
				rebuild(lastLabels);
			}
			ctx.labels.set(lastLabels);

			const tr = fmtErr(trainErrs[degree - 1]);
			if (tr !== lastTrain) {
				lastTrain = tr;
				ctx.emit('train', tr);
			}
			const te = showTest >= 0.5 ? fmtErr(testErrs[degree - 1]) : '—';
			if (te !== lastTest) {
				lastTest = te;
				ctx.emit('test', te);
			}
		},
		draw(pass) {
			base.draw(pass);
			ghostLines.draw(pass);
			curveLines.draw(pass);
			dots.draw(pass);
			chartLines.draw(pass);
			chartDots.draw(pass);
		},
		command(id) {
			if (id === 'resample') {
				seed++;
				refit();
				dirty = true;
			}
		},
		highlight(t) {
			hl = t;
			dirty = true;
		},
		dispose() {
			base.destroy();
			ghostLines.destroy();
			curveLines.destroy();
			chartLines.destroy();
			dots.destroy();
			chartDots.destroy();
		}
	};
	return scene;
};
