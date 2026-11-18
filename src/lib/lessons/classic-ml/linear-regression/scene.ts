import { Camera2D } from '../../../gpu/camera.ts';
import { tone } from '../../../gpu/colors.ts';
import { Lines } from '../../../gpu/primitives/lines.ts';
import { Points } from '../../../gpu/primitives/points.ts';
import { Quads } from '../../../gpu/primitives/quads.ts';
import type { Label, Scene, SceneContext, SceneFactory, ScenePointer, Tone } from '../../../gpu/types.ts';
import { gradient, leastSquares, makeLinearData, mse } from '../../../ml/linreg.ts';
import { fitCamera, lessonInsets, type WorldRect } from '../../fit.ts';

const N_POINTS = 14;
const EXT = 1.3; // the plot covers [-EXT, EXT] on both axes
const WORLD: WorldRect = { x0: -1.45, x1: 2.15, y0: -1.5, y1: 1.4 };
const BAR = { x0: 1.5, x1: 1.7, y0: -EXT, h: 2 * EXT }; // full height = MSE of 1
const LR = 0.3;
const STEPS_PER_SECOND = 40;
const MAX_STEPS_PER_FRAME = 4;
const GRAB_PX = 14;

const clamp = (v: number, a: number, b: number) => Math.min(b, Math.max(a, v));
const round2 = (v: number) => Math.round(v * 100) / 100;
const signed = (v: number, d: number) => (v < 0 ? '−' : '') + Math.abs(v).toFixed(d);

export const createLinearRegressionScene: SceneFactory = () => {
	let ctx: SceneContext;
	let camera: Camera2D;
	let quads: Quads;
	let lines: Lines;
	let dots: Points;

	// params
	let w = 0.2;
	let b = -0.3;
	let show = 3;
	let bestOn = 0;
	let noise = 0.18;
	let scenario = 1;
	let paramsSeen = false;
	let noiseUsed = -1;
	let hl: Tone | null = null;

	// state
	let seed = 1;
	let xs: Float64Array = new Float64Array(0);
	let ys: Float64Array = new Float64Array(0);
	let best = { w: 0, b: 0 };
	let fitting = false;
	let acc = 0;
	let lastEmitW = NaN;
	let lastEmitB = NaN;
	let grabbed = -1;
	let hover = -1;

	// emitted readouts
	let lastMse = -1;
	let lastBest = NaN;

	const mul = (t: Tone) => (hl === null || hl === t ? 1 : 0.3);

	function regenerate() {
		const d = makeLinearData(N_POINTS, 0.8, 0.1, noise, seed + scenario * 101);
		xs = d.x;
		ys = d.y;
		if (scenario === 2) {
			let k = 0;
			for (let i = 1; i < xs.length; i++) if (xs[i] > xs[k]) k = i;
			ys[k] = -1.1;
		}
		noiseUsed = noise;
		grabbed = -1;
		hover = -1;
		best = leastSquares(xs, ys);
	}

	function emitParams() {
		const rw = round2(w);
		const rb = round2(b);
		if (rw !== lastEmitW) {
			lastEmitW = rw;
			ctx.emit('param:w', rw);
		}
		if (rb !== lastEmitB) {
			lastEmitB = rb;
			ctx.emit('param:b', rb);
		}
	}

	function stepFit() {
		const g = gradient(xs, ys, w, b);
		if (Math.abs(g.dw) < 1e-3 && Math.abs(g.db) < 1e-3) {
			fitting = false;
			return;
		}
		w -= LR * g.dw;
		b -= LR * g.db;
	}

	/** Pointer in CSS px -> index of the nearest point within the grab radius, or -1. */
	function pick(e: ScenePointer): number {
		let bestI = -1;
		let bestD = GRAB_PX;
		for (let i = 0; i < xs.length; i++) {
			const [px, py] = ctx.view.project(xs[i], ys[i]);
			const d = Math.hypot(px - e.x, py - e.y);
			if (d <= bestD) {
				bestD = d;
				bestI = i;
			}
		}
		return bestI;
	}

	function drawStatic() {
		const axis = tone('muted', 1, 0.5);
		const frame = tone('muted', 1, 0.6);
		let n = 0;
		const seg = (ax: number, ay: number, bx: number, by: number, c: [number, number, number, number]) =>
			lines.set(n++, { a: [ax, ay], b: [bx, by], width: 1, color: c });
		seg(-EXT, 0, EXT, 0, axis);
		seg(0, -EXT, 0, EXT, axis);
		for (const t of [-1, 1]) {
			seg(t, -0.04, t, 0.04, frame);
			seg(-0.04, t, 0.04, t, frame);
		}
		const rect = (x0: number, y0: number, x1: number, y1: number) => {
			seg(x0, y0, x1, y0, frame);
			seg(x1, y0, x1, y1, frame);
			seg(x1, y1, x0, y1, frame);
			seg(x0, y1, x0, y0, frame);
		};
		rect(-EXT, -EXT, EXT, EXT);
		if (show >= 3) rect(BAR.x0, BAR.y0, BAR.x1, BAR.y0 + BAR.h);
		return n;
	}

	/** Clip y = w*x + b to the plot rectangle. Returns [x0, x1] or null. */
	function clipLine(lw: number, lb: number): [number, number] | null {
		let lo = -EXT;
		let hi = EXT;
		if (Math.abs(lw) < 1e-9) return Math.abs(lb) <= EXT ? [lo, hi] : null;
		const a = (-EXT - lb) / lw;
		const c = (EXT - lb) / lw;
		lo = Math.max(lo, Math.min(a, c));
		hi = Math.min(hi, Math.max(a, c));
		return lo < hi ? [lo, hi] : null;
	}

	const scene: Scene = {
		init(c) {
			ctx = c;
			camera = new Camera2D();
			quads = new Quads(c, { capacity: 64, blend: 'alpha' });
			lines = new Lines(c, { capacity: 256, blend: 'alpha' });
			dots = new Points(c, { capacity: 64 * 3, blend: 'alpha' });
		},
		setParams(p) {
			const prev = { scenario };
			scenario = Math.round(p.scenario ?? scenario);
			show = Math.round(p.show ?? show);
			bestOn = Math.round(p.best ?? bestOn);
			noise = p.noise ?? noise;
			if (fitting) {
				const dw = Math.abs((p.w ?? w) - lastEmitW);
				const db = Math.abs((p.b ?? b) - lastEmitB);
				if (dw > 0.006 || db > 0.006) {
					fitting = false;
					w = p.w ?? w;
					b = p.b ?? b;
				}
			} else {
				w = p.w ?? w;
				b = p.b ?? b;
			}
			if (!paramsSeen || prev.scenario !== scenario || Math.abs(noise - noiseUsed) > 0.001) {
				if (paramsSeen) fitting = false;
				regenerate();
			}
			paramsSeen = true;
		},
		update(fr) {
			const v = ctx.view;
			fitCamera(camera, v, WORLD, lessonInsets(v));
			v.setViewProj(camera.matrix(v.aspect));

			if (fitting && fr.dt > 0) {
				acc += fr.dt * STEPS_PER_SECOND;
				let n = Math.floor(acc);
				acc -= n;
				if (n > MAX_STEPS_PER_FRAME) {
					n = MAX_STEPS_PER_FRAME;
					acc = 0;
				}
				for (let i = 0; i < n && fitting; i++) stepFit();
				emitParams();
			}

			const n = xs.length;
			const mGrad = mul('grad');
			const mParam = mul('param');
			const mOk = mul('ok');
			const loss = mse(xs, ys, w, b);
			const bestLoss = mse(xs, ys, best.w, best.b);

			// squares
			let qn = 0;
			if (show >= 3) {
				for (let i = 0; i < n; i++) {
					const yh = w * xs[i] + b;
					const s = Math.abs(yh - ys[i]);
					if (s < 1e-4) continue;
					const dir = xs[i] < 0 ? 1 : -1;
					quads.set(qn++, {
						x: xs[i] + (dir * s) / 2,
						y: (yh + ys[i]) / 2,
						w: s,
						h: s,
						color: tone('grad', 0.5, 0.2 * mGrad)
					});
				}
			}

			// lines
			let ln = drawStatic();
			if (show >= 2) {
				for (let i = 0; i < n; i++) {
					const yh = w * xs[i] + b;
					lines.set(ln++, { a: [xs[i], ys[i]], b: [xs[i], yh], width: 1.5, color: tone('grad', 1, 0.65 * mGrad) });
					if (show >= 3) {
						const s = Math.abs(yh - ys[i]);
						if (s < 1e-4) continue;
						const dir = xs[i] < 0 ? 1 : -1;
						const x2 = xs[i] + dir * s;
						const c = tone('grad', 1, 0.5 * mGrad);
						lines.set(ln++, { a: [xs[i], yh], b: [x2, yh], width: 1, color: c });
						lines.set(ln++, { a: [x2, yh], b: [x2, ys[i]], width: 1, color: c });
						lines.set(ln++, { a: [x2, ys[i]], b: [xs[i], ys[i]], width: 1, color: c });
					}
				}
			}
			if (bestOn) {
				const r = clipLine(best.w, best.b);
				if (r) {
					lines.set(ln++, {
						a: [r[0], best.w * r[0] + best.b],
						b: [r[1], best.w * r[1] + best.b],
						width: 1.5,
						color: tone('ok', 1, 0.65 * mOk)
					});
				}
			}
			const labels: Label[] = [];
			const clip = show >= 1 ? clipLine(w, b) : null;
			if (clip) {
				const a: [number, number] = [clip[0], w * clip[0] + b];
				const e: [number, number] = [clip[1], w * clip[1] + b];
				lines.set(ln++, { a, b: e, width: 8, color: tone('param', 1, 0.12 * mParam) });
				lines.set(ln++, { a, b: e, width: 2.5, color: tone('param', 1.4, 0.95 * mParam) });
				// Fixed in the plot's top-left corner, which a rising trend leaves empty; a label that
				// follows the line's end lands on the squares or on the line itself.
				const [ex, ey] = v.project(-EXT, EXT);
				labels.push({
					id: 'line-eq',
					x: ex + 12,
					y: ey + 16,
					text: `ŷ = ${signed(w, 2)}x ${b < 0 ? '−' : '+'} ${Math.abs(b).toFixed(2)}`,
					tone: 'param',
					mono: true,
					align: 'left',
					opacity: mParam
				});
			}

			// loss bar
			if (show >= 3) {
				const hBar = BAR.h * clamp(loss, 0, 1);
				const hTick = BAR.h * clamp(bestLoss, 0, 1);
				const ty = BAR.y0 + hTick;
				lines.set(ln++, { a: [BAR.x0 - 0.05, ty], b: [BAR.x1 + 0.05, ty], width: 2, color: tone('ok', 1.2, 0.9 * mOk) });
				const [tx, tyPx] = v.project(BAR.x1 + 0.1, ty);
				labels.push({ id: 'bar-best', x: tx, y: tyPx, text: 'best', tone: 'ok', mono: true, align: 'left', opacity: mOk });
				const [cx, cy] = v.project((BAR.x0 + BAR.x1) / 2, BAR.y0);
				labels.push({ id: 'bar-error', x: cx, y: cy + 18, text: 'error', tone: 'grad', mono: true, opacity: mGrad });
				if (hBar > 1e-4) {
					quads.set(qn++, {
						x: (BAR.x0 + BAR.x1) / 2,
						y: BAR.y0 + hBar / 2,
						w: BAR.x1 - BAR.x0,
						h: hBar,
						color: tone('grad', 0.7, 0.75 * mGrad)
					});
				}
			}
			quads.count = qn;
			quads.upload();
			lines.count = ln;
			lines.upload();

			// points
			const mData = mul('data');
			let pn = 0;
			for (let pass = 0; pass < 2; pass++) {
				for (let i = 0; i < n; i++) {
					const active = i === hover || i === grabbed;
					if (pass === 0) dots.set(pn++, { x: xs[i], y: ys[i], radius: 5, color: tone('data', 1.3, 0.95 * mData) });
					else
						dots.set(pn++, {
							x: xs[i],
							y: ys[i],
							radius: active ? 9 : 7,
							ring: active ? 1.6 : 1,
							color: tone('data', active ? 0.8 : 0.3, 0.9 * mData)
						});
				}
			}
			dots.count = pn;
			dots.upload();

			const [ox, oy] = v.project(EXT, 0);
			labels.push({ id: 'axis-x', x: ox - 8, y: oy + 12, text: 'x', tone: 'muted', mono: true });
			const [yx, yy] = v.project(0, EXT);
			labels.push({ id: 'axis-y', x: yx + 12, y: yy + 10, text: 'y', tone: 'muted', mono: true });
			ctx.labels.set(labels);

			// No line on screen yet means there is no error to report.
			const shownLoss = show >= 1 ? loss : NaN;
			if (!Object.is(shownLoss, lastMse)) {
				lastMse = shownLoss;
				ctx.emit('mse', Number.isNaN(shownLoss) ? '—' : shownLoss);
			}
			if (bestLoss !== lastBest) {
				lastBest = bestLoss;
				ctx.emit('best', bestLoss);
			}
		},
		draw(pass) {
			quads.draw(pass);
			lines.draw(pass);
			dots.draw(pass);
		},
		command(id) {
			if (id === 'fit') {
				if (fitting) {
					fitting = false;
				} else if (ctx.reducedMotion) {
					w = best.w;
					b = best.b;
					emitParams();
				} else {
					fitting = true;
					acc = 0;
					lastEmitW = round2(w);
					lastEmitB = round2(b);
				}
			} else if (id === 'shuffle') {
				fitting = false;
				seed++;
				regenerate();
			}
		},
		highlight(t) {
			hl = t;
		},
		pointer(e) {
			if (e.type === 'down') {
				grabbed = pick(e);
				hover = grabbed;
			} else if (e.type === 'move') {
				if (grabbed >= 0) {
					const [wx, wy] = ctx.view.unproject(e.x, e.y);
					xs[grabbed] = clamp(wx, -EXT, EXT);
					ys[grabbed] = clamp(wy, -EXT, EXT);
					best = leastSquares(xs, ys);
				} else {
					hover = pick(e);
				}
			} else {
				if (grabbed >= 0) hover = e.type === 'leave' ? -1 : pick(e);
				grabbed = -1;
				if (e.type === 'leave') hover = -1;
			}
		},
		dispose() {
			quads.destroy();
			lines.destroy();
			dots.destroy();
		}
	};
	return scene;
};
