import { Camera2D } from '../../../gpu/camera.ts';
import { tone } from '../../../gpu/colors.ts';
import { Lines } from '../../../gpu/primitives/lines.ts';
import { Points } from '../../../gpu/primitives/points.ts';
import type { Label, Scene, SceneContext, SceneFactory, Tone } from '../../../gpu/types.ts';
import { activate, derivative, type ActivationId } from '../../../ml/activations.ts';
import { fitCamera, lessonInsets, type WorldRect } from '../../fit.ts';
import { X_MAX, X_MIN, center, evalFit, fitBasis, target, unit, type BasisFit } from './basis.ts';

// Both views share this world rectangle. Plot area is x in [-5, 5], y in [-1.6, 1.6].
const WORLD: WorldRect = { x0: -5.7, x1: 5.7, y0: -2.0, y1: 2.0 };
const Y_LO = -1.6;
const Y_HI = 1.6;
const Z_MAX = 5;
const BUILD_X = Z_MAX / X_MAX;
const EASE_TAU = 0.1;
const GAPS = 20;

type Pt = [number, number];

/** Output value range shown for each activation id. */
const RANGES: Record<ActivationId, [number, number]> = {
	0: [-1.5, 5],
	1: [-0.25, 1.25],
	2: [-1.5, 5],
	3: [-1.25, 1.25],
	4: [-0.25, 1.25],
	5: [-1.5, 5]
};


// ---- label placement -------------------------------------------------------------------------
// Captions that sit near curves are placed by scoring candidate spots against everything drawn
// around them (axes, ticks, curves, tangent), so they stay clear at any slider value.

type Px = [number, number];
type Box = { x0: number; y0: number; x1: number; y1: number };
type Cand = { x: number; y: number; align: 'left' | 'center' | 'right'; cost: number };

const MONO_W = 7.3;
const TEXT_W = 6.6;
const TEXT_H = 16;

function boxOf(x: number, y: number, text: string, align: 'left' | 'center' | 'right', mono: boolean): Box {
	const w = text.length * (mono ? MONO_W : TEXT_W);
	const x0 = align === 'left' ? x : align === 'right' ? x - w : x - w / 2;
	return { x0, x1: x0 + w, y0: y - TEXT_H / 2, y1: y + TEXT_H / 2 };
}

/** Pixel-space points every few px along a polyline, so a line counts by its length. */
function densify(poly: Px[], step = 4): Px[] {
	const out: Px[] = [];
	for (let i = 0; i < poly.length; i++) {
		out.push(poly[i]);
		if (i === 0) continue;
		const [ax, ay] = poly[i - 1];
		const [bx, by] = poly[i];
		const n = Math.floor(Math.hypot(bx - ax, by - ay) / step);
		for (let k = 1; k < n; k++) out.push([ax + ((bx - ax) * k) / n, ay + ((by - ay) * k) / n]);
	}
	return out;
}

function boxesOverlap(a: Box, b: Box) {
	return a.x0 < b.x1 && b.x0 < a.x1 && a.y0 < b.y1 && b.y0 < a.y1;
}

/** Pick the candidate whose text box touches the fewest obstacle points and boxes. */
function pickSpot(
	text: string,
	mono: boolean,
	cands: Cand[],
	pts: Px[],
	boxes: Box[],
	bounds: Box,
	extra?: (b: Box, c: Cand) => number
): { cand: Cand; box: Box } {
	const PAD = 3;
	let best = cands[0];
	let bestBox = boxOf(best.x, best.y, text, best.align, mono);
	let bestScore = Infinity;
	for (const c of cands) {
		const b = boxOf(c.x, c.y, text, c.align, mono);
		let score = c.cost;
		if (b.x0 < bounds.x0 || b.x1 > bounds.x1 || b.y0 < bounds.y0 || b.y1 > bounds.y1) score += 200;
		for (const p of pts) if (p[0] > b.x0 - PAD && p[0] < b.x1 + PAD && p[1] > b.y0 - PAD && p[1] < b.y1 + PAD) score += 1;
		for (const o of boxes) if (boxesOverlap(b, { x0: o.x0 - PAD, x1: o.x1 + PAD, y0: o.y0 - PAD, y1: o.y1 + PAD })) score += 60;
		if (extra) score += extra(b, c);
		if (score < bestScore) {
			bestScore = score;
			best = c;
			bestBox = b;
		}
	}
	return { cand: best, box: bestBox };
}

/** Candidate spots on rings of the given radii around (x, y); text anchors away from the center. */
function ring(x: number, y: number, radii: number[], baseCost: number, steps = 12): Cand[] {
	const out: Cand[] = [];
	for (const r of radii) {
		for (let i = 0; i < steps; i++) {
			const a = (i / steps) * Math.PI * 2;
			const dx = Math.cos(a) * r;
			const dy = Math.sin(a) * r;
			out.push({ x: x + dx, y: y + dy, align: dx > 6 ? 'left' : dx < -6 ? 'right' : 'center', cost: baseCost + r * 0.05 });
		}
	}
	return out;
}

const clamp = (v: number, a: number, b: number) => Math.min(b, Math.max(a, v));

export const createActivationsScene: SceneFactory = () => {
	let ctx: SceneContext;
	let camera: Camera2D;
	let lines: Lines;
	let glows: Points;
	let dots: Points;

	let act: ActivationId = 0;
	let z = 1;
	let showDeriv = 0;
	let mode = 0;
	let units = 3;
	let hl: Tone | null = null;

	let lo = RANGES[0][0];
	let hi = RANGES[0][1];
	let rangeInit = false;

	let fit: BasisFit = fitBasis(0, 3);
	let fitKey = '0|3';

	let lastOut: number | string = '';
	let lastSlope: number | string = '';
	let lastErr: number | string = '';
	const emitIf = (key: string, value: number | string, last: number | string): number | string => {
		if (value !== last) ctx.emit(key, value);
		return value;
	};

	const mul = (t: Tone) => (hl === null || hl === t ? 1 : 0.3);
	const yOf = (v: number) => Y_LO + ((v - lo) / (hi - lo)) * (Y_HI - Y_LO);

	/** Split a sampled curve into runs that lie inside the plot's vertical range, clipping at the edges. */
	function clipRuns(pts: Pt[]): Pt[][] {
		const runs: Pt[][] = [];
		let cur: Pt[] = [];
		const inside = (p: Pt) => p[1] >= Y_LO && p[1] <= Y_HI;
		for (let i = 0; i < pts.length; i++) {
			const p = pts[i];
			if (i > 0) {
				const q = pts[i - 1];
				const pin = inside(p);
				const qin = inside(q);
				if (pin !== qin) {
					const edge = (pin ? q[1] < Y_LO : p[1] < Y_LO) ? Y_LO : Y_HI;
					const t = (edge - q[1]) / (p[1] - q[1]);
					const c: Pt = [q[0] + t * (p[0] - q[0]), edge];
					if (qin) {
						cur.push(c);
						runs.push(cur);
						cur = [];
					} else cur = [c];
				}
			}
			if (inside(p)) cur.push(p);
		}
		if (cur.length > 1) runs.push(cur);
		return runs.filter((r) => r.length > 1);
	}

	/** Sample f over [-Z_MAX, Z_MAX]; discontinuous curves are split at 0 so the jump is not drawn slanted. */
	function curvePieces(f: (z: number) => number, split: boolean): Pt[][] {
		const N = 200;
		const mk = (a: number, b: number, n: number, eps: number): Pt[] => {
			const out: Pt[] = [];
			for (let i = 0; i <= n; i++) {
				const zz = a + ((b - a) * i) / n;
				out.push([zz, yOf(f(i === n && eps ? zz - eps : zz))]);
			}
			return out;
		};
		const raw = split ? [mk(-Z_MAX, 0, N / 2, 1e-9), mk(0, Z_MAX, N / 2, 0)] : [mk(-Z_MAX, Z_MAX, N, 0)];
		return raw.flatMap(clipRuns);
	}

	function drawPieces(n: number, pieces: Pt[][], t: Tone, width: number, intensity: number, alpha: number, halo: boolean) {
		for (const p of pieces) {
			if (halo) n = lines.setPolyline(n, p, { width: 8, color: tone(t, 0.8, 0.1 * alpha) });
			n = lines.setPolyline(n, p, { width, color: tone(t, intensity, alpha) });
		}
		return n;
	}

	function buildOne(o: number, labels: Label[]) {
		const v = ctx.view;
		let n = lines.count;
		let d = dots.count;
		let gl = glows.count;
		const mm = mul('muted') * o;
		const dm = mul('data') * o;
		const gm = mul('grad') * o;
		const sh = clamp(showDeriv, 0, 1);

		// axes
		const y0 = yOf(0);
		lines.set(n++, { a: [-Z_MAX, y0], b: [Z_MAX, y0], width: 1, color: tone('muted', 1, 0.5 * mm) });
		lines.set(n++, { a: [0, Y_LO], b: [0, Y_HI], width: 1, color: tone('muted', 1, 0.5 * mm) });

		// curves: derivative first so the function stays on top where the two coincide
		// (the step function and its zero slope share the left half of the axis).
		if (sh > 0.002) {
			const dsplit = act === 2 || act === 5;
			n = drawPieces(n, curvePieces((zz) => derivative(act, zz), dsplit), 'grad', 2, 1.6, 0.95 * gm * sh, false);
		}
		n = drawPieces(n, curvePieces((zz) => activate(act, zz), act === 4), 'data', 2.5, 1.6, 0.95 * dm, true);
		if (act === 4) {
			dots.set(d++, { x: 0, y: yOf(0), radius: 4, ring: 1.3, color: tone('data', 1.6, 0.9 * dm) });
			dots.set(d++, { x: 0, y: yOf(1), radius: 4, color: tone('data', 2, 0.9 * dm) });
		}

		// marker
		const zc = clamp(z, -Z_MAX, Z_MAX);
		const out = activate(act, zc);
		const slope = derivative(act, zc);
		const oy = clamp(yOf(out), Y_LO, Y_HI);
		const sy = clamp(yOf(slope), Y_LO, Y_HI);
		lines.set(n++, { a: [zc, y0], b: [zc, oy], width: 1, color: tone('data', 1, 0.45 * dm) });
		glows.set(gl++, { x: zc, y: oy, radius: 30, color: tone('data', 2, 0.45 * dm), softness: 1 });
		dots.set(d++, { x: zc, y: oy, radius: 7, color: tone('data', 3, dm) });
		if (sh > 0.002) {
			const k = (slope * (Y_HI - Y_LO)) / (hi - lo);
			const l = Math.hypot(1, k);
			const hx = (1 / l) * 0.9;
			const hy = (k / l) * 0.9;
			lines.set(n++, {
				a: [zc - hx, oy - hy],
				b: [zc + hx, oy + hy],
				width: 2,
				color: tone('grad', 2.5, gm * sh)
			});
			dots.set(d++, { x: zc, y: sy, radius: 5.5, color: tone('grad', 3, gm * sh) });
		}
		lines.count = n;
		dots.count = d;
		glows.count = gl;

		// labels
		const pm = mul('muted') * o;
		const [ox0, oy0] = v.project(-Z_MAX, y0);
		void ox0;
		const fixed: Box[] = [];
		for (const t of [-4, -2, 2, 4]) {
			const [px, py] = v.project(t, y0);
			labels.push({ id: `tz${t}`, x: px, y: py + 14, text: String(t), tone: 'muted', mono: true, opacity: 0.8 * pm });
			fixed.push(boxOf(px, py + 14, String(t), 'center', true));
		}
		const yTicks = [1];
		if (act === 0 || act === 3 || act === 5) yTicks.push(-1);
		for (const t of yTicks) {
			const [px, py] = v.project(0, yOf(t));
			labels.push({ id: `ty${t}`, x: px - 8, y: py, text: String(t), tone: 'muted', mono: true, align: 'right', opacity: 0.8 * pm });
			fixed.push(boxOf(px - 8, py, String(t), 'right', true));
		}
		const [axR, axY] = v.project(Z_MAX, y0);
		labels.push({ id: 'axz', x: axR, y: axY + 14, text: 'z', tone: 'muted', opacity: pm });
		fixed.push(boxOf(axR, axY + 14, 'z', 'center', false));
		const [axX, axT] = v.project(0, Y_HI);
		labels.push({ id: 'axo', x: axX + 8, y: axT - 4, text: 'output', tone: 'muted', align: 'left', opacity: pm });
		fixed.push(boxOf(axX + 8, axT - 4, 'output', 'left', false));
		void oy0;

		const [dpx, dpy] = v.project(zc, oy);

		// Everything the two value captions must stay clear of, in pixels.
		const toPx = (poly: Pt[]): Px[] => poly.map(([a, b]) => v.project(a, b));
		const obst: Px[] = [];
		obst.push(...densify([v.project(-Z_MAX, y0), v.project(Z_MAX, y0)]));
		obst.push(...densify([v.project(0, Y_LO), v.project(0, Y_HI)]));
		obst.push(...densify([v.project(zc, y0), [dpx, dpy]]));
		for (const piece of curvePieces((zz) => activate(act, zz), act === 4)) obst.push(...densify(toPx(piece), 3));
		if (sh > 0.002) {
			const dsplit = act === 2 || act === 5;
			for (const piece of curvePieces((zz) => derivative(act, zz), dsplit)) obst.push(...densify(toPx(piece), 3));
		}
		// the ball and its glow core
		for (let i = 0; i < 8; i++) {
			const a = (i / 8) * Math.PI * 2;
			obst.push([dpx + Math.cos(a) * 9, dpy + Math.sin(a) * 9]);
		}
		obst.push([dpx, dpy]);
		const bounds: Box = { x0: 12, x1: ctx.view.width - 12, y0: 134, y1: ctx.view.height - 80 };

		// output value: right beside the ball, on whichever side is emptiest
		const outText = out.toFixed(2);
		const outPick = pickSpot(outText, true, ring(dpx, dpy, [20, 28, 38], 0), obst, fixed, bounds);
		labels.push({
			id: 'outv',
			x: outPick.cand.x,
			y: outPick.cand.y,
			text: outText,
			tone: 'data',
			mono: true,
			align: outPick.cand.align,
			opacity: dm
		});

		// slope caption: at an end of the tangent segment (or by the slope dot), clear of the output value
		if (sh > 0.002) {
			const k = (slope * (Y_HI - Y_LO)) / (hi - lo);
			const l = Math.hypot(1, k);
			const hx = (1 / l) * 0.9;
			const hy = (k / l) * 0.9;
			const [e1x, e1y] = v.project(zc + hx, oy + hy);
			const [e0x, e0y] = v.project(zc - hx, oy - hy);
			const sp = v.project(zc, sy);
			const slopeObst = [...obst];
			slopeObst.push(...densify([[e0x, e0y], [e1x, e1y]]));
			for (let i = 0; i < 8; i++) {
				const a = (i / 8) * Math.PI * 2;
				slopeObst.push([sp[0] + Math.cos(a) * 7, sp[1] + Math.sin(a) * 7]);
			}
			const slopeText = `slope ${slope.toFixed(2)}`;
			const cands = [
				...ring(e1x, e1y, [14, 24, 34], 0),
				...ring(e0x, e0y, [14, 24, 34], 0.5),
				...ring(sp[0], sp[1], [14, 24], 2)
			];
			const pick = pickSpot(slopeText, true, cands, slopeObst, [...fixed, outPick.box], bounds);
			labels.push({
				id: 'slopev',
				x: pick.cand.x,
				y: pick.cand.y,
				text: slopeText,
				tone: 'grad',
				mono: true,
				align: pick.cand.align,
				opacity: gm * sh
			});
		}
	}

	function buildBuild(o: number, labels: Label[]) {
		const v = ctx.view;
		let n = lines.count;
		let d = dots.count;
		const mm = mul('muted') * o;
		const dm = mul('data') * o;
		const pm = mul('param') * o;
		const wx = (x: number) => x * BUILD_X;
		const cy = (y: number) => clamp(y, Y_LO - 0.2, Y_HI + 0.2);
		const S = 120;
		const xs: number[] = [];
		for (let i = 0; i <= S; i++) xs.push(X_MIN + ((X_MAX - X_MIN) * i) / S);

		lines.set(n++, { a: [wx(X_MIN), 0], b: [wx(X_MAX), 0], width: 1, color: tone('muted', 1, 0.5 * mm) });

		// gaps between fit and target
		for (let i = 0; i < GAPS; i++) {
			const x = X_MIN + ((X_MAX - X_MIN) * (i + 0.5)) / GAPS;
			lines.set(n++, {
				a: [wx(x), cy(target(x))],
				b: [wx(x), cy(evalFit(act, units, fit, x))],
				width: 1,
				color: tone('neg', 1, 0.5 * o * mul('neg'))
			});
		}
		// target
		n = lines.setPolyline(
			n,
			xs.map((x): Pt => [wx(x), target(x)]),
			{ width: 1.5, color: tone('muted', 1, 0.65 * mm) }
		);
		// unit contributions
		for (let k = 0; k < units; k++) {
			n = lines.setPolyline(
				n,
				xs.map((x): Pt => [wx(x), cy(fit.v[k] * unit(act, units, k, x))]),
				{ width: 1.25, color: tone('param', 1, 0.65 * pm) }
			);
			dots.set(d++, { x: wx(center(units, k)), y: 0, radius: 3.5, color: tone('param', 1.8, 0.9 * pm) });
		}
		// fit
		const fitPts = xs.map((x): Pt => [wx(x), cy(evalFit(act, units, fit, x))]);
		n = lines.setPolyline(n, fitPts, { width: 8, color: tone('data', 0.8, 0.1 * dm) });
		n = lines.setPolyline(n, fitPts, { width: 2.5, color: tone('data', 1.6, 0.95 * dm) });
		lines.count = n;
		dots.count = d;

		// labels: "target" and "network output" go in free space beside the curve they name
		const px = (x: number, y: number): Px => v.project(wx(x), y);
		const tgtPx = densify(xs.map((x) => px(x, target(x))), 3);
		const fitPx = densify(xs.map((x) => px(x, cy(evalFit(act, units, fit, x)))), 3);
		const obst: Px[] = [...tgtPx, ...fitPx];
		obst.push(...densify([v.project(wx(X_MIN), 0), v.project(wx(X_MAX), 0)]));
		for (let k = 0; k < units; k++) {
			obst.push(...densify(xs.map((x) => v.project(wx(x), cy(fit.v[k] * unit(act, units, k, x)))), 3));
		}
		for (let i = 0; i < GAPS; i++) {
			const x = X_MIN + ((X_MAX - X_MIN) * (i + 0.5)) / GAPS;
			obst.push(...densify([px(x, cy(target(x))), px(x, cy(evalFit(act, units, fit, x)))]));
		}
		const [hx, hy] = v.project(0, Y_HI + 0.15);
		const unitsText = `${units} hidden ${units === 1 ? 'unit' : 'units'}`;
		const fixed: Box[] = [boxOf(hx, hy, unitsText, 'center', true)];
		const bounds: Box = { x0: 12, x1: ctx.view.width - 12, y0: 134, y1: ctx.view.height - 80 };
		const near = (p: Px, pts: Px[]) => {
			let m = Infinity;
			for (const q of pts) m = Math.min(m, Math.hypot(p[0] - q[0], p[1] - q[1]));
			return m;
		};
		const spots = (pts: Px[]): Cand[] => {
			const out: Cand[] = [];
			for (let i = 0; i < pts.length; i += 6) {
				for (const d of [14, 22, 32]) {
					out.push({ x: pts[i][0], y: pts[i][1] - d, align: 'center', cost: d * 0.05 });
					out.push({ x: pts[i][0], y: pts[i][1] + d, align: 'center', cost: d * 0.05 });
				}
			}
			return out;
		};
		const own = (mine: Px[], other: Px[]) => (b: Box) => {
			const c: Px = [(b.x0 + b.x1) / 2, (b.y0 + b.y1) / 2];
			return near(c, mine) > near(c, other) ? 40 : 0;
		};
		const tPick = pickSpot('target', false, spots(tgtPx), obst, fixed, bounds, own(tgtPx, fitPx));
		const nPick = pickSpot('network output', false, spots(fitPx), obst, [...fixed, tPick.box], bounds, own(fitPx, tgtPx));
		labels.push(
			{ id: 'tgt', x: tPick.cand.x, y: tPick.cand.y, text: 'target', tone: 'muted', align: 'center', opacity: mm },
			{ id: 'net', x: nPick.cand.x, y: nPick.cand.y, text: 'network output', tone: 'data', align: 'center', opacity: dm }
		);
		labels.push({
			id: 'nunits',
			x: hx,
			y: hy,
			text: unitsText,
			tone: 'param',
			mono: true,
			opacity: pm
		});
	}


	const scene: Scene = {
		init(c) {
			ctx = c;
			camera = new Camera2D();
			lines = new Lines(c, { capacity: 3200, blend: 'alpha' });
			glows = new Points(c, { capacity: 8, blend: 'additive' });
			dots = new Points(c, { capacity: 64, blend: 'alpha' });
		},
		setParams(p) {
			act = Math.round(p.act ?? act) as ActivationId;
			if (!(act in RANGES)) act = 0;
			z = p.z ?? z;
			showDeriv = p.showDeriv ?? showDeriv;
			mode = p.mode ?? mode;
			units = clamp(Math.round(p.units ?? units), 1, 6);
			const key = `${act}|${units}`;
			if (key !== fitKey) {
				fitKey = key;
				fit = fitBasis(act, units);
			}
		},
		update(frame) {
			const v = ctx.view;
			fitCamera(camera, v, WORLD, lessonInsets(v));
			v.setViewProj(camera.matrix(v.aspect));

			const [tl, th] = RANGES[act];
			if (!rangeInit || ctx.reducedMotion) {
				lo = tl;
				hi = th;
				rangeInit = true;
			} else {
				const k = 1 - Math.exp(-frame.dt / EASE_TAU);
				lo += (tl - lo) * k;
				hi += (th - hi) * k;
			}

			const o2 = clamp(mode, 0, 1);
			const o1 = 1 - o2;
			lines.count = 0;
			dots.count = 0;
			glows.count = 0;
			const labels: Label[] = [];
			if (o1 > 0.002) buildOne(o1, labels);
			if (o2 > 0.002) buildBuild(o2, labels);
			lines.upload();
			dots.upload();
			glows.upload();
			ctx.labels.set(labels);

			if (mode >= 0.5) {
				lastOut = emitIf('out', '–', lastOut);
				lastSlope = emitIf('slope', '–', lastSlope);
				lastErr = emitIf('err', fit.error, lastErr);
			} else {
				const zc = clamp(z, -Z_MAX, Z_MAX);
				lastOut = emitIf('out', activate(act, zc), lastOut);
				lastSlope = emitIf('slope', derivative(act, zc), lastSlope);
				lastErr = emitIf('err', '–', lastErr);
			}
		},
		draw(pass) {
			lines.draw(pass);
			glows.draw(pass);
			dots.draw(pass);
		},
		highlight(t) {
			hl = t;
		},
		dispose() {
			lines.destroy();
			glows.destroy();
			dots.destroy();
		}
	};
	return scene;
};
