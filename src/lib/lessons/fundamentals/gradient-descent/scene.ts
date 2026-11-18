import { Camera2D } from '../../../gpu/camera.ts';
import { tone } from '../../../gpu/colors.ts';
import { Lines } from '../../../gpu/primitives/lines.ts';
import { Points } from '../../../gpu/primitives/points.ts';
import type { Label, Scene, SceneContext, SceneFactory, ScenePointer, Tone } from '../../../gpu/types.ts';
import { contourSegments } from '../../../ml/contours.ts';
import {
	DOMAIN,
	SURFACES,
	curve,
	initState,
	step,
	type OptState,
	type OptimizerId,
	type Vec2
} from '../../../ml/optimizers.ts';
import { fitCamera, lessonInsets, type WorldRect } from '../../fit.ts';

const WORLD: WorldRect = { x0: -2.6, x1: 2.6, y0: -1.7, y1: 1.7 };

// 1D mapping: weight w -> world x, loss L -> world y.
const W_MAX = 2.2;
const L_MAX = 4.84;
const Y_BASE = -1.3;
const Y_TOP = 1.4;
const Y_PER_LOSS = (Y_TOP - Y_BASE) / L_MAX;
const DIVERGE_AT = 50;
const TRAIL_1D = 40;
const TRAIL_2D = 600;

const GRID_COLS = 120;
const GRID_ROWS = 78;
const LEVELS = 12;

type Mode = '1d' | '2d';

interface Racer {
	kind: OptimizerId;
	tone: Tone;
	st: OptState;
	trail: Vec2[];
	frozen: boolean;
}

const RACERS: ReadonlyArray<{ kind: OptimizerId; tone: Tone }> = [
	{ kind: 'sgd', tone: 'grad' },
	{ kind: 'momentum', tone: 'data' },
	{ kind: 'adam', tone: 'ok' }
];

const clamp = (v: number, a: number, b: number) => Math.min(b, Math.max(a, v));
const wx = (w: number) => w;
const ly = (L: number) => Y_BASE + clamp(L, 0, L_MAX) * Y_PER_LOSS;
const fmt = (v: number) => (Math.abs(v) >= 1000 ? v.toExponential(1) : v.toFixed(2));

export const createGradientDescentScene: SceneFactory = () => {
	let ctx: SceneContext;
	let camera: Camera2D;
	let contourLines: Lines;
	let lines: Lines;
	let glows: Points;
	let dots: Points;

	// params
	let lr = 0.2;
	let beta = 0.9;
	let map = 0;
	let surfaceId = 0;
	let race = 0;
	let scenario = 1;
	let paramsSeen = false;
	let mode: Mode = '1d';
	let hl: Tone | null = null;

	// simulation
	let running = false;
	let acc = 0;
	let steps = 0;
	let w = curve.start;
	let trail1: number[] = [];
	let diverged = false;
	let racers: Racer[] = [];
	let customStart: Vec2 | null = null;

	// contours
	let contourSegs = new Float32Array(0);
	let contourLevels = new Uint8Array(0);
	let contourSurface = -1;
	let contourKey = '';

	// emitted readouts
	let lastLoss = NaN;
	let lastSteps = -1;

	const surface = () => SURFACES[surfaceId as 0 | 1 | 2] ?? SURFACES[0];
	const startPoint = (): Vec2 => customStart ?? surface().start;
	const mul = (t: Tone) => (hl === null || hl === t ? 1 : 0.3);

	function reset() {
		running = false;
		acc = 0;
		steps = 0;
		diverged = false;
		w = curve.start;
		trail1 = [];
		const s = startPoint();
		racers = RACERS.map((r) => ({
			kind: r.kind,
			tone: r.tone,
			st: initState(s),
			trail: [[s[0], s[1]]],
			frozen: false
		}));
	}

	function buildContours() {
		const s = surface();
		contourSurface = surfaceId;
		contourKey = '';
		const vals = new Float32Array(GRID_COLS * GRID_ROWS);
		let fmax = -Infinity;
		const [mx, my] = s.minima[0];
		const fmin = s.f(mx, my);
		for (let r = 0; r < GRID_ROWS; r++) {
			const y = DOMAIN.y[0] + (r / (GRID_ROWS - 1)) * (DOMAIN.y[1] - DOMAIN.y[0]);
			for (let c = 0; c < GRID_COLS; c++) {
				const x = DOMAIN.x[0] + (c / (GRID_COLS - 1)) * (DOMAIN.x[1] - DOMAIN.x[0]);
				const f = s.f(x, y);
				vals[r * GRID_COLS + c] = f;
				if (f > fmax) fmax = f;
			}
		}
		const smax = Math.sqrt(Math.max(1e-9, fmax - fmin));
		const out: number[] = [];
		const lv: number[] = [];
		const sx = (DOMAIN.x[1] - DOMAIN.x[0]) / (GRID_COLS - 1);
		const sy = (DOMAIN.y[1] - DOMAIN.y[0]) / (GRID_ROWS - 1);
		for (let i = 1; i <= LEVELS; i++) {
			const sv = (smax * i) / (LEVELS + 1);
			const level = fmin + sv * sv;
			for (const [x0, y0, x1, y1] of contourSegments(vals, GRID_COLS, GRID_ROWS, level)) {
				out.push(
					DOMAIN.x[0] + x0 * sx,
					DOMAIN.y[0] + y0 * sy,
					DOMAIN.x[0] + x1 * sx,
					DOMAIN.y[0] + y1 * sy
				);
				lv.push(i - 1);
			}
		}
		contourSegs = new Float32Array(out);
		contourLevels = new Uint8Array(lv);
	}

	function writeContours(opacity: number) {
		const m = mul('param') * opacity;
		const key = `${contourSurface}|${m.toFixed(3)}`;
		if (key === contourKey) return;
		contourKey = key;
		const n = Math.min(contourLevels.length, contourLines.capacity);
		if (opacity <= 0.002) {
			contourLines.count = 0;
			return;
		}
		for (let i = 0; i < n; i++) {
			const t = contourLevels[i] / (LEVELS - 1);
			// Contours stay below the bloom threshold on purpose. A 1 px line only reaches full
			// brightness where it is pixel-aligned (the flat middle of each ellipse), so letting
			// it bloom produced a hazy vertical band there and nowhere else.
			const intensity = 0.68 + (0.3 - 0.68) * t;
			contourLines.set(i, {
				a: [contourSegs[i * 4], contourSegs[i * 4 + 1]],
				b: [contourSegs[i * 4 + 2], contourSegs[i * 4 + 3]],
				width: 1.25,
				color: tone('param', intensity, 0.9 * m)
			});
		}
		contourLines.count = n;
		contourLines.upload();
	}

	function updateOnce() {
		steps++;
		if (mode === '1d') {
			const g = curve.grad(w);
			trail1.push(w);
			if (trail1.length > TRAIL_1D) trail1.shift();
			w = w - lr * g;
			if (!Number.isFinite(w) || Math.abs(w) > DIVERGE_AT) {
				diverged = true;
				running = false;
				if (!Number.isFinite(w)) w = DIVERGE_AT * 2;
			} else if (Math.abs(curve.grad(w)) < 1e-3) running = false;
			return;
		}
		const s = surface();
		const active = race ? 3 : 1;
		let allDone = true;
		for (let i = 0; i < active; i++) {
			const r = racers[i];
			if (r.frozen) continue;
			const g = s.grad(r.st.p[0], r.st.p[1]);
			const nx = step(r.kind, r.st, g, lr, beta);
			const [px, py] = nx.p;
			if (
				!Number.isFinite(px) ||
				!Number.isFinite(py) ||
				Math.abs(px) > 3 * DOMAIN.x[1] ||
				Math.abs(py) > 3 * DOMAIN.y[1]
			) {
				r.frozen = true;
				continue;
			}
			const last = r.trail[r.trail.length - 1];
			r.st = nx;
			if (Math.hypot(px - last[0], py - last[1]) >= 1e-4) {
				r.trail.push([px, py]);
				if (r.trail.length > TRAIL_2D) r.trail.shift();
			}
			const g2 = s.grad(px, py);
			if (Math.hypot(g2[0], g2[1]) >= 1e-3) allDone = false;
		}
		if (allDone) running = false;
	}

	function resetIfModeChanged() {
		const next: Mode = map >= 0.5 ? '2d' : '1d';
		if (next !== mode) {
			mode = next;
			reset();
		}
	}

	function pointer(e: ScenePointer) {
		if (e.type !== 'down' || mode !== '2d') return;
		const [x, y] = ctx.view.unproject(e.x, e.y);
		if (x < DOMAIN.x[0] || x > DOMAIN.x[1] || y < DOMAIN.y[0] || y > DOMAIN.y[1]) return;
		customStart = [x, y];
		reset();
	}

	function build1D(o: number, labels: Label[]) {
		const v = ctx.view;
		const wc = clamp(w, -W_MAX, W_MAX);
		const g = curve.grad(wc);
		let n = lines.count;
		let d = dots.count;
		let gl = glows.count;

		// curve with soft halo
		const pts: Array<[number, number]> = [];
		for (let i = 0; i <= 96; i++) {
			const x = -W_MAX + (2 * W_MAX * i) / 96;
			pts.push([wx(x), ly(curve.f(x))]);
		}
		const pm = mul('param') * o;
		n = lines.setPolyline(n, pts, { width: 8, color: tone('param', 0.8, 0.1 * pm) });
		n = lines.setPolyline(n, pts, { width: 2, color: tone('param', 1.6, 0.95 * pm) });

		// baseline and minimum marker
		const mm = mul('muted') * o;
		lines.set(n++, {
			a: [-W_MAX - 0.2, Y_BASE],
			b: [W_MAX + 0.2, Y_BASE],
			width: 1,
			color: tone('muted', 1, 0.7 * mm)
		});
		dots.set(d++, { x: 0, y: Y_BASE, radius: 5, ring: 1.5, color: tone('ok', 1.4, 0.9 * mul('ok') * o) });

		// trail
		const gm = mul('grad') * o;
		for (let i = 0; i < trail1.length; i++) {
			const tw = clamp(trail1[i], -W_MAX, W_MAX);
			const a = 0.15 + (0.55 * (i + 1)) / trail1.length;
			dots.set(d++, { x: wx(tw), y: ly(curve.f(tw)), radius: 2.5, color: tone('grad', 1.2, a * gm) });
		}

		// step arrow
		const dm = mul('data') * o;
		const target = clamp(w - lr * curve.grad(w), -W_MAX, W_MAX);
		if (Math.abs(target - wc) > 0.03) {
			const dir = Math.sign(target - wc);
			const col = tone('data', 2, 0.95 * dm);
			lines.set(n++, { a: [wc, Y_BASE], b: [target, Y_BASE], width: 2.5, color: col });
			lines.set(n++, { a: [target, Y_BASE], b: [target - dir * 0.13, Y_BASE + 0.08], width: 2.5, color: col });
			lines.set(n++, { a: [target, Y_BASE], b: [target - dir * 0.13, Y_BASE - 0.08], width: 2.5, color: col });
		}

		// drop line, tangent, ball
		const by = ly(curve.f(wc));
		lines.set(n++, { a: [wc, by], b: [wc, Y_BASE], width: 1, color: tone('grad', 1, 0.45 * gm) });
		const tx = 1;
		const ty = g * Y_PER_LOSS;
		const tl = Math.hypot(tx, ty);
		const hx = ((tx / tl) * 1.2) / 2;
		const hy = ((ty / tl) * 1.2) / 2;
		lines.set(n++, {
			a: [wc - hx, by - hy],
			b: [wc + hx, by + hy],
			width: 2,
			color: tone('grad', 2.5, gm)
		});
		glows.set(gl++, { x: wc, y: by, radius: 30, color: tone('grad', 2, 0.5 * gm), softness: 1 });
		dots.set(d++, { x: wc, y: by, radius: 8, color: tone('grad', 3, gm) });

		lines.count = n;
		dots.count = d;
		glows.count = gl;

		// labels
		const [bpx, bpy] = v.project(wc, by);
		const [, basePy] = v.project(0, Y_BASE);
		const [, axisPy] = v.project(0, Y_BASE);
		const [centerPx] = v.project(0, 0);
		const side = (wc >= 0 ? 1 : -1) * (Math.abs(wc) > 1.5 ? -1 : 1);
		const slopeDy = Math.abs(wc) > 1.5 ? -4 : 26;
		labels.push(
			{ id: 'w', x: bpx, y: basePy + 22, text: `w = ${fmt(w)}`, tone: 'grad', mono: true, opacity: gm },
			{ id: 'axis', x: centerPx, y: axisPy + 44, text: 'weight w', tone: 'muted', opacity: mm },
			{
				id: 'slope',
				x: bpx + side * 36,
				y: bpy + slopeDy,
				text: `slope ${g >= 0 ? '+' : ''}${fmt(curve.grad(w))}`,
				tone: 'grad',
				mono: true,
				align: side > 0 ? 'left' : 'right',
				opacity: gm
			}
		);
		const [lpx, lpy] = v.project(-W_MAX, Y_TOP);
		labels.push({ id: 'loss', x: lpx - 14, y: lpy, text: 'loss', tone: 'muted', align: 'right', opacity: mm });
		if (diverged) {
			const [cx, cy] = v.project(0, 0.2);
			labels.push({ id: 'diverged', x: cx, y: cy, text: 'Diverged — press Reset', tone: 'neg', opacity: o });
		}
	}

	function build2D(o: number, labels: Label[]) {
		const v = ctx.view;
		const s = surface();
		let n = lines.count;
		let d = dots.count;
		let gl = glows.count;

		// minima markers
		const om = mul('ok') * o;
		for (const [x, y] of s.minima) {
			dots.set(d++, { x, y, radius: 7, ring: 1.5, color: tone('ok', 1.6, 0.9 * om) });
		}
		const [gx, gy] = s.minima[0];
		const [mpx, mpy] = v.project(gx, gy);
		labels.push({ id: 'minimum', x: mpx, y: mpy + 26, text: 'minimum', tone: 'ok', opacity: om });

		// start marker
		const st = startPoint();
		dots.set(d++, { x: st[0], y: st[1], radius: 6, ring: 1.2, color: tone('muted', 1.8, 0.9 * mul('muted') * o) });

		const active = race ? 3 : 1;
		// draw the slower optimizers first so plain descent stays on top
		for (let k = active - 1; k >= 0; k--) {
			const r = racers[k];
			const m = mul(r.tone) * o;
			const tr = r.trail;
			const len = tr.length;
			for (let i = 0; i + 1 < len && n < lines.capacity; i++) {
				const a = 0.25 + (0.7 * (i + 1)) / len;
				lines.set(n++, {
					a: tr[i],
					b: tr[i + 1],
					width: 1.8,
					color: tone(r.tone, 1.6, a * m)
				});
			}
			for (let i = 0; i < len - 1 && d < dots.capacity - 8; i++) {
				const a = 0.3 + (0.6 * (i + 1)) / len;
				dots.set(d++, { x: tr[i][0], y: tr[i][1], radius: 2, color: tone(r.tone, 1.4, a * m) });
			}
			if (!r.frozen) {
				const [x, y] = r.st.p;
				glows.set(gl++, { x, y, radius: 26, color: tone(r.tone, 2.2, 0.55 * m), softness: 1 });
				dots.set(d++, { x, y, radius: 6, color: tone(r.tone, 3, m) });
			}
		}
		lines.count = n;
		dots.count = d;
		glows.count = gl;
	}

	const scene: Scene = {
		init(c) {
			ctx = c;
			camera = new Camera2D();
			contourLines = new Lines(c, { capacity: 16000, blend: 'alpha' });
			lines = new Lines(c, { capacity: 2800, blend: 'alpha' });
			glows = new Points(c, { capacity: 16, blend: 'additive' });
			dots = new Points(c, { capacity: 2200, blend: 'alpha' });
			reset();
		},
		setParams(p) {
			const prevScenario = scenario;
			const prevSurface = surfaceId;
			const prevRace = race;
			lr = p.lr ?? lr;
			beta = p.beta ?? beta;
			map = p.map ?? map;
			surfaceId = Math.round(p.surface ?? surfaceId);
			race = (p.race ?? race) >= 0.5 ? 1 : 0;
			scenario = Math.round(p.scenario ?? scenario);
			const changedKey = scenario !== prevScenario || surfaceId !== prevSurface;
			if (changedKey) customStart = null;
			const next: Mode = map >= 0.5 ? '2d' : '1d';
			const modeFlip = next !== mode;
			mode = next;
			if (!paramsSeen || changedKey || race !== prevRace || modeFlip) reset();
			paramsSeen = true;
			if (surfaceId !== contourSurface) buildContours();
		},
		update(frame) {
			const v = ctx.view;
			fitCamera(camera, v, WORLD, lessonInsets(v));
			v.setViewProj(camera.matrix(v.aspect));
			resetIfModeChanged();

			if (running && frame.dt > 0) {
				// Reduced motion: discrete steps, at most 3 per second.
				const rate = mode === '1d' || ctx.reducedMotion ? 3 : 30;
				acc += frame.dt * rate;
				let n = Math.floor(acc);
				acc -= n;
				if (n > 5) {
					n = 5;
					acc = 0;
				}
				for (let i = 0; i < n && running; i++) updateOnce();
			}

			const o2 = clamp(map, 0, 1);
			const o1 = 1 - o2;
			lines.count = 0;
			dots.count = 0;
			glows.count = 0;
			const labels: Label[] = [];
			if (o1 > 0.002) build1D(o1, labels);
			if (o2 > 0.002) build2D(o2, labels);
			writeContours(o2);
			lines.upload();
			dots.upload();
			glows.upload();
			ctx.labels.set(labels);

			const loss = mode === '1d' ? curve.f(w) : surface().f(racers[0].st.p[0], racers[0].st.p[1]);
			if (loss !== lastLoss && !(Number.isNaN(loss) && Number.isNaN(lastLoss))) {
				lastLoss = loss;
				ctx.emit('loss', loss);
			}
			if (steps !== lastSteps) {
				lastSteps = steps;
				ctx.emit('steps', steps);
			}
		},
		draw(pass) {
			contourLines.draw(pass);
			lines.draw(pass);
			glows.draw(pass);
			dots.draw(pass);
		},
		command(id) {
			if (id === 'run') {
				if (mode === '1d' && diverged) return;
				running = !running;
				acc = 0;
			} else if (id === 'step') {
				running = false;
				if (!(mode === '1d' && diverged)) updateOnce();
			} else if (id === 'reset') {
				reset();
			}
		},
		highlight(t) {
			hl = t;
			contourKey = '';
		},
		pointer,
		dispose() {
			contourLines.destroy();
			lines.destroy();
			glows.destroy();
			dots.destroy();
		}
	};
	return scene;
};
