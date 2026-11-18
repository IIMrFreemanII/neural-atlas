import { mat4 } from 'wgpu-matrix';
import { Camera2D, OrbitCamera } from '../../../gpu/camera.ts';
import { mixColor, tone } from '../../../gpu/colors.ts';
import { Lines } from '../../../gpu/primitives/lines.ts';
import { SurfaceMesh } from '../../../gpu/primitives/mesh.ts';
import { Points } from '../../../gpu/primitives/points.ts';
import type { Label, Scene, SceneContext, SceneFactory, ScenePointer, Tone } from '../../../gpu/types.ts';
import { ViewUniforms } from '../../../gpu/view.ts';
import { fitCamera, lessonInsets, type WorldRect } from '../../fit.ts';
import { RANGE, data, loss, lossGrid, predict, type ModelId } from './model.ts';

const N = 72;
const H_MAX = 1.6;
const AXIS_TOP = 1.95;
const PLOT: WorldRect = { x0: -2, x1: 2, y0: -2, y1: 2 };
const PLOT_LIM = 1.8;
const CURVE_N = 80;
const TRAIL_MAX = 60;
const EASE_SECONDS = 0.4;

const clamp = (v: number, a: number, b: number) => Math.min(b, Math.max(a, v));
const smooth = (t: number) => t * t * (3 - 2 * t);
const fmt = (v: number) => (v < -0.005 ? '−' : '') + Math.abs(v).toFixed(2);
const fmtAdd = (v: number) => (v < -0.005 ? '− ' : '+ ') + Math.abs(v).toFixed(2);

interface Grid {
	heights: Float32Array<ArrayBuffer>;
	min: number;
	argmin: { w: number; b: number };
}

export const createLossLandscapeScene: SceneFactory = () => {
	let ctx: SceneContext;
	let leftView: ViewUniforms;
	let cam2d: Camera2D;
	let orbit: OrbitCamera;
	let mesh: SurfaceMesh;
	let lines2: Lines;
	let glow2: Points;
	let dots2: Points;
	let lines3: Lines;
	let glow3: Points;
	let dots3: Points;

	let w = 0.2;
	let b = 0.6;
	let model: ModelId = 0;
	let resid = 0;
	let surf = 0;
	let hl: Tone | null = null;
	let seen = false;
	let okT = 0;
	let lastLoss = NaN;

	const grids = new Map<ModelId, Grid>();
	const from = new Float32Array(N * N);
	let to: Float32Array = new Float32Array(N * N);
	let easeT = 1;
	let trail: Array<[number, number]> = [];
	let dragging = false;
	let lastX = 0;
	let lastY = 0;

	const mul = (t: Tone) => (hl === null || hl === t ? 1 : 0.3);

	function grid(m: ModelId): Grid {
		let g = grids.get(m);
		if (!g) {
			const lg = lossGrid(m, N, N);
			const span = Math.max(1e-9, lg.max - lg.min);
			const heights = new Float32Array(N * N);
			for (let i = 0; i < heights.length; i++) {
				heights[i] = H_MAX * Math.sqrt(Math.max(0, (lg.values[i] - lg.min) / span));
			}
			g = { heights, min: lg.min, argmin: lg.argmin };
			grids.set(m, g);
		}
		return g;
	}

	/** Height of the displayed surface at (w, b), bilinear on the mesh grid. */
	function surfaceY(pw: number, pb: number): number {
		const h = mesh.heights;
		const fc = clamp(((pw - RANGE.w[0]) / (RANGE.w[1] - RANGE.w[0])) * (N - 1), 0, N - 1);
		const fr = clamp(((pb - RANGE.b[0]) / (RANGE.b[1] - RANGE.b[0])) * (N - 1), 0, N - 1);
		const c0 = Math.min(N - 2, Math.floor(fc));
		const r0 = Math.min(N - 2, Math.floor(fr));
		const tc = fc - c0;
		const tr = fr - r0;
		const a = h[r0 * N + c0] * (1 - tc) + h[r0 * N + c0 + 1] * tc;
		const c = h[(r0 + 1) * N + c0] * (1 - tc) + h[(r0 + 1) * N + c0 + 1] * tc;
		return a * (1 - tr) + c * tr;
	}

	function setModel(m: ModelId, instant: boolean) {
		model = m;
		const g = grid(m);
		if (instant || ctx.reducedMotion) {
			mesh.heights.set(g.heights);
			easeT = 1;
		} else {
			from.set(mesh.heights);
			to = g.heights;
			easeT = 0;
		}
		trail = [];
	}

	function onPointer(e: ScenePointer) {
		if (e.type === 'down') {
			if (surf > 0.5 && e.x > ctx.view.width * 0.4) {
				dragging = true;
				lastX = e.x;
				lastY = e.y;
			}
		} else if (e.type === 'move' && dragging) {
			orbit.azimuth -= (e.x - lastX) * 0.008;
			orbit.elevation = clamp(orbit.elevation + (e.y - lastY) * 0.008, 0.15, 1.35);
			lastX = e.x;
			lastY = e.y;
		} else if (e.type === 'up' || e.type === 'leave') {
			dragging = false;
		}
	}

	function build2D(dt: number, labels: Label[]) {
		const v = leftView;
		const insets = lessonInsets(ctx.view);
		insets.right = insets.left + (ctx.view.width * 0.6 - insets.left) * surf;
		fitCamera(cam2d, ctx.view, PLOT, insets);
		v.setSize(ctx.view.width, ctx.view.height, ctx.view.dpr);
		v.setViewProj(cam2d.matrix(ctx.view.aspect));
		v.flush();

		const pts = data(model);
		const g = grid(model);
		const L = loss(model, w, b);
		const near = L <= g.min * 1.02 + 1e-6 ? 1 : 0;
		okT = ctx.reducedMotion ? near : okT + (near - okT) * Math.min(1, dt * 10);
		const curveColor = (intensity: number, alpha: number) =>
			mixColor(tone('param', intensity, alpha), tone('ok', intensity, alpha), okT);

		let n = 0;
		const mm = mul('muted');
		lines2.set(n++, { a: [-PLOT_LIM, 0], b: [PLOT_LIM, 0], width: 1, color: tone('muted', 1, 0.6 * mm) });
		lines2.set(n++, { a: [0, -PLOT_LIM], b: [0, PLOT_LIM], width: 1, color: tone('muted', 1, 0.6 * mm) });

		const curve: Array<[number, number]> = [];
		for (let i = 0; i < CURVE_N; i++) {
			const x = -PLOT_LIM + (2 * PLOT_LIM * i) / (CURVE_N - 1);
			curve.push([x, clamp(predict(model, w, b, x), -PLOT_LIM, PLOT_LIM)]);
		}
		const pm = mul('param');
		n = lines2.setPolyline(n, curve, { width: 9, color: curveColor(0.8, 0.1 * pm) });
		n = lines2.setPolyline(n, curve, { width: 2.5, color: curveColor(1.6, 0.95 * pm) });

		const gm = mul('grad') * resid;
		if (resid > 0.002) {
			for (const p of pts) {
				const py = clamp(predict(model, w, b, p.x), -PLOT_LIM, PLOT_LIM);
				lines2.set(n++, { a: [p.x, p.y], b: [p.x, py], width: 2, color: tone('grad', 1.4, 0.9 * gm) });
			}
		}
		lines2.count = n;

		const dm = mul('data');
		let d = 0;
		let gl = 0;
		for (const p of pts) {
			glow2.set(gl++, { x: p.x, y: p.y, radius: 16, color: tone('data', 1.6, 0.45 * dm), softness: 1 });
			dots2.set(d++, { x: p.x, y: p.y, radius: 4.5, color: tone('data', 2, dm) });
		}
		glow2.count = gl;
		dots2.count = d;

		const [px, pyRaw] = v.project(-PLOT_LIM, PLOT_LIM + 0.1);
		// Keep the formula below the legend chips, which wrap to two rows in this lesson.
		const py = Math.max(pyRaw, 104);
		const text =
			model === 0
				? `y = ${fmt(w)}·x ${fmtAdd(b)}`
				: `y = tanh(${fmt(w)}·x ${fmtAdd(b)})`;
		labels.push({ id: 'formula', x: px, y: py, text, tone: 'param', mono: true, align: 'left', opacity: pm });
		const [xx, xy] = v.project(PLOT_LIM + 0.15, 0);
		const [yx, yy] = v.project(0, PLOT_LIM + 0.15);
		labels.push(
			{ id: 'ax-x', x: xx, y: xy, text: 'x', tone: 'muted', opacity: mm },
			{ id: 'ax-y', x: yx, y: yy, text: 'y', tone: 'muted', opacity: mm }
		);
	}

	function build3D(dt: number, labels: Label[]) {
		const v = ctx.view;
		const a = v.aspect;
		const ins = lessonInsets(v);
		orbit.target = [0, 0.7, 0];
		const region = Math.min(v.width * 0.6 - ins.left, v.height - ins.top - ins.bottom);
		orbit.distance = (6.4 * v.height) / (0.828 * Math.max(200, region));
		const shift = mat4.translation([0.42, (ins.bottom - ins.top) / v.height, 0]);
		v.setViewProj(mat4.multiply(shift, orbit.matrix(a)) as Float32Array);

		// height ease
		if (easeT < 1) {
			easeT = Math.min(1, easeT + dt / EASE_SECONDS);
			const e = smooth(easeT);
			for (let i = 0; i < mesh.heights.length; i++) mesh.heights[i] = from[i] + (to[i] - from[i]) * e;
		}
		const pm = mul('param');
		mesh.opacity = surf * pm;
		mesh.upload();

		// trail
		const last = trail[trail.length - 1];
		if (!last || Math.hypot(last[0] - w, last[1] - b) > 0.004) {
			trail.push([w, b]);
			if (trail.length > TRAIL_MAX) trail.shift();
		}

		const gm = mul('grad') * surf;
		const mu = mul('muted') * surf;
		let n = 0;
		let d = 0;
		let gl = 0;

		// floor frame
		const [w0, w1] = RANGE.w;
		const [b0, b1] = RANGE.b;
		const fr = tone('muted', 0.8, 0.6 * mu);
		const frDim = tone('muted', 0.8, 0.3 * mu);
		lines3.set(n++, { a: [w0, 0, b0], b: [w1, 0, b0], width: 1.25, color: fr });
		lines3.set(n++, { a: [w0, 0, b0], b: [w0, 0, b1], width: 1.25, color: fr });
		lines3.set(n++, { a: [w1, 0, b0], b: [w1, 0, b1], width: 1, color: frDim });
		lines3.set(n++, { a: [w0, 0, b1], b: [w1, 0, b1], width: 1, color: frDim });
		const cx = Math.sin(orbit.azimuth) >= 0 ? w0 : w1;
		const cz = Math.cos(orbit.azimuth) >= 0 ? b0 : b1;
		lines3.set(n++, { a: [cx, 0, cz], b: [cx, AXIS_TOP, cz], width: 1.25, color: fr });

		// ball, drop line
		const by = surfaceY(w, b) + 0.05;
		lines3.set(n++, { a: [w, by, b], b: [w, 0, b], width: 1, color: tone('grad', 1, 0.55 * gm) });
		lines3.count = n;

		for (let i = 0; i < trail.length - 1; i++) {
			const [tw, tb] = trail[i];
			const f = (i + 1) / trail.length;
			dots3.set(d++, { x: tw, y: surfaceY(tw, tb) + 0.04, z: tb, radius: 2.5, color: tone('grad', 1.3, 0.7 * f * gm) });
		}
		glow3.set(gl++, { x: w, y: by, z: b, radius: 24, color: tone('grad', 2, 0.55 * gm), softness: 1 });
		dots3.set(d++, { x: w, y: by, z: b, radius: 7, color: tone('grad', 3, gm) });

		// minimum marker
		const am = grid(model).argmin;
		const my = surfaceY(am.w, am.b);
		const om = mul('ok') * surf;
		dots3.set(d++, { x: am.w, y: my + 0.03, z: am.b, radius: 9, ring: 1.8, color: tone('ok', 1.6, 0.95 * om) });
		dots3.count = d;
		glow3.count = gl;

		// labels
		const [mx, mpy] = v.project(am.w, my + 0.03, am.b);
		labels.push({ id: 'lowest', x: mx, y: mpy + 30, text: 'lowest loss', tone: 'ok', opacity: om });
		const [wx, wy] = v.project(w1 + 0.2, 0, b0);
		const [bx, byy] = v.project(w0, 0, b1 + 0.3);
		const [tx, ty] = v.project(cx, AXIS_TOP + 0.15, cz);
		labels.push(
			{ id: 'ax-w', x: wx, y: wy, text: 'w', tone: 'muted', opacity: mu },
			{ id: 'ax-b', x: bx, y: byy, text: 'b', tone: 'muted', opacity: mu },
			{ id: 'ax-loss', x: tx, y: ty, text: 'loss', tone: 'muted', opacity: mu }
		);
	}

	const scene: Scene = {
		init(c) {
			ctx = c;
			leftView = new ViewUniforms(c.gpu);
			const left = { gpu: c.gpu, target: c.target, view: leftView };
			cam2d = new Camera2D();
			orbit = new OrbitCamera();
			orbit.azimuth = 0.7;
			orbit.elevation = 0.55;
			lines2 = new Lines(left, { capacity: 256, blend: 'alpha' });
			glow2 = new Points(left, { capacity: 16, blend: 'additive' });
			dots2 = new Points(left, { capacity: 16, blend: 'alpha' });

			mesh = new SurfaceMesh(c, { cols: N, rows: N });
			mesh.rect = [RANGE.w[0], RANGE.b[0], RANGE.w[1], RANGE.b[1]];
			mesh.range = [0, H_MAX];
			lines3 = new Lines(c, { capacity: 16, blend: 'alpha', depthTest: true });
			glow3 = new Points(c, { capacity: 2, blend: 'additive', depthTest: true });
			dots3 = new Points(c, { capacity: TRAIL_MAX + 4, blend: 'alpha', depthTest: true });
			mesh.heights.set(grid(0).heights);
			mesh.upload();
		},
		setParams(p) {
			w = p.w ?? w;
			b = p.b ?? b;
			resid = clamp(p.resid ?? resid, 0, 1);
			surf = clamp(p.surface ?? surf, 0, 1);
			const m = (Math.round(p.model ?? model) === 1 ? 1 : 0) as ModelId;
			if (!seen || m !== model) setModel(m, !seen);
			seen = true;
		},
		update(frame) {
			const labels: Label[] = [];
			build2D(frame.dt, labels);
			if (surf > 0.003) build3D(frame.dt, labels);
			lines2.upload();
			glow2.upload();
			dots2.upload();
			if (surf > 0.003) {
				lines3.upload();
				glow3.upload();
				dots3.upload();
			}
			ctx.labels.set(labels);
			const L = loss(model, w, b);
			if (L !== lastLoss) {
				lastLoss = L;
				ctx.emit('loss', L);
			}
		},
		draw(pass) {
			lines2.draw(pass);
			glow2.draw(pass);
			dots2.draw(pass);
			if (surf > 0.003) {
				mesh.draw(pass);
				lines3.draw(pass);
				glow3.draw(pass);
				dots3.draw(pass);
			}
		},
		highlight(t) {
			hl = t;
		},
		pointer: onPointer,
		dispose() {
			lines2.destroy();
			glow2.destroy();
			dots2.destroy();
			mesh.destroy();
			lines3.destroy();
			glow3.destroy();
			dots3.destroy();
			leftView.destroy();
		}
	};
	return scene;
};
