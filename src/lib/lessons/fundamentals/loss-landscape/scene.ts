import { mat4 } from 'wgpu-matrix';
import { Camera2D, OrbitRig, type Pose } from '../../../gpu/camera.ts';
import { mixColor, paperFill, sheetFill, tone } from '../../../gpu/colors.ts';
import { Lines } from '../../../gpu/primitives/lines.ts';
import { SurfaceMesh } from '../../../gpu/primitives/mesh.ts';
import { Points } from '../../../gpu/primitives/points.ts';
import { Shadows } from '../../../gpu/primitives/shadows.ts';
import { easeInOut, type Vec3 } from '../../../gpu/space.ts';
import type { Insets, Label, Scene, SceneContext, SceneFactory, ScenePointer, Tone } from '../../../gpu/types.ts';
import { ViewUniforms } from '../../../gpu/view.ts';
import { contourSegments } from '../../../ml/contours.ts';
import { SceneOrbit } from '../../shared/orbit.ts';
import { fitCamera, insetAspect, insetMatrix, lessonInsets, type WorldRect } from '../../fit.ts';
import { RANGE, data, loss, lossGrid, predict, type ModelId } from './model.ts';

const N = 72;
const H_MAX = 1.6;
const AXIS_TOP = 1.9;
const PLOT: WorldRect = { x0: -2, x1: 2, y0: -2, y1: 2 };
const PLOT_LIM = 1.8;
const CURVE_N = 80;
const TRAIL_MAX = 240;
const EASE_SECONDS = 0.4;
const REVEAL_SECONDS = 0.9;
const FLOOR_Y = -0.05;
const FLOOR_H = 2.3;
const BALL_R = 0.11;
const CONTOUR_LEVELS = 7;
const CONTOUR_LIFT = 0.016;
/** Nominal eye distance: the frustum is tall enough for the figure; narrow stage areas are handled by a zoom. */
const DIST = 8;
/** Share of the uncovered stage width given to the 2D plot while the surface is on. */
const LEFT_SHARE = 0.46;

/** Camera poses (lesson param `pose`): 1 surface, 2 bowl minimum, 3 tanh plateaus, 4 the way down. */
const POSES: Record<1 | 2 | 3 | 4, Pose> = {
	1: { target: [0, 0.55, 0], azimuth: 0.75, elevation: 0.85, distance: DIST },
	2: { target: [0, 0.4, 0], azimuth: 1.2, elevation: 0.4, distance: DIST },
	3: { target: [0, 0.55, 0], azimuth: -0.9, elevation: 0.5, distance: DIST },
	4: { target: [0, 0.5, 0], azimuth: 0.35, elevation: 0.75, distance: DIST }
};

const clamp = (v: number, a: number, b: number) => Math.min(b, Math.max(a, v));
const smooth = (t: number) => t * t * (3 - 2 * t);
const fmt = (v: number) => (v < -0.005 ? '−' : '') + Math.abs(v).toFixed(2);
const fmtAdd = (v: number) => (v < -0.005 ? '− ' : '+ ') + Math.abs(v).toFixed(2);

interface Grid {
	heights: Float32Array<ArrayBuffer>;
	min: number;
	argmin: { w: number; b: number };
	/** Contour segments of the displayed heights: [w0, y, b0, w1, b1, sweep] per segment (sweep = angle fraction round the minimum). */
	contours: Float32Array;
}

interface Fit {
	s: number;
	cx: number;
	cy: number;
}

/** Scale and shift that fit the projected bounding box (8 corners) of the figure inside clip space with a margin. */
function fitBox(m: Float32Array, box: [number, number, number, number, number, number], margin: number): Fit {
	let x0 = Infinity;
	let x1 = -Infinity;
	let y0 = Infinity;
	let y1 = -Infinity;
	for (const x of [box[0], box[1]]) {
		for (const y of [box[2], box[3]]) {
			for (const z of [box[4], box[5]]) {
				const cw = m[3] * x + m[7] * y + m[11] * z + m[15] || 1e-6;
				const px = (m[0] * x + m[4] * y + m[8] * z + m[12]) / cw;
				const py = (m[1] * x + m[5] * y + m[9] * z + m[13]) / cw;
				x0 = Math.min(x0, px);
				x1 = Math.max(x1, px);
				y0 = Math.min(y0, py);
				y1 = Math.max(y1, py);
			}
		}
	}
	const s = Math.min(2 / Math.max(1e-6, x1 - x0), 2 / Math.max(1e-6, y1 - y0)) / (1 + margin);
	const cx = (x0 + x1) / 2;
	const cy = (y0 + y1) / 2;
	return { s, cx, cy };
}

const fitMatrix = (f: Fit) => new Float32Array([f.s, 0, 0, 0, 0, f.s, 0, 0, 0, 0, 1, 0, -f.s * f.cx, -f.s * f.cy, 0, 1]);

export const createLossLandscapeScene: SceneFactory = () => {
	let ctx: SceneContext;
	let leftView: ViewUniforms;
	let cam2d: Camera2D;
	let rig: OrbitRig;
	let orbit: SceneOrbit;
	let fitRig: OrbitRig;
	let fit: Fit = { s: 1, cx: 0, cy: 0 };
	let fitSeen = false;
	let mesh: SurfaceMesh;
	let shadows: Shadows;
	let lines2: Lines;
	let dots2: Points;
	let floorL: Lines;
	let contourL: Lines;
	let trailL: Lines;
	let dropL: Lines;
	let ball: Points;
	let ringP: Points;
	let footP: Points;

	let w = 0.2;
	let b = 0.6;
	let model: ModelId = 0;
	let resid = 0;
	let surf = 0;
	let poseParam = 0;
	let hl: Tone | null = null;
	let seen = false;
	let okT = 0;
	let lastLoss = NaN;

	// layout, recomputed every frame
	let split = 0;
	let ins2: Insets = { top: 0, right: 0, bottom: 0, left: 0 };
	let ins3: Insets = { top: 0, right: 0, bottom: 0, left: 0 };

	// camera state
	let poseKey = -1;
	let engaged = false;
	let contourT = 1;
	let contourOn = 0;
	let trailT = 1;

	const grids = new Map<ModelId, Grid>();
	const from = new Float32Array(N * N);
	let to: Float32Array = new Float32Array(N * N);
	let easeT = 1;
	let trail: Array<[number, number]> = [];

	const mul = (t: Tone) => (hl === null || hl === t ? 1 : 0.3);

	function contoursOf(heights: Float32Array, am: { w: number; b: number }): Float32Array {
		const out: number[] = [];
		const sw = (RANGE.w[1] - RANGE.w[0]) / (N - 1);
		const sb = (RANGE.b[1] - RANGE.b[0]) / (N - 1);
		for (let k = 1; k <= CONTOUR_LEVELS; k++) {
			const level = (H_MAX * k) / (CONTOUR_LEVELS + 1);
			for (const [x0, r0, x1, r1] of contourSegments(heights, N, N, level)) {
				const w0 = RANGE.w[0] + x0 * sw;
				const b0 = RANGE.b[0] + r0 * sb;
				const w1 = RANGE.w[0] + x1 * sw;
				const b1 = RANGE.b[0] + r1 * sb;
				const ang = Math.atan2((b0 + b1) / 2 - am.b, (w0 + w1) / 2 - am.w);
				out.push(w0, level + CONTOUR_LIFT, b0, w1, b1, ang / (2 * Math.PI) + 0.5);
			}
		}
		return new Float32Array(out);
	}

	function grid(m: ModelId): Grid {
		let g = grids.get(m);
		if (!g) {
			const lg = lossGrid(m, N, N);
			const span = Math.max(1e-9, lg.max - lg.min);
			const heights = new Float32Array(N * N);
			for (let i = 0; i < heights.length; i++) {
				heights[i] = H_MAX * Math.sqrt(Math.max(0, (lg.values[i] - lg.min) / span));
			}
			g = { heights, min: lg.min, argmin: lg.argmin, contours: contoursOf(heights, lg.argmin) };
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

	/** The pose for the current params: the lesson's `pose`, or a sensible one by model when it is unset. */
	function wantedPose(): 1 | 2 | 3 | 4 {
		const p = Math.round(poseParam);
		if (p >= 1 && p <= 4) return p as 1 | 2 | 3 | 4;
		return model === 1 ? 3 : 1;
	}

	/** Glide to the step's pose and start the draw-ins when the figure is (re)entered. */
	function syncPose() {
		const key = surf > 0.5 ? wantedPose() : -1;
		if (key === poseKey) return;
		poseKey = key;
		orbit.setDraggable(key >= 0);
		if (key < 0) {
			engaged = false;
			return;
		}
		orbit.stepPose(POSES[key as 1 | 2 | 3 | 4]);
		if (key === 2) contourT = ctx.reducedMotion ? 1 : 0;
		if (key === 4) {
			trail = [];
			trailT = ctx.reducedMotion ? 1 : 0;
		}
	}

	function onPointer(e: ScenePointer) {
		if (e.type === 'down') {
			engaged = surf > 0.5 && e.x > split;
			if (!engaged) return;
		} else if (!engaged) {
			return;
		}
		orbit.pointer(e);
		if (e.type === 'up' || e.type === 'leave') engaged = false;
	}

	function layout() {
		const v = ctx.view;
		const ins = lessonInsets(v);
		const avail = Math.max(1, v.width - ins.left - ins.right);
		split = ins.left + avail * LEFT_SHARE;
		ins2 = { ...ins, right: ins.right + (v.width - split - ins.right) * surf };
		ins3 = { top: ins.top, bottom: ins.bottom, left: split, right: ins.right };
	}

	function build2D(dt: number, labels: Label[]) {
		const v = leftView;
		fitCamera(cam2d, ctx.view, PLOT, ins2);
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

		const gm = mul('grad') * resid;
		if (resid > 0.002) {
			for (const p of pts) {
				const py = clamp(predict(model, w, b, p.x), -PLOT_LIM, PLOT_LIM);
				lines2.set(n++, { a: [p.x, p.y], b: [p.x, py], width: 2, color: tone('grad', 1, 0.9 * gm) });
			}
		}

		const curve: Array<[number, number]> = [];
		for (let i = 0; i < CURVE_N; i++) {
			const x = -PLOT_LIM + (2 * PLOT_LIM * i) / (CURVE_N - 1);
			curve.push([x, clamp(predict(model, w, b, x), -PLOT_LIM, PLOT_LIM)]);
		}
		const pm = mul('param');
		n = lines2.setPolyline(n, curve, { width: 2.5, color: curveColor(1, 0.95 * pm) });
		lines2.count = n;

		// Data: ink dots with a paper ring so the curve and the residuals stop short of them.
		const dm = mul('data');
		let d = 0;
		for (const p of pts) {
			dots2.set(d++, { x: p.x, y: p.y, radius: 6.5, color: paperFill(dm) });
			dots2.set(d++, { x: p.x, y: p.y, radius: 4.5, color: tone('data', 1, dm) });
		}
		dots2.count = d;

		// Centered above the top of the y axis (and its "y" label) so no axis line crosses the text.
		const [px, pyRaw] = v.project(0, PLOT_LIM + 0.15);
		// Keep the formula below the legend chips, which wrap to two rows in this lesson.
		const py = Math.max(pyRaw - 24, 104);
		const text = model === 0 ? `y = ${fmt(w)}·x ${fmtAdd(b)}` : `y = tanh(${fmt(w)}·x ${fmtAdd(b)})`;
		labels.push({ id: 'formula', x: px, y: py, text, tone: 'param', mono: true, opacity: pm });
		const [xx, xy] = v.project(PLOT_LIM + 0.15, 0);
		const [yx, yy] = v.project(0, PLOT_LIM + 0.15);
		labels.push(
			{ id: 'ax-x', x: xx, y: xy, text: 'x', tone: 'muted', opacity: mm },
			{ id: 'ax-y', x: yx, y: yy, text: 'y', tone: 'muted', opacity: mm }
		);
	}

	function build3D(dt: number, labels: Label[]) {
		const v = ctx.view;
		orbit.update(dt);
		const a3 = insetAspect(v, ins3);
		// Fit from the step's home pose (not the live one), eased during a glide, so dragging only rotates.
		fitRig.setPose(rig.home);
		const goal = fitBox(fitRig.matrix(a3), [-2.2, 2.2, FLOOR_Y, AXIS_TOP, -2.2, 2.2], 0.08);
		if (!fitSeen || ctx.reducedMotion) fit = goal;
		else {
			const k = Math.min(1, dt * 6);
			fit = { s: fit.s + (goal.s - fit.s) * k, cx: fit.cx + (goal.cx - fit.cx) * k, cy: fit.cy + (goal.cy - fit.cy) * k };
		}
		fitSeen = true;
		const vp = mat4.multiply(insetMatrix(v, ins3), mat4.multiply(fitMatrix(fit), rig.matrix(a3))) as Float32Array;
		v.setViewProj(vp);

		// height ease between models
		if (easeT < 1) {
			easeT = Math.min(1, easeT + dt / EASE_SECONDS);
			const e = smooth(easeT);
			for (let i = 0; i < mesh.heights.length; i++) mesh.heights[i] = from[i] + (to[i] - from[i]) * e;
		}
		const pm = mul('param');
		mesh.opacity = surf * pm;
		// The theme can change under a live scene, so the inks are set every frame.
		mesh.setInk({
			fill: sheetFill(),
			ink: tone('muted', 0.55),
			gridEvery: 4,
			hatch: true,
			contours: false
		});
		mesh.upload();

		const mu = mul('muted') * surf;
		const mg = mul('grad') * surf;
		const mo = mul('ok') * surf;
		const key = poseKey;

		// Floor: a faint ruled grid, its frame, four posts under the surface corners, and the loss axis.
		let n = 0;
		const rule = tone('text', 0.12, mu);
		for (let k = -4; k <= 4; k++) {
			const t = k * 0.5;
			floorL.set(n++, { a: [-FLOOR_H, FLOOR_Y, t], b: [FLOOR_H, FLOOR_Y, t], width: 1, color: rule });
			floorL.set(n++, { a: [t, FLOOR_Y, -FLOOR_H], b: [t, FLOOR_Y, FLOOR_H], width: 1, color: rule });
		}
		const frame = tone('text', 0.3, mu);
		const fc: Vec3[] = [
			[-FLOOR_H, FLOOR_Y, -FLOOR_H],
			[FLOOR_H, FLOOR_Y, -FLOOR_H],
			[FLOOR_H, FLOOR_Y, FLOOR_H],
			[-FLOOR_H, FLOOR_Y, FLOOR_H]
		];
		for (let i = 0; i < 4; i++) floorL.set(n++, { a: fc[i], b: fc[(i + 1) % 4], width: 1, color: frame });
		const [w0, w1] = RANGE.w;
		const [b0, b1] = RANGE.b;
		for (const [cw, cb] of [
			[w0, b0],
			[w1, b0],
			[w1, b1],
			[w0, b1]
		]) {
			floorL.set(n++, { a: [cw, FLOOR_Y, cb], b: [cw, surfaceY(cw, cb), cb], width: 1, color: tone('muted', 0.35, mu) });
		}
		const az = rig.current.azimuth;
		const ex = Math.sin(az);
		const ez = Math.cos(az);
		const cx = ex >= 0 ? -FLOOR_H : FLOOR_H;
		const cz = ez >= 0 ? -FLOOR_H : FLOOR_H;
		floorL.set(n++, { a: [cx, FLOOR_Y, cz], b: [cx, AXIS_TOP, cz], width: 1.25, color: tone('muted', 0.6, mu) });
		floorL.count = n;

		// Contours of the bowl, drawn in like a pen stroke; the mesh's own are off.
		const g = grid(model);
		contourOn = ctx.reducedMotion
			? key === 2
				? 1
				: 0
			: contourOn + ((key === 2 ? 1 : 0) - contourOn) * Math.min(1, dt * 8);
		if (key === 2 && contourT < 1) contourT = Math.min(1, contourT + dt / REVEAL_SECONDS);
		const showContours = contourOn > 0.01 && easeT >= 1;
		if (showContours) {
			const segs = g.contours;
			const cnt = Math.min(contourL.capacity, segs.length / 6);
			const cCol = tone('param', 0.7, contourOn * pm * surf);
			for (let i = 0; i < cnt; i++) {
				const o = i * 6;
				contourL.set(i, {
					a: [segs[o], segs[o + 1], segs[o + 2]],
					b: [segs[o + 3], segs[o + 1], segs[o + 4]],
					width: 1.4,
					color: cCol,
					t0: segs[o + 5],
					t1: segs[o + 5]
				});
			}
			contourL.count = cnt;
			contourL.setReveal(easeInOut(contourT));
		} else {
			contourL.count = 0;
		}

		// The walked path, only on the last step: it grows with the ball and draws in when the step is entered.
		const last = trail[trail.length - 1];
		if (key === 4 && (!last || Math.hypot(last[0] - w, last[1] - b) > 0.004)) {
			trail.push([w, b]);
			if (trail.length > TRAIL_MAX) trail.shift();
		}
		if (key === 4 && trailT < 1) trailT = Math.min(1, trailT + dt / REVEAL_SECONDS);
		if (key === 4 && trail.length > 1) {
			const pts: Vec3[] = trail.map(([tw, tb]) => [tw, surfaceY(tw, tb) + 0.03, tb]);
			trailL.count = trailL.setPolyline(0, pts, { width: 2.4, color: tone('grad', 1, 0.95 * mg) });
			trailL.setReveal(easeInOut(trailT));
		} else {
			trailL.count = 0;
		}

		// The ball sits on the surface at the current setting, with a soft contact shadow under it.
		const by = surfaceY(w, b);
		shadows.blob(0, { center: [w, by + 0.04, b], radius: BALL_R * 1.7, alpha: surf * pm });
		shadows.count = 1;
		ball.set(0, { x: w, y: by + BALL_R * 0.85, z: b, radius: BALL_R, color: tone('param', 1, surf * pm) });
		ball.count = 1;
		dropL.set(0, { a: [w, by, b], b: [w, FLOOR_Y, b], width: 1, color: tone('muted', 0.5, mu) });
		dropL.count = 1;
		footP.set(0, { x: w, y: FLOOR_Y, z: b, radius: 3, color: tone('muted', 1, 0.7 * mu) });
		footP.count = 1;

		// The minimum: a ring lying on the surface.
		const am = g.argmin;
		const my = surfaceY(am.w, am.b);
		ringP.set(0, { x: am.w, y: my + 0.06, z: am.b, radius: 10, ring: 2, color: tone('ok', 1, 0.95 * mo) });
		ringP.count = 1;

		// labels
		const [mx, mpy] = v.project(am.w, my + 0.06, am.b);
		labels.push({ id: 'lowest', x: mx, y: mpy + 30, text: 'lowest loss', tone: 'ok', opacity: mo });
		const edge = FLOOR_H + 0.35;
		const [wx, wy] = v.project(0, FLOOR_Y, ez >= 0 ? edge : -edge);
		const [bx, byy] = v.project(ex >= 0 ? edge : -edge, FLOOR_Y, 0);
		const [tx, ty] = v.project(cx, AXIS_TOP + 0.15, cz);
		labels.push(
			{ id: 'ax-w', x: wx, y: wy, text: 'w', tone: 'muted', opacity: mu },
			{ id: 'ax-b', x: bx, y: byy, text: 'b', tone: 'muted', opacity: mu },
			{ id: 'ax-loss', x: tx, y: ty, text: 'loss', tone: 'muted', opacity: mu }
		);

		floorL.upload();
		contourL.upload();
		trailL.upload();
		dropL.upload();
		ball.upload();
		ringP.upload();
		footP.upload();
		shadows.upload();
	}

	const scene: Scene = {
		init(c) {
			ctx = c;
			leftView = new ViewUniforms(c.gpu);
			const left = { gpu: c.gpu, target: c.target, view: leftView };
			cam2d = new Camera2D();
			rig = new OrbitRig();
			fitRig = new OrbitRig();
			orbit = new SceneOrbit(c, rig, { draggable: false });
			rig.setPose(POSES[1]);
			lines2 = new Lines(left, { capacity: 256 });
			dots2 = new Points(left, { capacity: 40 });

			mesh = new SurfaceMesh(c, { cols: N, rows: N, style: 'ink' });
			mesh.rect = [RANGE.w[0], RANGE.b[0], RANGE.w[1], RANGE.b[1]];
			mesh.range = [0, H_MAX];
			mesh.heights.set(grid(0).heights);
			mesh.upload();
			shadows = new Shadows(c, { capacity: 1 });
			floorL = new Lines(c, { capacity: 48, depthTest: true });
			contourL = new Lines(c, { capacity: 4000, depthTest: true });
			trailL = new Lines(c, { capacity: TRAIL_MAX, depthTest: true });
			dropL = new Lines(c, { capacity: 2 });
			ringP = new Points(c, { capacity: 2, depthTest: true });
			ball = new Points(c, { capacity: 2, shade: 'sphere', size: 'world', depthTest: true, depthWrite: true });
			footP = new Points(c, { capacity: 2 });
		},
		setParams(p) {
			w = p.w ?? w;
			b = p.b ?? b;
			resid = clamp(p.resid ?? resid, 0, 1);
			surf = clamp(p.surface ?? surf, 0, 1);
			poseParam = p.pose ?? poseParam;
			const m = (Math.round(p.model ?? model) === 1 ? 1 : 0) as ModelId;
			if (!seen || m !== model) setModel(m, !seen);
			seen = true;
			syncPose();
		},
		update(frame) {
			const labels: Label[] = [];
			layout();
			build2D(frame.dt, labels);
			lines2.upload();
			dots2.upload();
			if (surf > 0.003) build3D(frame.dt, labels);
			ctx.labels.set(labels);
			const L = loss(model, w, b);
			if (L !== lastLoss) {
				lastLoss = L;
				ctx.emit('loss', L);
			}
		},
		draw(pass) {
			lines2.draw(pass);
			dots2.draw(pass);
			if (surf > 0.003) {
				mesh.draw(pass);
				floorL.draw(pass);
				shadows.draw(pass);
				contourL.draw(pass);
				trailL.draw(pass);
				ringP.draw(pass);
				ball.draw(pass);
				dropL.draw(pass);
				footP.draw(pass);
			}
		},
		command(id) {
			orbit.command(id);
		},
		highlight(t) {
			hl = t;
		},
		pointer: onPointer,
		dispose() {
			lines2.destroy();
			dots2.destroy();
			mesh.destroy();
			shadows.dispose();
			floorL.destroy();
			contourL.destroy();
			trailL.destroy();
			dropL.destroy();
			ringP.destroy();
			ball.destroy();
			footP.destroy();
			leftView.destroy();
		}
	};
	return scene;
};
