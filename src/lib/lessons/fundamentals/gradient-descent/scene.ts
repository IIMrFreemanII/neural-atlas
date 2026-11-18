import { mat4 } from 'wgpu-matrix';
import { Camera2D, OrbitRig, type Pose } from '../../../gpu/camera.ts';
import { paperFill, sheetFill, tone } from '../../../gpu/colors.ts';
import { Lines } from '../../../gpu/primitives/lines.ts';
import { SurfaceMesh } from '../../../gpu/primitives/mesh.ts';
import { Points } from '../../../gpu/primitives/points.ts';
import { Shadows } from '../../../gpu/primitives/shadows.ts';
import { intersectPlane, rayFromCss, type Vec3 } from '../../../gpu/space.ts';
import type { Label, Scene, SceneContext, SceneFactory, ScenePointer, Tone } from '../../../gpu/types.ts';
import { ViewUniforms } from '../../../gpu/view.ts';
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
import { SceneOrbit } from '../../shared/orbit.ts';
import { fitCamera, insetAspect, insetMatrix, lessonInsets, type WorldRect } from '../../fit.ts';

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

// 3D map: parameter x -> world x, parameter y -> world z, loss -> height.
const MESH_COLS = 100;
const MESH_ROWS = 64;
const H_MAX = 1.5;
/** Heights follow the loss to this power: a gentle rise at the minimum instead of a cusp. */
const H_POWER = 0.75;
const FLOOR_Y = -0.05;
const FLOOR_X = 2.8;
const FLOOR_Z = 2.0;
const BALL_R = 0.1;
const LIFT = 0.03;
const SUB_LEN = 0.15;
const DIST = 8;

type Mode = '1d' | '2d';
type MapPose = 5 | 6 | 7;

/** One pose per map step (scenario 5, 6, 7). */
const POSES: Record<MapPose, Pose> = {
	5: { target: [0, 0.4, 0], azimuth: 0.55, elevation: 0.85, distance: DIST },
	6: { target: [0, 0.35, 0], azimuth: -1.05, elevation: 0.5, distance: DIST },
	7: { target: [0, 0.35, 0], azimuth: -1.05, elevation: 0.62, distance: DIST }
};

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

export const createGradientDescentScene: SceneFactory = () => {
	let ctx: SceneContext;
	let flatView: ViewUniforms;
	let camera: Camera2D;
	let rig: OrbitRig;
	let orbit: SceneOrbit;
	let fitRig: OrbitRig;
	let fit: Fit = { s: 1, cx: 0, cy: 0 };
	let fitSeen = false;
	// flat (1D) layers
	let lines1: Lines;
	let dots1: Points;
	// 3D layers
	let mesh: SurfaceMesh;
	let shadows: Shadows;
	let floorL: Lines;
	let trailL: Lines;
	let ringP: Points;
	let balls: Points;

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

	// surface
	let meshSurface = -1;
	let fMin = 0;
	let fSpan = 1;
	let poseKey = -1;
	let vp3: Float32Array = new Float32Array(16);

	// emitted readouts
	let lastLoss = NaN;
	let lastSteps = -1;

	const surface = () => SURFACES[surfaceId as 0 | 1 | 2] ?? SURFACES[0];
	const startPoint = (): Vec2 => customStart ?? surface().start;
	const mul = (t: Tone) => (hl === null || hl === t ? 1 : 0.3);

	/** Displayed height of the loss surface at parameter point (x, y); clamped to the domain. */
	function hOf(x: number, y: number): number {
		const f = surface().f(clamp(x, DOMAIN.x[0], DOMAIN.x[1]), clamp(y, DOMAIN.y[0], DOMAIN.y[1]));
		return H_MAX * Math.pow(Math.max(0, (f - fMin) / fSpan), H_POWER);
	}

	function buildMesh() {
		meshSurface = surfaceId;
		const s = surface();
		let lo = Infinity;
		let hi = -Infinity;
		const vals = mesh.heights;
		for (let r = 0; r < MESH_ROWS; r++) {
			const y = DOMAIN.y[0] + (r / (MESH_ROWS - 1)) * (DOMAIN.y[1] - DOMAIN.y[0]);
			for (let c = 0; c < MESH_COLS; c++) {
				const x = DOMAIN.x[0] + (c / (MESH_COLS - 1)) * (DOMAIN.x[1] - DOMAIN.x[0]);
				const f = s.f(x, y);
				vals[r * MESH_COLS + c] = f;
				if (f < lo) lo = f;
				if (f > hi) hi = f;
			}
		}
		fMin = lo;
		fSpan = Math.max(1e-9, hi - lo);
		for (let i = 0; i < vals.length; i++) vals[i] = H_MAX * Math.pow(Math.max(0, (vals[i] - fMin) / fSpan), H_POWER);
	}

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

	/** Glide to the map step's pose; the figure only turns on the map steps. */
	function syncPose() {
		const key = mode === '2d' ? (scenario === 6 || scenario === 7 ? scenario : 5) : -1;
		if (key === poseKey) return;
		poseKey = key;
		orbit.setDraggable(key >= 0);
		if (key >= 0) orbit.stepPose(POSES[key as MapPose]);
	}

	/** The parameter point under a pointer: the first place the ray meets the surface, else the floor plane. */
	function pick(cssX: number, cssY: number): Vec2 | null {
		const v = ctx.view;
		const ray = rayFromCss(vp3, v.width, v.height, cssX, cssY);
		const inside = (x: number, z: number) => x >= DOMAIN.x[0] && x <= DOMAIN.x[1] && z >= DOMAIN.y[0] && z <= DOMAIN.y[1];
		const at = (t: number): Vec3 => [
			ray.origin[0] + ray.dir[0] * t,
			ray.origin[1] + ray.dir[1] * t,
			ray.origin[2] + ray.dir[2] * t
		];
		let prev = 0;
		for (let t = 0; t < 40; t += 0.04) {
			const p = at(t);
			if (inside(p[0], p[2]) && p[1] <= hOf(p[0], p[2])) {
				let lo = prev;
				let hi = t;
				for (let i = 0; i < 14; i++) {
					const mid = (lo + hi) / 2;
					const q = at(mid);
					if (inside(q[0], q[2]) && q[1] <= hOf(q[0], q[2])) hi = mid;
					else lo = mid;
				}
				const q = at(hi);
				return [clamp(q[0], DOMAIN.x[0], DOMAIN.x[1]), clamp(q[2], DOMAIN.y[0], DOMAIN.y[1])];
			}
			prev = t;
		}
		const hit = intersectPlane(ray, [0, 0, 0], [0, 1, 0]);
		return hit && inside(hit[0], hit[2]) ? [hit[0], hit[2]] : null;
	}

	function pointer(e: ScenePointer) {
		if (mode !== '2d') return;
		if (orbit.pointer(e) !== 'click') return;
		const hit = pick(e.x, e.y);
		if (!hit) return;
		customStart = hit;
		reset();
	}

	function build1D(o: number, labels: Label[]) {
		const v = flatView;
		const wc = clamp(w, -W_MAX, W_MAX);
		const g = curve.grad(wc);
		let n = 0;
		let d = 0;

		// loss curve: one ink stroke
		const pts: Array<[number, number]> = [];
		for (let i = 0; i <= 96; i++) {
			const x = -W_MAX + (2 * W_MAX * i) / 96;
			pts.push([wx(x), ly(curve.f(x))]);
		}
		const pm = mul('param') * o;
		n = lines1.setPolyline(n, pts, { width: 2.2, color: tone('param', 1, 0.95 * pm) });

		// baseline and minimum marker
		const mm = mul('muted') * o;
		lines1.set(n++, {
			a: [-W_MAX - 0.2, Y_BASE],
			b: [W_MAX + 0.2, Y_BASE],
			width: 1,
			color: tone('muted', 1, 0.7 * mm)
		});

		// step arrow
		const dm = mul('data') * o;
		const target = clamp(w - lr * curve.grad(w), -W_MAX, W_MAX);
		if (Math.abs(target - wc) > 0.03) {
			const dir = Math.sign(target - wc);
			const col = tone('data', 1, 0.95 * dm);
			lines1.set(n++, { a: [wc, Y_BASE], b: [target, Y_BASE], width: 2.5, color: col });
			lines1.set(n++, { a: [target, Y_BASE], b: [target - dir * 0.13, Y_BASE + 0.08], width: 2.5, color: col });
			lines1.set(n++, { a: [target, Y_BASE], b: [target - dir * 0.13, Y_BASE - 0.08], width: 2.5, color: col });
		}

		// drop line and tangent
		const gm = mul('grad') * o;
		const by = ly(curve.f(wc));
		lines1.set(n++, { a: [wc, by], b: [wc, Y_BASE], width: 1, color: tone('grad', 1, 0.45 * gm) });
		const tx = 1;
		const ty = g * Y_PER_LOSS;
		const tl = Math.hypot(tx, ty);
		const hx = ((tx / tl) * 1.2) / 2;
		const hy = ((ty / tl) * 1.2) / 2;
		lines1.set(n++, { a: [wc - hx, by - hy], b: [wc + hx, by + hy], width: 2, color: tone('grad', 1, gm) });
		lines1.count = n;

		// trail dots, the minimum ring, then the ball: a solid dot with a paper ring
		for (let i = 0; i < trail1.length; i++) {
			const tw = clamp(trail1[i], -W_MAX, W_MAX);
			const a = 0.25 + (0.6 * (i + 1)) / trail1.length;
			dots1.set(d++, { x: wx(tw), y: ly(curve.f(tw)), radius: 2.5, color: tone('grad', 1, a * gm) });
		}
		dots1.set(d++, { x: 0, y: Y_BASE, radius: 5, ring: 1.5, color: tone('ok', 1, 0.9 * mul('ok') * o) });
		dots1.set(d++, { x: wc, y: by, radius: 11, color: paperFill(pm) });
		dots1.set(d++, { x: wc, y: by, radius: 7.5, color: tone('param', 1, pm) });
		dots1.count = d;

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

	/** A trail segment draped over the surface: split long hops so the line follows the ground. */
	function drape(n: number, a: Vec2, b: Vec2, width: number, color: [number, number, number, number]): number {
		const ax = clamp(a[0], DOMAIN.x[0], DOMAIN.x[1]);
		const ay = clamp(a[1], DOMAIN.y[0], DOMAIN.y[1]);
		const bx = clamp(b[0], DOMAIN.x[0], DOMAIN.x[1]);
		const by = clamp(b[1], DOMAIN.y[0], DOMAIN.y[1]);
		const k = clamp(Math.ceil(Math.hypot(bx - ax, by - ay) / SUB_LEN), 1, 8);
		let px = ax;
		let py = ay;
		for (let i = 1; i <= k && n < trailL.capacity; i++) {
			const t = i / k;
			const qx = ax + (bx - ax) * t;
			const qy = ay + (by - ay) * t;
			trailL.set(n++, { a: [px, hOf(px, py) + LIFT, py], b: [qx, hOf(qx, qy) + LIFT, qy], width, color });
			px = qx;
			py = qy;
		}
		return n;
	}

	function build3D(o: number, dt: number, labels: Label[]) {
		const v = ctx.view;
		orbit.update(dt);
		const ins = lessonInsets(v);
		const a3 = insetAspect(v, ins);
		// Fit from the step's home pose (not the live one), eased during a glide, so dragging only rotates.
		fitRig.setPose(rig.home);
		const goal = fitBox(fitRig.matrix(a3), [DOMAIN.x[0], DOMAIN.x[1], FLOOR_Y, H_MAX, DOMAIN.y[0], DOMAIN.y[1]], 0.08);
		if (!fitSeen || ctx.reducedMotion) fit = goal;
		else {
			const k = Math.min(1, dt * 6);
			fit = { s: fit.s + (goal.s - fit.s) * k, cx: fit.cx + (goal.cx - fit.cx) * k, cy: fit.cy + (goal.cy - fit.cy) * k };
		}
		fitSeen = true;
		vp3 = mat4.multiply(insetMatrix(v, ins), mat4.multiply(fitMatrix(fit), rig.matrix(a3))) as Float32Array;
		v.setViewProj(vp3);

		const sf = surface();
		const pm = mul('param') * o;
		mesh.opacity = pm;
		// The theme can change under a live scene, so the inks are set every frame.
		mesh.setInk({
			fill: sheetFill(),
			ink: tone('muted', 0.55),
			gridEvery: 4,
			hatch: true,
			contours: true,
			contourInk: tone('param', 0.7)
		});
		mesh.upload();

		// floor: a faint ruled grid, its frame and four posts under the surface corners
		const mu = mul('muted') * o;
		let n = 0;
		const rule = tone('text', 0.12, mu);
		for (let k = -5; k <= 5; k++) {
			const t = k * 0.5;
			floorL.set(n++, { a: [t, FLOOR_Y, -FLOOR_Z], b: [t, FLOOR_Y, FLOOR_Z], width: 1, color: rule });
		}
		for (let k = -4; k <= 4; k++) {
			const t = k * 0.5;
			floorL.set(n++, { a: [-FLOOR_X, FLOOR_Y, t], b: [FLOOR_X, FLOOR_Y, t], width: 1, color: rule });
		}
		const fc: Vec3[] = [
			[-FLOOR_X, FLOOR_Y, -FLOOR_Z],
			[FLOOR_X, FLOOR_Y, -FLOOR_Z],
			[FLOOR_X, FLOOR_Y, FLOOR_Z],
			[-FLOOR_X, FLOOR_Y, FLOOR_Z]
		];
		for (let i = 0; i < 4; i++) floorL.set(n++, { a: fc[i], b: fc[(i + 1) % 4], width: 1, color: tone('text', 0.3, mu) });
		for (const cx of DOMAIN.x) {
			for (const cz of DOMAIN.y) {
				floorL.set(n++, { a: [cx, FLOOR_Y, cz], b: [cx, hOf(cx, cz), cz], width: 1, color: tone('muted', 0.35, mu) });
			}
		}
		floorL.count = n;

		// start and minima rings, lying on the surface
		const om = mul('ok') * o;
		let rn = 0;
		for (const [x, y] of sf.minima) {
			ringP.set(rn++, { x, y: hOf(x, y) + 0.06, z: y, radius: 9, ring: 1.6, color: tone('ok', 1, 0.95 * om) });
		}
		const st = startPoint();
		ringP.set(rn++, {
			x: st[0],
			y: hOf(st[0], st[1]) + 0.06,
			z: st[1],
			radius: 7,
			ring: 1.4,
			color: tone('muted', 1, 0.9 * mul('muted') * o)
		});
		ringP.count = rn;

		// trails (slower optimisers first so plain descent stays on top), then the racers
		const active = race ? 3 : 1;
		let tn = 0;
		let bn = 0;
		let sh = 0;
		for (let k = active - 1; k >= 0; k--) {
			const r = racers[k];
			const m = mul(r.tone) * o;
			const tr = r.trail;
			const len = tr.length;
			for (let i = 0; i + 1 < len; i++) {
				const a = 0.4 + (0.55 * (i + 1)) / len;
				tn = drape(tn, tr[i], tr[i + 1], 2, tone(r.tone, 1, a * m));
			}
			const x = clamp(r.st.p[0], DOMAIN.x[0], DOMAIN.x[1]);
			const z = clamp(r.st.p[1], DOMAIN.y[0], DOMAIN.y[1]);
			const h = hOf(x, z);
			shadows.blob(sh++, { center: [x, h + 0.04, z], radius: BALL_R * 1.7, alpha: m });
			balls.set(bn++, { x, y: h + BALL_R * 0.85, z, radius: BALL_R, color: tone(r.tone, 1, m) });
		}
		trailL.count = tn;
		balls.count = bn;
		shadows.count = sh;

		// label above the global minimum
		const [gx, gy] = sf.minima[0];
		const [mpx, mpy] = v.project(gx, hOf(gx, gy) + 0.06, gy);
		labels.push({ id: 'minimum', x: mpx, y: mpy + 26, text: 'minimum', tone: 'ok', opacity: om });

		floorL.upload();
		trailL.upload();
		ringP.upload();
		balls.upload();
		shadows.upload();
	}

	const scene: Scene = {
		init(c) {
			ctx = c;
			flatView = new ViewUniforms(c.gpu);
			const flat = { gpu: c.gpu, target: c.target, view: flatView };
			camera = new Camera2D();
			rig = new OrbitRig();
			fitRig = new OrbitRig();
			orbit = new SceneOrbit(c, rig, { draggable: false });
			rig.setPose(POSES[5]);

			lines1 = new Lines(flat, { capacity: 160 });
			dots1 = new Points(flat, { capacity: 64 });

			mesh = new SurfaceMesh(c, { cols: MESH_COLS, rows: MESH_ROWS, style: 'ink' });
			mesh.rect = [DOMAIN.x[0], DOMAIN.y[0], DOMAIN.x[1], DOMAIN.y[1]];
			mesh.range = [0, H_MAX];
			mesh.bands = 12;
			buildMesh();
			mesh.upload();
			shadows = new Shadows(c, { capacity: 3 });
			floorL = new Lines(c, { capacity: 64, depthTest: true });
			trailL = new Lines(c, { capacity: 12000, depthTest: true });
			ringP = new Points(c, { capacity: 4, depthTest: true });
			balls = new Points(c, { capacity: 3, shade: 'sphere', size: 'world', depthTest: true, depthWrite: true });
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
			if (surfaceId !== meshSurface) buildMesh();
			syncPose();
		},
		update(frame) {
			const v = ctx.view;
			fitCamera(camera, v, WORLD, lessonInsets(v));
			flatView.setSize(v.width, v.height, v.dpr);
			flatView.setViewProj(camera.matrix(v.aspect));
			flatView.flush();
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
			lines1.count = 0;
			dots1.count = 0;
			const labels: Label[] = [];
			if (o1 > 0.002) build1D(o1, labels);
			lines1.upload();
			dots1.upload();
			if (o2 > 0.002) build3D(o2, frame.dt, labels);
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
			lines1.draw(pass);
			dots1.draw(pass);
			if (map > 0.002) {
				mesh.draw(pass);
				floorL.draw(pass);
				shadows.draw(pass);
				trailL.draw(pass);
				ringP.draw(pass);
				balls.draw(pass);
			}
		},
		command(id) {
			if (orbit.command(id)) return;
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
		},
		pointer,
		dispose() {
			lines1.destroy();
			dots1.destroy();
			mesh.destroy();
			shadows.dispose();
			floorL.destroy();
			trailL.destroy();
			ringP.destroy();
			balls.destroy();
			flatView.destroy();
		}
	};
	return scene;
};
