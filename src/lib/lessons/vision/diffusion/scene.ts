import { mat4 } from 'wgpu-matrix';
import { OrbitRig, type Pose } from '../../../gpu/camera.ts';
import { mixColor, sheetFill, tone } from '../../../gpu/colors.ts';
import { ParticleCloud } from '../../../gpu/compute/particles.ts';
import { Lines } from '../../../gpu/primitives/lines.ts';
import { Points } from '../../../gpu/primitives/points.ts';
import { Shadows } from '../../../gpu/primitives/shadows.ts';
import { Sheets, type SheetSpec, type SheetTexture } from '../../../gpu/primitives/sheets.ts';
import { projectToFloor, type Vec3 } from '../../../gpu/space.ts';
import { ViewUniforms } from '../../../gpu/view.ts';
import type { Label, Scene, SceneContext, SceneFactory, Tone } from '../../../gpu/types.ts';
import { denoised, noise, reverseStep, sampleData, schedule } from '../../../ml/diffusion2d.ts';
import { gaussian, mulberry32 } from '../../../ml/rng.ts';
import { insetAspect, insetMatrix, lessonInsets } from '../../fit.ts';
import { SceneOrbit } from '../../shared/orbit.ts';
import { N, PATTERNS } from './data.ts';

// Data space: the pattern fills about +-2.6 by +-2.1; noise reaches 3 standard deviations. A sheet shows this square.
const HALF = 3.2;
const COLS = 13;
const ROWS = 13;
const GRID_HALF = 2.7;
const MAX_ARROW = 0.4;
const HEAD = 0.13;
const NOISE_RATE = 0.25; // t per second while "Add noise" plays
const GEN_SECONDS = 3.2;
const MIN_INTERVAL = 0.05;
const INSTANT_DT = 100; // seconds fed to the particle easing so it lands on its target in one frame
const REDUCED_TICK = 1 / 3; // reduced motion: one visible denoising update every third of a second
const REDUCED_TICKS = 10; // and about this many updates for a whole generation

// The timeline of sheets: one upright sheet per noise level, laid along x, noise on the left, picture on the right.
const LEVELS = 5;
const SHEET = 2.7; // sheet side, world units
const PITCH = 3.1; // centre to centre
const SHEET_Y0 = 0.05; // sheet bottom edge
const K = SHEET / (2 * HALF); // data units -> world units on a sheet
const TEX = 48;
const TEX_GAIN = 0.12;
const TEX_MAX = 0.92;
const CUR_TEX_DIM = 0.35; // the sheet under the live cloud shows its picture fainter, so the dots read
const AXIS_Z = 1.7; // the timeline lies on the floor in front of the sheets
const FLOOR_X = 8.8;
const FLOOR_Z0 = -2.4;
const FLOOR_Z1 = 3.2;
const LIVE_Z = 0.03; // the cloud and arrows float this far in front of their sheet
const DRAW_IN = 0.9; // seconds for a new line to draw itself
const MAX_DRAG = 1.0; // keep the camera in front of the sheets

/** x of the sheet (and of the live cloud) at noise level t: t = 1 on the left, t = 0 on the right. */
const xAt = (t: number) => 2 * PITCH * (1 - 2 * t);
const levelT = (i: number) => 1 - i / (LEVELS - 1);
const sheetX = (i: number) => xAt(levelT(i));

// Particle look.
const DOT_RADIUS = 1.1;
const DOT_SOFTNESS = 0.3;
const DOT_DATA = { intensity: 1, alpha: 0.55 };
const DOT_NOISE = { intensity: 1, alpha: 0.6 };
const NOISE_MIX = 0.8; // how far the dot color moves from data to noise at t = 1

const clamp = (v: number, a: number, b: number) => Math.min(b, Math.max(a, v));
/** How far the ink has moved from data to noise colour at level t (cyan and amber mix to gray, so the hue changes late). */
function noiseMix(t: number): number {
	const u = clamp((t - 0.62) / 0.33, 0, 1);
	return u * u * (3 - 2 * u) * NOISE_MIX;
}
const round2 = (v: number) => Math.round(v * 100) / 100;

export const createDiffusionScene: SceneFactory = () => {
	let ctx: SceneContext;
	let rig: OrbitRig;
	let orbit: SceneOrbit;
	let liveView: ViewUniforms; // the cloud and the arrows are drawn through this: camera x (model matrix onto the live sheet)
	let cloud: ParticleCloud;
	let arrows: Lines;
	let floor: Sheets;
	let grid: Lines;
	let axis: Lines;
	let ticks: Lines;
	let shadows: Shadows;
	let sheets: Sheets;
	let textures: SheetTexture[] = [];
	let marker: Points;
	let vp: Float32Array = new Float32Array(16);

	// camera / draw-in state
	let poseKey = '';
	let poseAspect = 1;
	let axisReveal = 0;
	let fieldReveal = 1;
	let fieldShown = false;
	let sheetsDirty = true;
	let curSheet = -1;
	let closeMode = false;

	// params
	let pattern = 0;
	let steps = 40;
	let fieldOn = 0;
	let scenario = 1;
	let paramsSeen = false;
	let hl: Tone | null = null;

	// per particle state
	const x0 = new Float64Array(N * 2);
	const eps = new Float64Array(N * 2);
	const pos = new Float64Array(N * 2);
	const start = new Float32Array(N * 2);
	let reseeds = 0;

	// time and modes
	let tCur = 0;
	let viewT = 0; // the noise level the camera is framed for; eases toward tCur
	let viewSeen = false;
	let known = 0; // the value of param t as the UI last saw it from us
	let dirty = true;
	let teleport = false;
	let animating = false;
	let animT = 0;
	let mode: 'scrub' | 'gen' = 'scrub';
	let genSteps = 40;
	let genIndex = 0;
	let genTimer = 0;

	// direction field cache
	let fieldKey = '';
	let fieldCount = 0;

	// emitted readouts
	let lastLevel = '';
	let lastStep = '';

	const epsSeed = () => 7000 + pattern * 31 + scenario * 977 + reseeds;

	function fillEps() {
		const rand = mulberry32(epsSeed());
		for (let i = 0; i < eps.length; i++) eps[i] = gaussian(rand);
	}

	function resample() {
		const m = PATTERNS[pattern].mixture;
		x0.set(sampleData(m, N, mulberry32(100 + pattern * 7 + scenario * 131)));
		reseeds = 0;
		fillEps();
		mode = 'scrub';
		animating = false;
		dirty = true;
		teleport = true;
		sheetsDirty = true;
	}

	function emitT(v: number) {
		const r = round2(v);
		if (r === known) return;
		known = r;
		ctx.emit('param:t', r);
	}

	/** Leave generate mode so the learner can scrub: the particles become the new "clean" data. */
	function exitGen() {
		if (mode !== 'gen') return;
		mode = 'scrub';
		if (tCur <= 1e-6) {
			x0.set(pos);
		} else {
			const m = PATTERNS[pattern].mixture;
			const { alpha, sigma } = schedule(tCur);
			for (let i = 0; i < N; i++) {
				const [dx, dy] = denoised(m, pos[i * 2], pos[i * 2 + 1], tCur);
				x0[i * 2] = dx;
				x0[i * 2 + 1] = dy;
				eps[i * 2] = (pos[i * 2] - alpha * dx) / sigma;
				eps[i * 2 + 1] = (pos[i * 2 + 1] - alpha * dy) / sigma;
			}
		}
		dirty = true;
		sheetsDirty = true;
	}

	function genStep() {
		const from = 1 - genIndex / genSteps;
		const to = 1 - (genIndex + 1) / genSteps;
		reverseStep(PATTERNS[pattern].mixture, pos, from, to);
		genIndex++;
		tCur = genIndex >= genSteps ? 0 : to;
		emitT(tCur);
		dirty = true;
	}

	function startGenerate() {
		animating = false;
		mode = 'gen';
		genSteps = clamp(Math.round(steps), 1, 100);
		genIndex = 0;
		genTimer = Math.max(MIN_INTERVAL, GEN_SECONDS / genSteps); // the first step fires on the next frame
		const wasNoisy = tCur >= 0.99;
		pos.set(eps);
		tCur = 1;
		emitT(1);
		dirty = true;
		// Already a blob: the dots are in place. Otherwise they snap to the fresh noise.
		teleport = !wasNoisy;
		if (ctx.reducedMotion) genTimer = REDUCED_TICK; // the first update lands on the next frame
	}

	function fieldMul() {
		return hl === null || hl === 'ok' ? 1 : 0.3;
	}

	function dotMul() {
		if (hl === null) return 1;
		if (hl === 'data') return tCur < 0.5 ? 1 : 0.3;
		if (hl === 'grad') return tCur >= 0.5 ? 1 : 0.3;
		return 0.3;
	}

	/** Denoising arrows in data coordinates; liveView carries them onto the live sheet. */
	function drawField() {
		const t = round2(tCur);
		const visible = fieldOn === 1 && t >= 0.02;
		const key = visible ? `${pattern}|${t}|${hl === null || hl === 'ok' ? 1 : 0}` : 'hidden';
		if (visible && !fieldShown) fieldReveal = ctx.reducedMotion ? 1 : 0;
		fieldShown = visible;
		if (key === fieldKey) return;
		fieldKey = key;
		let n = 0;
		if (visible) {
			const m = PATTERNS[pattern].mixture;
			const mul = fieldMul();
			const sx = (2 * GRID_HALF) / (COLS - 1);
			const sy = (2 * GRID_HALF) / (ROWS - 1);
			for (let r = 0; r < ROWS; r++) {
				for (let c = 0; c < COLS; c++) {
					const px = -GRID_HALF + c * sx;
					const py = -GRID_HALF + r * sy;
					const [gx, gy] = denoised(m, px, py, t);
					const dx = gx - px;
					const dy = gy - py;
					const dist = Math.hypot(dx, dy);
					if (dist < 1e-6) continue;
					const len = Math.min(MAX_ARROW, 0.5 * dist);
					const ux = dx / dist;
					const uy = dy / dist;
					const bx = px + ux * len;
					const by = py + uy * len;
					const color = tone('ok', 0.4 + 0.6 * Math.min(1, dist / 1.2), mul);
					arrows.set(n++, { a: [px, py], b: [bx, by], width: 1.4, color });
					for (const sgn of [1, -1]) {
						const ang = sgn * 0.5;
						const hx = -(ux * Math.cos(ang) - uy * Math.sin(ang)) * HEAD;
						const hy = -(uy * Math.cos(ang) + ux * Math.sin(ang)) * HEAD;
						arrows.set(n++, { a: [bx, by], b: [bx + hx, by + hy], width: 1.4, color });
					}
				}
			}
		}
		arrows.count = n;
		fieldCount = n;
		arrows.upload();
	}

	/** A CPU density picture of the current pattern noised to level t: ink where the dots are, paper elsewhere. */
	function writeSheet(i: number, tex: SheetTexture) {
		const t = levelT(i);
		const { alpha, sigma } = schedule(t);
		const counts = new Float32Array(TEX * TEX);
		const inv = TEX / (2 * HALF);
		for (let p = 0; p < N; p++) {
			const x = alpha * x0[p * 2] + sigma * eps[p * 2];
			const y = alpha * x0[p * 2 + 1] + sigma * eps[p * 2 + 1];
			const cx = Math.floor((x + HALF) * inv);
			const cy = Math.floor((y + HALF) * inv);
			if (cx >= 0 && cx < TEX && cy >= 0 && cy < TEX) counts[cy * TEX + cx]++;
		}
		const dim = i === curSheet ? CUR_TEX_DIM : closeMode ? 0.25 : 1;
		const k = noiseMix(t);
		tex.write((x, y) => {
			const d = TEX_MAX * (1 - Math.exp(-TEX_GAIN * counts[y * TEX + x])) * dim;
			if (d < 0.015) return [0, 0, 0, 0];
			return mixColor(tone('data', d), tone('grad', d), k);
		});
	}

	function setCloudStyle() {
		// Cyan and amber mix to gray, so the hue changes late and quickly instead of linearly with t.
		const k = noiseMix(tCur);
		cloud.neutralColor = mixColor(
			tone('data', DOT_DATA.intensity, DOT_DATA.alpha),
			tone('grad', DOT_NOISE.intensity, DOT_NOISE.alpha),
			k
		);
		cloud.neutralColor[3] *= dotMul();
		cloud.radius = DOT_RADIUS;
		cloud.softness = DOT_SOFTNESS;
		cloud.groupCount = 0;
		cloud.jitter = 0;
		cloud.follow = mode === 'gen' ? Math.max(3, 1.2 / Math.max(MIN_INTERVAL, GEN_SECONDS / genSteps)) : 14;
	}

	function pushTargets() {
		const t = cloud.targets;
		for (let i = 0; i < N; i++) {
			t[i * 2] = pos[i * 2];
			t[i * 2 + 1] = pos[i * 2 + 1];
		}
		cloud.uploadTargets();
		if (teleport) {
			start.set(t);
			cloud.setPositions(start);
			teleport = false;
		}
	}

	function emitReadouts() {
		const level = `${Math.round(tCur * 100)}%`;
		if (level !== lastLevel) {
			lastLevel = level;
			ctx.emit('level', level);
		}
		const step = mode === 'gen' ? `${genIndex} / ${genSteps}` : '—';
		if (step !== lastStep) {
			lastStep = step;
			ctx.emit('step', step);
		}
	}

	/** Keep every sheet corner inside the uncovered area, including the nearer, larger end of the perspective view. */
	function overviewPose(aspect: number): Pose {
		const half = sheetX(LEVELS - 1) + SHEET / 2;
		const mid = SHEET_Y0 + SHEET / 2;
		const pose: Pose = { target: [0, mid, 0], azimuth: 0.38, elevation: 0.26, distance: (2 * half) / (0.828 * aspect * 0.88) };
		const probe = new OrbitRig();
		const corners: Vec3[] = [];
		for (const x of [-half, half]) {
			for (const y of [SHEET_Y0, SHEET_Y0 + SHEET]) {
				for (const z of [0, LIVE_Z]) corners.push([x, y, z]);
			}
		}
		for (let it = 0; it < 8; it++) {
			probe.setPose(pose);
			const m = probe.matrix(aspect);
			const extent = Math.max(...corners.map((p) => {
				const w = m[3] * p[0] + m[7] * p[1] + m[11] * p[2] + m[15];
				const x = (m[0] * p[0] + m[4] * p[1] + m[8] * p[2] + m[12]) / w;
				const y = (m[1] * p[0] + m[5] * p[1] + m[9] * p[2] + m[13]) / w;
				return Math.max(Math.abs(x), Math.abs(y));
			}));
			if (extent <= 0.88) break;
			pose.distance *= extent / 0.88;
		}
		return pose;
	}

	/** The current sheet fills about 70% of the uncovered height, centred. */
	function closePose(i: number, _aspect: number): Pose {
		return { target: [sheetX(i), SHEET_Y0 + SHEET / 2, 0], azimuth: 0.3, elevation: 0.2, distance: SHEET / (0.828 * 0.7) };
	}

	/** The sheets with their shadows, the strong edge on the one under the cloud, and the marker on the timeline. */
	function placeSheets(near: number, t: number) {
		for (let i = 0; i < LEVELS; i++) {
			const x = sheetX(i) - SHEET / 2;
			const spec: SheetSpec = {
				origin: [x, SHEET_Y0, 0],
				u: [SHEET, 0, 0],
				v: [0, SHEET, 0],
				fill: sheetFill(),
				edge: i === near ? tone('grad', 1) : tone('text', closeMode ? 0.15 : 0.3),
				edgePx: i === near ? 2.2 : 1,
				texture: textures[i]
			};
			sheets.set(i, spec);
			const o = spec.origin;
			const corners = [o, [o[0] + SHEET, o[1], o[2]], [o[0] + SHEET, o[1] + SHEET, o[2]], [o[0], o[1] + SHEET, o[2]]].map((p) =>
				projectToFloor(p as Vec3, 0)
			);
			shadows.quad(i, { corners: corners as [Vec3, Vec3, Vec3, Vec3], blur: 0.12, alpha: 0.9 });
		}
		const mx = xAt(t);
		const mr = 0.012 * rig.current.distance; // constant size on screen
		shadows.blob(LEVELS, { center: [mx, 0, AXIS_Z], radius: mr * 1.6 });
		shadows.count = LEVELS + 1;
		marker.set(0, { x: mx, y: mr, z: AXIS_Z, radius: mr, color: tone('grad', 1) });
		marker.upload();
	}

	/** t labels under the ticks, and a word over each end of the row. Labels off the stage are left out. */
	function labelsFor(near: number) {
		const v = ctx.view;
		const out: Label[] = [];
		const add = (id: string, p: [number, number], text: string, o: Partial<Label>) => {
			if (p[0] < 20 || p[0] > v.width - 20 || p[1] < 20 || p[1] > v.height - 20) return;
			out.push({ id, x: p[0], y: p[1], text, ...o });
		};
		for (let i = 0; i < LEVELS; i++) {
			const [px, py] = v.project(sheetX(i), 0, AXIS_Z);
			add(`t${i}`, [px, py + 16], `t = ${levelT(i)}`, { tone: i === near ? 'grad' : 'muted', mono: true, size: 12 });
		}
		const top = (i: number): [number, number] => v.project(sheetX(i), SHEET_Y0 + SHEET + 0.3, 0);
		add('noise', top(0), 'pure noise', { tone: 'muted', size: 13 });
		add('picture', top(LEVELS - 1), 'the pattern', { tone: 'muted', size: 13 });
		ctx.labels.set(out);
	}

	const scene: Scene = {
		init(c) {
			ctx = c;
			rig = new OrbitRig({ maxDrag: MAX_DRAG });
			orbit = new SceneOrbit(c, rig);
			liveView = new ViewUniforms(c.gpu);
			const live = { gpu: c.gpu, target: c.target, view: liveView };
			cloud = new ParticleCloud(live, { capacity: N, blend: 'alpha' });
			arrows = new Lines(live, { capacity: COLS * ROWS * 3, blend: 'alpha' });

			floor = new Sheets(c, { capacity: 1 });
			floor.set(0, {
				origin: [-FLOOR_X, 0, FLOOR_Z1],
				u: [2 * FLOOR_X, 0, 0],
				v: [0, 0, FLOOR_Z0 - FLOOR_Z1],
				fill: sheetFill(),
				edge: tone('text', 0.2),
				edgePx: 1
			});
			floor.count = 1;
			grid = new Lines(c, { capacity: 64, blend: 'alpha' });
			let g = 0;
			const faint = tone('text', 0.12);
			for (let x = -Math.floor(FLOOR_X); x <= Math.floor(FLOOR_X); x++) {
				grid.set(g++, { a: [x, 0, FLOOR_Z0], b: [x, 0, FLOOR_Z1], width: 1, color: faint });
			}
			for (let z = Math.ceil(FLOOR_Z0); z <= Math.floor(FLOOR_Z1); z++) {
				grid.set(g++, { a: [-FLOOR_X, 0, z], b: [FLOOR_X, 0, z], width: 1, color: faint });
			}
			grid.count = g;
			grid.upload();

			// The timeline: an ink axis with an arrowhead pointing toward t = 0, and a tick under every sheet.
			axis = new Lines(c, { capacity: 4, blend: 'alpha' });
			const ink = tone('text', 0.7);
			const x1 = sheetX(LEVELS - 1) + SHEET / 2 + 0.9;
			axis.count = axis.setPolyline(0, [[sheetX(0) - SHEET / 2 - 0.9, 0, AXIS_Z], [x1, 0, AXIS_Z]], { width: 1.8, color: ink });
			axis.upload();
			ticks = new Lines(c, { capacity: LEVELS + 2, blend: 'alpha' });
			let tk = 0;
			ticks.set(tk++, { a: [x1 - 0.28, 0, AXIS_Z - 0.16], b: [x1, 0, AXIS_Z], width: 1.8, color: ink });
			ticks.set(tk++, { a: [x1 - 0.28, 0, AXIS_Z + 0.16], b: [x1, 0, AXIS_Z], width: 1.8, color: ink });
			for (let i = 0; i < LEVELS; i++) {
				ticks.set(tk++, { a: [sheetX(i), 0, AXIS_Z - 0.14], b: [sheetX(i), 0, AXIS_Z + 0.14], width: 1.8, color: ink });
			}
			ticks.count = tk;
			ticks.upload();

			shadows = new Shadows(c, { capacity: LEVELS + 1 });
			sheets = new Sheets(c, { capacity: LEVELS });
			textures = [];
			for (let i = 0; i < LEVELS; i++) textures.push(sheets.createTexture(TEX, TEX));
			sheets.count = LEVELS;
			marker = new Points(c, { capacity: 1, shade: 'sphere', size: 'world', depthTest: true, depthWrite: true });
			marker.count = 1;
			axisReveal = c.reducedMotion ? 1 : 0;
			axis.setReveal(axisReveal);
			ticks.setReveal(0);
			setCloudStyle();
		},
		setParams(p) {
			const prevPattern = pattern;
			const prevScenario = scenario;
			pattern = clamp(Math.round(p.pattern ?? pattern), 0, PATTERNS.length - 1);
			scenario = Math.round(p.scenario ?? scenario);
			steps = clamp(Math.round(p.steps ?? steps), 1, 100);
			fieldOn = Math.round(p.field ?? fieldOn);
			const t = clamp(p.t ?? tCur, 0, 1);
			const sceneChanged = !paramsSeen || pattern !== prevPattern || scenario !== prevScenario;
			if (Math.abs(t - known) > 1e-6) {
				// The learner (or a step change) moved the slider.
				animating = false;
				if (!sceneChanged) exitGen();
				tCur = t;
				known = t;
				dirty = true;
			}
			if (sceneChanged) {
				if (!paramsSeen) {
					tCur = t;
					known = t;
				}
				resample();
			}
			paramsSeen = true;
		},
		update(fr) {
			const v = ctx.view;
			// The live cloud's place along the timeline eases toward the noise level, so a few-step run glides instead of jumping.
			viewT = ctx.reducedMotion || !viewSeen ? tCur : viewT + (tCur - viewT) * (1 - Math.exp(-fr.dt / 0.15));
			viewSeen = true;
			const ins = lessonInsets(v);
			const aspect = insetAspect(v, ins);
			const near = clamp(Math.round((1 - viewT) * (LEVELS - 1)), 0, LEVELS - 1);

			// Camera: an overview of the whole timeline, or a closer look at the sheet under the cloud.
			const close = scenario === 1 || scenario === 4;
			const key = close ? `c${near}` : 'o';
			if (close !== closeMode) {
				closeMode = close;
				sheetsDirty = true;
			}
			if (key !== poseKey || (!rig.moved && Math.abs(aspect / poseAspect - 1) > 0.15)) {
				poseKey = key;
				poseAspect = aspect;
				orbit.stepPose(close ? closePose(near, aspect) : overviewPose(aspect));
			}
			orbit.update(fr.dt);
			vp = mat4.multiply(insetMatrix(v, ins), rig.matrix(aspect)) as Float32Array;
			v.setViewProj(vp);

			// Draw-ins: the timeline when the scene first shows, the arrows when they first appear.
			if (axisReveal < 1) axisReveal = ctx.reducedMotion ? 1 : Math.min(1, axisReveal + fr.dt / DRAW_IN);
			axis.setReveal(axisReveal);
			ticks.setReveal(clamp((axisReveal - 0.55) / 0.45, 0, 1));
			if (fieldReveal < 1) fieldReveal = ctx.reducedMotion ? 1 : Math.min(1, fieldReveal + fr.dt / DRAW_IN);
			arrows.setReveal(fieldReveal);

			// The cloud and arrows live in data coordinates; this matrix stands them up on the sheet at the cloud's place.
			const model = mat4.scale(mat4.translation([xAt(viewT), SHEET_Y0 + SHEET / 2, LIVE_Z]), [K, K, 1]);
			liveView.setSize(v.width, v.height, v.dpr);
			liveView.setTime(fr.time);
			liveView.setViewProj(mat4.multiply(vp, model) as Float32Array);
			liveView.flush();

			if (near !== curSheet) {
				curSheet = near;
				sheetsDirty = true;
			}
			if (sheetsDirty) {
				sheetsDirty = false;
				for (let i = 0; i < LEVELS; i++) writeSheet(i, textures[i]);
			}
			placeSheets(near, viewT);
			labelsFor(near);

			if (animating && fr.dt > 0) {
				animT = Math.min(1, animT + fr.dt * NOISE_RATE);
				tCur = animT;
				emitT(animT);
				dirty = true;
				if (animT >= 1) animating = false;
			}
			if (mode === 'gen' && genIndex < genSteps && fr.dt > 0 && ctx.reducedMotion) {
				// Reduced motion: a few discrete updates, each one a batch of denoising steps shown at once.
				genTimer += fr.dt;
				if (genTimer >= REDUCED_TICK) {
					genTimer = 0;
					const batch = Math.ceil(genSteps / REDUCED_TICKS);
					for (let i = 0; i < batch && genIndex < genSteps; i++) genStep();
				}
			} else if (mode === 'gen' && genIndex < genSteps && fr.dt > 0) {
				genTimer += fr.dt;
				const interval = Math.max(MIN_INTERVAL, GEN_SECONDS / genSteps);
				let guard = 2;
				while (genTimer >= interval && genIndex < genSteps && guard-- > 0) {
					genTimer -= interval;
					genStep();
				}
				if (genTimer > interval) genTimer = interval;
			}
			if (dirty) {
				if (mode === 'scrub') noise(x0, eps, tCur, pos);
				pushTargets();
				dirty = false;
			}

			setCloudStyle();
			// Reduced motion: dots jump to their targets instead of easing there.
			cloud.step(ctx.reducedMotion && fr.dt > 0 ? INSTANT_DT : fr.dt, fr.time);
			drawField();

			// No on-stage caption: the noise level and the step count are in the readouts, and text over the blob was unreadable.
			emitReadouts();
		},
		draw(pass) {
			floor.draw(pass);
			grid.draw(pass);
			axis.draw(pass);
			ticks.draw(pass);
			shadows.draw(pass);
			sheets.draw(pass);
			marker.draw(pass);
			cloud.draw(pass);
			if (fieldCount > 0) arrows.draw(pass);
		},
		pointer(e) {
			orbit.pointer(e);
		},
		command(id) {
			if (orbit.command(id)) return;
			if (id === 'noise') {
				if (animating) {
					animating = false;
					return;
				}
				exitGen();
				if (ctx.reducedMotion) {
					tCur = 1;
					emitT(1);
					dirty = true;
					return;
				}
				if (tCur >= 0.995) {
					tCur = 0;
					emitT(0);
				}
				animT = tCur;
				animating = true;
				dirty = true;
			} else if (id === 'generate') {
				startGenerate();
			} else if (id === 'again') {
				const wasNoisy = tCur >= 0.99;
				animating = false;
				mode = 'scrub';
				reseeds++;
				fillEps();
				sheetsDirty = true;
				tCur = 1;
				emitT(1);
				dirty = true;
				teleport = wasNoisy;
			}
		},
		highlight(t) {
			hl = t;
			fieldKey = '';
		},
		dispose() {
			cloud.destroy();
			arrows.destroy();
			liveView.destroy();
			floor.dispose();
			grid.destroy();
			axis.destroy();
			ticks.destroy();
			shadows.dispose();
			sheets.dispose();
			marker.destroy();
		}
	};
	return scene;
};
