import { Camera2D } from '../../../gpu/camera.ts';
import { mixColor, tone } from '../../../gpu/colors.ts';
import { ParticleCloud } from '../../../gpu/compute/particles.ts';
import { Lines } from '../../../gpu/primitives/lines.ts';
import type { Scene, SceneContext, SceneFactory, Tone } from '../../../gpu/types.ts';
import { denoised, noise, reverseStep, sampleData, schedule } from '../../../ml/diffusion2d.ts';
import { gaussian, mulberry32 } from '../../../ml/rng.ts';
import { fitCamera, lessonInsets, type WorldRect } from '../../fit.ts';
import { N, PATTERNS } from './data.ts';

const WORLD: WorldRect = { x0: -2.6, x1: 2.6, y0: -2.1, y1: 2.1 };
/** Noise dots reach this far (3 standard deviations) at full noise; practically all of them stay inside. */
const NOISE_BOUND = 3;
const OY = 0.05; // data (0, 0) sits at world (0, OY)
const COLS = 15;
const ROWS = 12;
const GRID_X0 = -2.4;
const GRID_Y0 = -1.9;
const MAX_ARROW = 0.16;
const HEAD = 0.05;
const NOISE_RATE = 0.25; // t per second while "Add noise" plays
const GEN_SECONDS = 3.2;
const MIN_INTERVAL = 0.05;
const INSTANT_DT = 100; // seconds fed to the particle easing so it lands on its target in one frame
const REDUCED_TICK = 1 / 3; // reduced motion: one visible denoising update every third of a second
const REDUCED_TICKS = 10; // and about this many updates for a whole generation

// Particle look (tuned by eye against the clean pattern and the blob).
const DOT_RADIUS = 1.7;
const DOT_SOFTNESS = 0.6;
const DOT_DATA = { intensity: 1.05, alpha: 0.5 };
const DOT_NOISE = { intensity: 1.5, alpha: 0.6 };
const NOISE_MIX = 0.8; // how far the dot color moves from data to noise at t = 1

const clamp = (v: number, a: number, b: number) => Math.min(b, Math.max(a, v));

/**
 * What the stage must show at noise level t: the pattern shrunk by alpha plus noise of spread sigma.
 * It equals WORLD at t = 0 (so the clean pattern keeps its size) and grows to the 3-sigma disc at t = 1,
 * which then fits between the chips and the control bar. It follows t, so scrubbing or generating zooms smoothly.
 */
function viewWorld(t: number): WorldRect {
	const { alpha, sigma } = schedule(t);
	const hx = Math.hypot(alpha * WORLD.x1, NOISE_BOUND * sigma);
	const hy = Math.hypot(alpha * WORLD.y1, NOISE_BOUND * sigma);
	return { x0: -hx, x1: hx, y0: -hy, y1: hy };
}
const round2 = (v: number) => Math.round(v * 100) / 100;

export const createDiffusionScene: SceneFactory = () => {
	let ctx: SceneContext;
	let camera: Camera2D;
	let cloud: ParticleCloud;
	let arrows: Lines;

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

	function drawField() {
		const t = round2(tCur);
		const visible = fieldOn === 1 && t >= 0.02;
		const key = visible ? `${pattern}|${t}|${hl === null || hl === 'ok' ? 1 : 0}` : 'hidden';
		if (key === fieldKey) return;
		fieldKey = key;
		let n = 0;
		if (visible) {
			const m = PATTERNS[pattern].mixture;
			const mul = fieldMul();
			const sx = (-2 * GRID_X0) / (COLS - 1);
			const sy = (-2 * GRID_Y0) / (ROWS - 1);
			for (let r = 0; r < ROWS; r++) {
				for (let c = 0; c < COLS; c++) {
					const px = GRID_X0 + c * sx;
					const py = GRID_Y0 + r * sy;
					const [gx, gy] = denoised(m, px, py - OY, t);
					const dx = gx - px;
					const dy = gy + OY - py;
					const dist = Math.hypot(dx, dy);
					if (dist < 1e-6) continue;
					const len = Math.min(MAX_ARROW, 0.5 * dist);
					const ux = dx / dist;
					const uy = dy / dist;
					const bx = px + ux * len;
					const by = py + uy * len;
					const color = tone('ok', 0.85, (0.15 + 0.65 * Math.min(1, dist / 1.2)) * mul);
					arrows.set(n++, { a: [px, py], b: [bx, by], width: 1.5, color });
					for (const s of [1, -1]) {
						const ang = s * 0.5;
						const hx = -(ux * Math.cos(ang) - uy * Math.sin(ang) * 1) * HEAD;
						const hy = -(uy * Math.cos(ang) + ux * Math.sin(ang) * 1) * HEAD;
						arrows.set(n++, { a: [bx, by], b: [bx + hx, by + hy], width: 1.5, color });
					}
				}
			}
		}
		arrows.count = n;
		fieldCount = n;
		arrows.upload();
	}

	function setCloudStyle() {
		// Cyan and amber mix to gray, so the hue changes late and quickly instead of linearly with t.
		const u = clamp((tCur - 0.62) / 0.33, 0, 1);
		const k = u * u * (3 - 2 * u) * NOISE_MIX;
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
			t[i * 2 + 1] = pos[i * 2 + 1] + OY;
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

	const scene: Scene = {
		init(c) {
			ctx = c;
			camera = new Camera2D();
			cloud = new ParticleCloud(c, { capacity: N, blend: 'alpha' });
			arrows = new Lines(c, { capacity: COLS * ROWS * 3, blend: 'alpha' });
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
			// The framing eases toward the noise level, so a few-step run does not make the whole view jump.
			viewT = ctx.reducedMotion || !viewSeen ? tCur : viewT + (tCur - viewT) * (1 - Math.exp(-fr.dt / 0.15));
			viewSeen = true;
			fitCamera(camera, v, viewWorld(viewT), lessonInsets(v));
			v.setViewProj(camera.matrix(v.aspect));

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
			cloud.draw(pass);
			if (fieldCount > 0) arrows.draw(pass);
		},
		command(id) {
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
		}
	};
	return scene;
};
