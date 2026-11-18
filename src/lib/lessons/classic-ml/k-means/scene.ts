import { Camera2D } from '../../../gpu/camera.ts';
import { tone } from '../../../gpu/colors.ts';
import { ParticleCloud } from '../../../gpu/compute/particles.ts';
import { Lines } from '../../../gpu/primitives/lines.ts';
import { Points } from '../../../gpu/primitives/points.ts';
import type { Label, Scene, SceneContext, SceneFactory, ScenePointer, Tone } from '../../../gpu/types.ts';
import { assign, initCenters, updateCenters } from '../../../ml/kmeans.ts';
import { mulberry32 } from '../../../ml/rng.ts';
import { fitCamera, lessonInsets, type WorldRect } from '../../fit.ts';
import { N, makeCloud } from './data.ts';
import { elbowCurve } from './elbow.ts';

const DATA_SEED = 1;
const EXTENT = 1.5; // the map covers [-EXTENT, EXTENT] on both axes
const MAX_K = 6;
const MAP_OX = -0.6; // data (0, 0) sits at world (MAP_OX, 0)
const WORLD: WorldRect = { x0: -2.15, x1: 2.7, y0: -1.68, y1: 1.55 };
const CHART = { x0: 1.3, x1: 2.35, y0: -0.9, y1: 0.5 };
const CLUSTER_TONES: Tone[] = ['data', 'param', 'grad', 'neg', 'ok', 'text'];
const SPOKES = 300;
const MAX_TRAIL = 24;
const MAX_HISTORY = 60;
const STEP_SECONDS = 0.6;
const GRAB_PX = 16;
const LIVE_REFRESH = 0.1;
const CONVERGED = 1e-4;
const INSTANT_DT = 100; // seconds fed to the particle easing so it lands on its target in one frame

const clamp = (v: number, a: number, b: number) => Math.min(b, Math.max(a, v));

export const createKMeansScene: SceneFactory = () => {
	let ctx: SceneContext;
	let camera: Camera2D;
	let cloud: ParticleCloud;
	let markers: Points;
	let chartPts: Points;
	let frame: Lines;
	let trails: Lines;
	let spokes: Lines;
	let chart: Lines;

	// params
	let k = 3;
	let datasetId = 0;
	let initMethod = 0;
	let prep = 2;
	let live = 0;
	let spokesOn = 0;
	let chartKind = 0;
	let autorun = 0;
	let seedParam = 1;
	let scenario = 1;
	let paramsSeen = false;
	let hl: Tone | null = null;

	// data
	let points: Float64Array = new Float64Array(0);
	let loadedDataset = -1;
	const spokeIdx = Uint32Array.from({ length: SPOKES }, (_, i) => i * Math.floor(N / SPOKES));

	// k-means state
	let reseeds = 0;
	let trueC: Float64Array = new Float64Array(0);
	const disp = new Float64Array(MAX_K * 2);
	const assignment = new Int32Array(N);
	let counts: Int32Array = new Int32Array(MAX_K);
	let inertia = NaN;
	let assigned = false;
	let centersVisible = false;
	let history: number[] = [];
	const trail: number[][] = Array.from({ length: MAX_K }, () => []);
	let lastOp: 'assign' | 'move' | null = null;
	let running = false;
	let runTimer = 0;
	let converged = false;

	// pointer
	let dragging = -1;
	let dragDirty = false;
	let liveTimer = 0;

	// elbow chart cache
	const elbow = new Float64Array(MAX_K);
	let elbowDataset = -1;

	// emitted readouts
	let lastIter = -1;
	let lastInertia = '';

	const mul = (t: Tone) => (hl === null || hl === t ? 1 : 0.3);

	function loadData(first: boolean) {
		if (loadedDataset === datasetId) return;
		loadedDataset = datasetId;
		points = makeCloud(datasetId, DATA_SEED);
		for (let i = 0; i < N; i++) {
			cloud.targets[i * 2] = points[i * 2] + MAP_OX;
			cloud.targets[i * 2 + 1] = points[i * 2 + 1];
		}
		cloud.uploadTargets();
		if (first) {
			const rand = mulberry32(99);
			const start = new Float32Array(N * 2);
			for (let i = 0; i < N; i++) {
				start[i * 2] = MAP_OX + (rand() * 2 - 1) * EXTENT;
				start[i * 2 + 1] = (rand() * 2 - 1) * EXTENT;
			}
			cloud.setPositions(start);
		}
	}

	function syncGpuCenters(src: Float64Array) {
		for (let c = 0; c < k; c++) {
			cloud.centers[c * 2] = src[c * 2] + MAP_OX;
			cloud.centers[c * 2 + 1] = src[c * 2 + 1];
		}
	}

	function pushTrail(c: number, x: number, y: number) {
		const t = trail[c];
		t.push(x, y);
		if (t.length > MAX_TRAIL * 2) t.splice(0, 2);
	}

	/** Assign on the CPU only (no history, no GPU). */
	function cpuAssign() {
		inertia = assign(points, trueC, assignment);
		assigned = true;
	}

	function doAssign() {
		cpuAssign();
		counts = updateCenters(points, assignment, trueC).counts;
		syncGpuCenters(trueC);
		cloud.groupCount = k;
		if (history.length < MAX_HISTORY) history.push(inertia);
		else history[history.length - 1] = inertia;
		lastOp = 'assign';
		converged = false;
	}

	function doMove() {
		if (!assigned) doAssign();
		const u = updateCenters(points, assignment, trueC);
		trueC = u.centers;
		counts = u.counts;
		for (let c = 0; c < k; c++) pushTrail(c, trueC[c * 2], trueC[c * 2 + 1]);
		lastOp = 'move';
		converged = u.moved < CONVERGED;
		if (converged) running = false;
	}

	function reset(autostart: boolean) {
		loadData(!paramsSeen);
		k = clamp(k, 1, MAX_K);
		trueC = initCenters(points, k, seedParam + reseeds, initMethod === 1 ? 'plusplus' : 'random');
		disp.set(trueC);
		for (let c = 0; c < MAX_K; c++) {
			trail[c].length = 0;
			if (c < k) pushTrail(c, trueC[c * 2], trueC[c * 2 + 1]);
		}
		history = [];
		inertia = NaN;
		assigned = false;
		converged = false;
		running = false;
		lastOp = null;
		dragging = -1;
		cloud.groupCount = 0;
		for (let c = 0; c < k; c++) {
			const col = tone(CLUSTER_TONES[c], 1, 0.5);
			cloud.palette.set(col, c * 4);
		}
		centersVisible = prep >= 1;
		if (prep >= 2) doAssign();
		if (autostart) {
			running = true;
			runTimer = STEP_SECONDS * 0.6;
		}
	}

	function computeElbow() {
		elbowDataset = datasetId;
		elbow.set(elbowCurve(points, MAX_K));
	}

	function setCloudStyle() {
		const m = mul('muted');
		cloud.neutralColor = tone('muted', 1.6, 0.4 * m);
		for (let c = 0; c < MAX_K; c++) {
			const col = tone(CLUSTER_TONES[c], 1, 0.5 * m);
			cloud.palette.set(col, c * 4);
		}
		cloud.radius = 1.6;
		cloud.softness = 0.35;
		cloud.follow = 5;
		cloud.colorFollow = 9;
		cloud.jitter = ctx.reducedMotion ? 0 : 0.004;
	}

	function drawFrameBox() {
		const x0 = MAP_OX - EXTENT;
		const x1 = MAP_OX + EXTENT;
		const pts: Array<[number, number]> = [
			[x0, -EXTENT],
			[x1, -EXTENT],
			[x1, EXTENT],
			[x0, EXTENT],
			[x0, -EXTENT]
		];
		frame.count = frame.setPolyline(0, pts, { width: 1, color: tone('muted', 1, 0.6) });
		frame.upload();
	}

	function drawCenters() {
		let n = 0;
		const m = mul('text');
		if (centersVisible) {
			for (let c = 0; c < k; c++) {
				const t = CLUSTER_TONES[c];
				const x = disp[c * 2] + MAP_OX;
				const y = disp[c * 2 + 1];
				markers.set(n++, { x, y, radius: 24, softness: 1, color: tone(t, 1, 0.2 * m) });
				markers.set(n++, { x, y, radius: 11, color: [0.004, 0.005, 0.01, 0.75 * m] });
				markers.set(n++, { x, y, radius: 6.5, color: tone(t, 1.25, 0.98 * m) });
				markers.set(n++, { x, y, radius: 10.5, ring: 2, color: tone('text', 1, 0.9 * m) });
			}
		}
		markers.count = n;
		markers.upload();
	}

	function drawTrails() {
		let n = 0;
		const m = mul('text');
		if (centersVisible) {
			for (let c = 0; c < k; c++) {
				const t = trail[c];
				const len = t.length / 2;
				// segments through the history, then one to the displayed marker
				const total = len; // len - 1 history segments + 1 to the marker
				for (let j = 0; j < total; j++) {
					const ax = t[j * 2] + MAP_OX;
					const ay = t[j * 2 + 1];
					const bx = (j + 1 < len ? t[j * 2 + 2] : disp[c * 2]) + MAP_OX;
					const by = j + 1 < len ? t[j * 2 + 3] : disp[c * 2 + 1];
					if (ax === bx && ay === by) continue;
					const f = (j + 1) / total;
					trails.set(n++, {
						a: [ax, ay],
						b: [bx, by],
						width: 1.5,
						color: tone(CLUSTER_TONES[c], 0.95, (0.12 + 0.55 * f) * m)
					});
				}
			}
		}
		trails.count = n;
		trails.upload();
	}

	function drawSpokes() {
		let n = 0;
		if (spokesOn && centersVisible && assigned) {
			const m = mul('grad');
			for (let s = 0; s < SPOKES; s++) {
				const i = spokeIdx[s];
				const c = assignment[i];
				if (c >= k) continue;
				spokes.set(n++, {
					a: [points[i * 2] + MAP_OX, points[i * 2 + 1]],
					b: [disp[c * 2] + MAP_OX, disp[c * 2 + 1]],
					width: 1,
					color: tone(CLUSTER_TONES[c], 0.9, 0.2 * m)
				});
			}
		}
		spokes.count = n;
		spokes.upload();
	}

	function drawChart(labels: Label[]) {
		if (chartKind === 0 && prep === 0) {
			chart.count = 0;
			chartPts.count = 0;
			return;
		}
		const v = ctx.view;
		const m = mul('grad');
		const w = CHART.x1 - CHART.x0;
		const h = CHART.y1 - CHART.y0;
		const axis = tone('muted', 1, 0.6);
		let n = 0;
		let pn = 0;
		chart.set(n++, { a: [CHART.x0, CHART.y0], b: [CHART.x0, CHART.y1], width: 1, color: axis });
		chart.set(n++, { a: [CHART.x0, CHART.y0], b: [CHART.x1, CHART.y0], width: 1, color: axis });

		const [ax, ay] = v.project(CHART.x0, CHART.y1);
		labels.push({ id: 'chart-y', x: ax, y: ay - 14, text: 'spread', tone: 'muted', mono: true, align: 'left' });
		const [bx, by] = v.project(CHART.x1, CHART.y0);
		const xCaption = chartKind === 1 ? 'centers' : 'rounds';
		labels.push({ id: 'chart-x', x: bx, y: by + (chartKind === 1 ? 32 : 16), text: xCaption, tone: 'muted', mono: true, align: 'right' });

		if (chartKind === 1) {
			if (elbowDataset !== datasetId) computeElbow();
			const top = Math.max(elbow[0], 1e-6) * 1.05;
			const pts: Array<[number, number]> = [];
			for (let i = 0; i < MAX_K; i++) {
				pts.push([CHART.x0 + (w * (i + 0.5)) / MAX_K, CHART.y0 + h * clamp(elbow[i] / top, 0, 1)]);
				const [tx, ty] = v.project(pts[i][0], CHART.y0);
				labels.push({ id: `tick-${i}`, x: tx, y: ty + 14, text: String(i + 1), tone: 'muted', mono: true });
			}
			n = chart.setPolyline(n, pts, { width: 7, color: tone('grad', 1, 0.08 * m) });
			n = chart.setPolyline(n, pts, { width: 2, color: tone('grad', 1.5, 0.95 * m) });
			for (let i = 0; i < MAX_K; i++) {
				chartPts.set(pn++, { x: pts[i][0], y: pts[i][1], radius: 3.5, color: tone('grad', 1.4, 0.95 * m) });
			}
			const cur = pts[clamp(k, 1, MAX_K) - 1];
			chartPts.set(pn++, { x: cur[0], y: cur[1], radius: 8, ring: 2, color: tone('text', 1.3, 0.95 * m) });
			const [ex, ey] = v.project(cur[0], cur[1]);
			labels.push({
				id: 'chart-now',
				x: ex + 12,
				y: ey - 12,
				text: elbow[clamp(k, 1, MAX_K) - 1].toFixed(0),
				tone: 'grad',
				mono: true,
				align: 'left',
				opacity: m
			});
		} else if (history.length > 0) {
			const xMax = Math.max(6, history.length);
			const top = Math.max(...history) * 1.3;
			const px = (i: number) => CHART.x0 + (w * (i + 0.5)) / xMax;
			const py = (val: number) => CHART.y0 + h * clamp(val / top, 0, 1);
			// stepped polyline: hold each value until the next assign
			const pts: Array<[number, number]> = [];
			for (let i = 0; i < history.length; i++) {
				pts.push([px(i), py(history[i])]);
				if (i + 1 < history.length) pts.push([px(i + 1), py(history[i])]);
				chartPts.set(pn++, { x: px(i), y: py(history[i]), radius: 3, color: tone('grad', 1.4, 0.95 * m) });
			}
			if (pts.length >= 2) {
				n = chart.setPolyline(n, pts, { width: 7, color: tone('grad', 1, 0.08 * m) });
				n = chart.setPolyline(n, pts, { width: 2, color: tone('grad', 1.5, 0.95 * m) });
			}
			const last = history.length - 1;
			const [ex, ey] = v.project(px(last), py(history[last]));
			labels.push({
				id: 'chart-now',
				x: ex,
				y: ey - 14,
				text: history[last].toFixed(1),
				tone: 'grad',
				mono: true,
				align: 'center',
				opacity: m
			});
		}
		chart.count = n;
		chart.upload();
		chartPts.count = pn;
		chartPts.upload();
	}

	function emitReadouts() {
		const iter = history.length;
		if (iter !== lastIter) {
			lastIter = iter;
			ctx.emit('iter', iter);
		}
		const text = assigned && Number.isFinite(inertia) ? inertia.toFixed(1) : '—';
		if (text !== lastInertia) {
			lastInertia = text;
			ctx.emit('inertia', text);
		}
	}

	function nearestMarker(x: number, y: number): number {
		let best = -1;
		let bd = GRAB_PX;
		for (let c = 0; c < k; c++) {
			const [sx, sy] = ctx.view.project(disp[c * 2] + MAP_OX, disp[c * 2 + 1]);
			const d = Math.hypot(sx - x, sy - y);
			if (d <= bd) {
				bd = d;
				best = c;
			}
		}
		return best;
	}

	function dragTo(x: number, y: number) {
		const [wx, wy] = ctx.view.unproject(x, y);
		const dx = clamp(wx - MAP_OX, -EXTENT, EXTENT);
		const dy = clamp(wy, -EXTENT, EXTENT);
		trueC[dragging * 2] = dx;
		trueC[dragging * 2 + 1] = dy;
		disp[dragging * 2] = dx;
		disp[dragging * 2 + 1] = dy;
		converged = false;
		dragDirty = true;
	}

	function endDrag() {
		if (dragging < 0) return;
		const c = dragging;
		dragging = -1;
		const t = trail[c];
		const lx = t[t.length - 2];
		const ly = t[t.length - 1];
		if (Math.hypot(trueC[c * 2] - lx, trueC[c * 2 + 1] - ly) > 0.01) pushTrail(c, trueC[c * 2], trueC[c * 2 + 1]);
		if (live && centersVisible) {
			cpuAssign();
			dragDirty = false;
		}
	}

	const scene: Scene = {
		init(c) {
			ctx = c;
			camera = new Camera2D();
			cloud = new ParticleCloud(c, { capacity: N, blend: 'alpha' });
			frame = new Lines(c, { capacity: 8, blend: 'alpha' });
			trails = new Lines(c, { capacity: MAX_K * (MAX_TRAIL + 1), blend: 'alpha' });
			spokes = new Lines(c, { capacity: SPOKES, blend: 'alpha' });
			chart = new Lines(c, { capacity: 3 * MAX_HISTORY * 2 + 16, blend: 'alpha' });
			markers = new Points(c, { capacity: MAX_K * 4, blend: 'alpha' });
			chartPts = new Points(c, { capacity: MAX_HISTORY + 8, blend: 'alpha' });
			setCloudStyle();
			drawFrameBox();
		},
		setParams(p) {
			const prev = { scenario, datasetId, k, initMethod, seedParam, prep };
			scenario = Math.round(p.scenario ?? scenario);
			datasetId = clamp(Math.round(p.dataset ?? datasetId), 0, 3);
			k = clamp(Math.round(p.k ?? k), 1, MAX_K);
			initMethod = Math.round(p.init ?? initMethod);
			seedParam = Math.round(p.seed ?? seedParam);
			prep = Math.round(p.prep ?? prep);
			live = Math.round(p.live ?? live);
			spokesOn = Math.round(p.spokes ?? spokesOn);
			chartKind = Math.round(p.chart ?? chartKind);
			autorun = Math.round(p.autorun ?? autorun);
			const kOnly =
				prev.k !== k &&
				prev.scenario === scenario &&
				prev.datasetId === datasetId &&
				prev.initMethod === initMethod &&
				prev.seedParam === seedParam &&
				prev.prep === prep;
			const changed =
				prev.scenario !== scenario ||
				prev.datasetId !== datasetId ||
				prev.k !== k ||
				prev.initMethod !== initMethod ||
				prev.seedParam !== seedParam ||
				prev.prep !== prep;
			if (!paramsSeen || changed) {
				if (!paramsSeen) reseeds = 0;
				reset(paramsSeen && kOnly && autorun === 1);
			}
			paramsSeen = true;
		},
		update(fr) {
			const v = ctx.view;
			fitCamera(camera, v, WORLD, lessonInsets(v));
			v.setViewProj(camera.matrix(v.aspect));

			// automatic alternation
			if (running && fr.dt > 0 && dragging < 0) {
				runTimer -= fr.dt;
				if (runTimer <= 0) {
					runTimer = STEP_SECONDS;
					if (lastOp === 'assign') doMove();
					else doAssign();
					if (history.length >= MAX_HISTORY) running = false;
				}
			}

			// marker easing toward the true positions
			const ease = ctx.reducedMotion ? 1 : 1 - Math.exp(-Math.max(0, fr.dt) * 6);
			for (let c = 0; c < k; c++) {
				if (c === dragging) continue;
				disp[c * 2] += (trueC[c * 2] - disp[c * 2]) * ease;
				disp[c * 2 + 1] += (trueC[c * 2 + 1] - disp[c * 2 + 1]) * ease;
			}

			// live mode: the GPU colors follow the markers every frame
			if (live === 1 && centersVisible) {
				syncGpuCenters(disp);
				cloud.groupCount = k;
				if (dragging >= 0 && dragDirty) {
					liveTimer -= fr.dt;
					if (liveTimer <= 0) {
						liveTimer = LIVE_REFRESH;
						cpuAssign();
						dragDirty = false;
					}
				}
			}

			setCloudStyle();
			// Reduced motion: positions and colors jump to their targets instead of easing there.
			cloud.step(ctx.reducedMotion && fr.dt > 0 ? INSTANT_DT : fr.dt, fr.time);

			drawTrails();
			drawSpokes();
			drawCenters();

			const labels: Label[] = [];
			drawChart(labels);
			if (converged) {
				const [mx, my] = v.project(MAP_OX, -EXTENT);
				labels.push({ id: 'converged', x: mx, y: my + 16, text: 'converged', tone: 'ok', mono: true });
			}
			ctx.labels.set(labels);
			emitReadouts();
		},
		draw(pass) {
			cloud.draw(pass);
			frame.draw(pass);
			spokes.draw(pass);
			trails.draw(pass);
			chart.draw(pass);
			chartPts.draw(pass);
			markers.draw(pass);
		},
		command(id) {
			if (id === 'assign') {
				running = false;
				if (centersVisible) doAssign();
			} else if (id === 'move') {
				running = false;
				if (centersVisible) doMove();
			} else if (id === 'run') {
				if (!centersVisible) return;
				running = !running;
				if (running) {
					converged = false;
					runTimer = 0;
				}
			} else if (id === 'reseed') {
				reseeds++;
				reset(false);
			}
		},
		highlight(t) {
			hl = t;
		},
		pointer(e: ScenePointer) {
			if (e.type === 'down') {
				if (!centersVisible) return;
				const c = nearestMarker(e.x, e.y);
				if (c >= 0) {
					dragging = c;
					running = false;
					liveTimer = 0;
					dragTo(e.x, e.y);
				}
			} else if (e.type === 'move') {
				if (dragging >= 0) dragTo(e.x, e.y);
			} else {
				endDrag();
			}
		},
		dispose() {
			cloud.destroy();
			markers.destroy();
			chartPts.destroy();
			frame.destroy();
			trails.destroy();
			spokes.destroy();
			chart.destroy();
		}
	};
	return scene;
};
