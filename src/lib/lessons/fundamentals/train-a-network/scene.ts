import { Camera2D } from '../../../gpu/camera.ts';
import { paperFill, tone } from '../../../gpu/colors.ts';
import { Field } from '../../../gpu/primitives/field.ts';
import { Lines } from '../../../gpu/primitives/lines.ts';
import { Points } from '../../../gpu/primitives/points.ts';
import type { Label, Scene, SceneContext, SceneFactory, Tone } from '../../../gpu/types.ts';
import { makeDataset, type Dataset, type DatasetId } from '../../../ml/datasets.ts';
import { createMLP, forward, lossValue, predict, trainStep, type MLP } from '../../../ml/mlp.ts';
import { fitCamera, lessonInsets, type WorldRect } from '../../fit.ts';
import { NetworkDiagram } from '../../shared/network-diagram.ts';
import { LossHistory, MAX_SAMPLES } from './loss-history.ts';

const N_POINTS = 200;
const DATA_SEED = 11;
const FIELD_N = 64;
const EXTENT = 1.4; // the decision map covers inputs [-EXTENT, EXTENT] on both axes
const STEPS_PER_SECOND = 180;
const MAX_STEPS_PER_FRAME = 4;

// World layout: the map's input (0, 0) sits at world (MAP_OX, 0).
const MAP_OX = -2.2;
const WORLD: WorldRect = { x0: -3.7, x1: 3.65, y0: -1.62, y1: 1.55 };
const DIAGRAM: WorldRect = { x0: 0.2, x1: 3.3, y0: 0.2, y1: 1.2 };
const CHART = { x0: 0.25, x1: 3.0, y0: -1.3, y1: -0.15 };

// Free mode (the playground): no in-scene loss chart, the network diagram on the left and the map on the right,
// both centred vertically. The map (2.8 wide) mirrors the lesson layout's, so the world is the same width.
const FREE_MAP_OX = 1.9;
const FREE_WORLD: WorldRect = { x0: -3.7, x1: 3.65, y0: -1.62, y1: 1.55 };
const FREE_DIAGRAM: WorldRect = { x0: -3.4, x1: -0.2, y0: -0.9, y1: 0.9 };

const clamp = (v: number, a: number, b: number) => Math.min(b, Math.max(a, v));

export const createTrainScene: SceneFactory = () => {
	let ctx: SceneContext;
	let camera: Camera2D;
	let field: Field;
	let fieldB: Field;
	let frame: Lines;
	let dots: Points;
	let chart: Lines;
	let diagram: NetworkDiagram | null = null;

	// params
	let datasetId = 0;
	let hidden = 4;
	let layers = 1;
	let act = 0;
	let lr = 0.3;
	let noise = 0.05;
	let scenario = 1;
	let free = 0;
	let paramsSeen = false;
	let noiseUsed = 0.05;
	let hl: Tone | null = null;

	// simulation
	let data: Dataset = makeDataset(0, N_POINTS, 0.05, DATA_SEED);
	let net: MLP;
	let seed = 5;
	let running = false;
	let acc = 0;
	let steps = 0;
	let blewUp = false;
	let dirty = false;
	const history = new LossHistory();
	const indices = Uint32Array.from({ length: N_POINTS }, (_, i) => i);
	const wrong = new Uint8Array(N_POINTS);
	let loss = 0;
	let accuracy = 0;

	// emitted readouts
	let lastLoss = NaN;
	let lastAcc = '';
	let lastSteps = -1;
	let lastRunning = -1;

	/** The active layout: the lesson's, or the free one. */
	const mapOx = () => (free ? FREE_MAP_OX : MAP_OX);
	const worldRect = () => (free ? FREE_WORLD : WORLD);
	const diagramRect = () => (free ? FREE_DIAGRAM : DIAGRAM);

	const mul = (t: Tone) => (hl === null || hl === t ? 1 : 0.3);

	function makeNet(): MLP {
		const sizes = [2, ...Array<number>(layers).fill(hidden), 1];
		return createMLP(sizes, act >= 0.5 ? 'relu' : 'tanh', 'sigmoid', seed);
	}

	/** Refill the decision map, accuracy, loss and hidden-node activity from the current weights. */
	function refresh() {
		dirty = false;
		const half = FIELD_N;
		const span = 2 * EXTENT;
		const px = new Float64Array(2);
		for (let r = 0; r < FIELD_N; r++) {
			px[1] = -EXTENT + ((r + 0.5) / half) * span;
			for (let c = 0; c < FIELD_N; c++) {
				px[0] = -EXTENT + ((c + 0.5) / half) * span;
				const p = predict(net, px)[0];
				field.data[r * FIELD_N + c] = Number.isFinite(p) ? p : 0.5;
			}
		}

		const L = net.sizes.length;
		const sums = net.sizes.map((n) => new Float64Array(n));
		let correct = 0;
		let total = 0;
		const out = new Float64Array(1);
		const target = new Float64Array(1);
		for (let k = 0; k < N_POINTS; k++) {
			const tr = forward(net, data.x.subarray(k * 2, k * 2 + 2));
			for (let l = 1; l < L - 1; l++) for (let i = 0; i < net.sizes[l]; i++) sums[l][i] += Math.abs(tr.a[l][i]);
			out[0] = tr.a[L - 1][0];
			target[0] = data.y[k];
			total += lossValue('bce', out, target);
			const ok = Math.round(out[0]) === data.y[k];
			wrong[k] = ok ? 0 : 1;
			if (ok) correct++;
		}
		loss = total / N_POINTS;
		accuracy = correct / N_POINTS;
		if (!Number.isFinite(loss)) blewUp = true;

		if (diagram) {
			for (let l = 1; l < L - 1; l++) {
				for (let i = 0; i < net.sizes[l]; i++) diagram.node(l, i).value = sums[l][i] / N_POINTS;
			}
			diagram.node(L - 1, 0).value = accuracy;
		}
		fieldB.data.set(field.data);
		field.upload();
		fieldB.upload();
	}

	function reset(fromAction: boolean) {
		if (fromAction) seed++;
		else seed = 5;
		running = false;
		acc = 0;
		steps = 0;
		blewUp = false;
		noiseUsed = noise;
		data = makeDataset(clamp(Math.round(datasetId), 0, 3) as DatasetId, N_POINTS, noise, DATA_SEED);
		net = makeNet();
		if (!diagram || diagram.sizes.join() !== net.sizes.join()) {
			diagram?.destroy();
			diagram = new NetworkDiagram(ctx, net.sizes, { nodeRadius: 8 });
			diagram.rect = diagramRect();
			const last = net.sizes.length - 1;
			for (let l = 0; l <= last; l++) {
				for (let i = 0; i < net.sizes[l]; i++) {
					const s = diagram.node(l, i);
					s.value = l === 0 ? 1 : 0;
					if (l === 0) s.name = i === 0 ? 'x₁' : 'x₂';
					if (l === last) s.name = 'ŷ';
				}
			}
		}
		refresh();
		history.reset(loss);
		lastLoss = NaN;
		lastAcc = '';
		lastSteps = -1;
		lastRunning = -1;
	}

	function doStep() {
		if (blewUp) {
			running = false;
			return;
		}
		const l = trainStep(net, data.x, data.y, indices, lr, 'bce');
		steps++;
		dirty = true;
		if (!Number.isFinite(l)) {
			blewUp = true;
			running = false;
			return;
		}
		history.push(l);
	}

	function drawDots(): void {
		const mData = mul('data');
		const mNeg = mul('neg');
		const mGrad = mul('grad');
		let n = 0;
		for (let pass = 0; pass < 3; pass++) {
			for (let k = 0; k < N_POINTS; k++) {
				const cls = data.y[k] >= 0.5;
				const t: Tone = cls ? 'data' : 'neg';
				const m = cls ? mData : mNeg;
				const x = data.x[k * 2] + mapOx();
				const y = data.x[k * 2 + 1];
				if (pass === 0) {
					// thin paper ring under the ink dot so it reads on top of the wash
					dots.set(n++, { x, y, radius: 5, color: paperFill(0.9) });
				} else if (pass === 1) {
					dots.set(n++, { x, y, radius: 3.5, color: tone(t, m) });
				} else if (wrong[k]) {
					dots.set(n++, { x, y, radius: 6.5, ring: 1.4, color: tone('grad', mGrad) });
				}
			}
		}
		dots.count = n;
		dots.upload();
	}

	function drawFrame(): void {
		const x0 = mapOx() - EXTENT;
		const x1 = mapOx() + EXTENT;
		const c = tone('muted', 0.6);
		const pts: Array<[number, number]> = [
			[x0, -EXTENT],
			[x1, -EXTENT],
			[x1, EXTENT],
			[x0, EXTENT],
			[x0, -EXTENT]
		];
		frame.count = frame.setPolyline(0, pts, { width: 1, color: c });
		frame.upload();
	}

	function drawChart(labels: Label[]): void {
		const v = ctx.view;
		const m = mul('grad');
		const w = CHART.x1 - CHART.x0;
		const h = CHART.y1 - CHART.y0;
		const axis = tone('muted', 0.6);
		let n = 0;
		chart.set(n++, { a: [CHART.x0, CHART.y0], b: [CHART.x0, CHART.y1], width: 1, color: axis });
		chart.set(n++, { a: [CHART.x0, CHART.y0], b: [CHART.x1, CHART.y0], width: 1, color: axis });

		const vals = history.values;
		const top = Math.max(history.max, 0.05);
		const pts: Array<[number, number]> = vals.map((lv, i) => [
			CHART.x0 + (w * i) / MAX_SAMPLES,
			CHART.y0 + h * clamp(lv / top, 0, 1)
		]);
		if (pts.length >= 2) {
			n = chart.setPolyline(n, pts, { width: 2, color: tone('grad', m) });
		}
		chart.count = n;
		chart.upload();

		const [ax, ay] = v.project(CHART.x0, CHART.y1);
		labels.push({ id: 'chart-loss', x: ax - 6, y: ay, text: 'loss', tone: 'muted', mono: true, align: 'right' });
		const [bx, by] = v.project(CHART.x1, CHART.y0);
		labels.push({ id: 'chart-steps', x: bx, y: by + 14, text: 'steps', tone: 'muted', mono: true, align: 'right' });
		const last = pts.length > 0 ? pts[pts.length - 1] : [CHART.x0, CHART.y0 + h * clamp(loss / top, 0, 1)];
		const [ex, ey] = v.project(last[0], last[1]);
		labels.push({
			id: 'chart-now',
			x: ex + 8,
			y: ey,
			text: Number.isFinite(loss) ? loss.toFixed(3) : '—',
			tone: 'grad',
			mono: true,
			align: 'left',
			opacity: m
		});
	}

	function updateDiagram(dt: number): void {
		if (!diagram) return;
		const L = net.sizes.length;
		const m = mul('param');
		for (let l = 0; l < L; l++) {
			for (let i = 0; i < net.sizes[l]; i++) diagram.node(l, i).emphasis = m;
		}
		for (let l = 0; l + 1 < L; l++) {
			for (let i = 0; i < net.sizes[l]; i++) {
				for (let j = 0; j < net.sizes[l + 1]; j++) {
					const e = diagram.edge(l, i, j);
					e.weight = 0.5 * net.W[l][j * net.sizes[l] + i];
					e.signal = 0;
					e.emphasis = m;
				}
			}
		}
		diagram.commit();
		diagram.update(dt);
	}

	const scene: Scene = {
		init(c) {
			ctx = c;
			camera = new Camera2D();
			field = new Field(c, { cols: FIELD_N, rows: FIELD_N });
			field.rect = [mapOx() - EXTENT, -EXTENT, mapOx() + EXTENT, EXTENT];
			field.iso = 0.5;
			fieldB = new Field(c, { cols: FIELD_N, rows: FIELD_N });
			fieldB.rect = field.rect;
			fieldB.iso = -1;
			field.isoWidth = 2;
			frame = new Lines(c, { capacity: 8, blend: 'alpha' });
			dots = new Points(c, { capacity: N_POINTS * 3, blend: 'alpha' });
			chart = new Lines(c, { capacity: 2 * MAX_SAMPLES + 8, blend: 'alpha' });
			drawFrame();
		},
		setParams(p) {
			const prev = { datasetId, hidden, layers, act, scenario };
			const prevFree = free;
			free = Math.round(p.free ?? 0);
			datasetId = Math.round(p.dataset ?? datasetId);
			hidden = Math.round(p.hidden ?? hidden);
			layers = Math.round(p.layers ?? layers);
			act = Math.round(p.act ?? act);
			scenario = Math.round(p.scenario ?? scenario);
			lr = p.lr ?? lr;
			noise = p.noise ?? noise;
			const changed =
				prev.datasetId !== datasetId ||
				prev.hidden !== hidden ||
				prev.layers !== layers ||
				prev.act !== act ||
				prev.scenario !== scenario ||
				Math.abs(noise - noiseUsed) > 0.001;
			if (prevFree !== free && field) {
				// Layout switched after init: move the map and the diagram to the active layout.
				field.rect = [mapOx() - EXTENT, -EXTENT, mapOx() + EXTENT, EXTENT];
				fieldB.rect = field.rect;
				if (diagram) diagram.rect = diagramRect();
			}
			if (!paramsSeen || changed) reset(false);
			paramsSeen = true;
		},
		update(fr) {
			const v = ctx.view;
			fitCamera(camera, v, worldRect(), lessonInsets(v));
			v.setViewProj(camera.matrix(v.aspect));

			if (running && fr.dt > 0) {
				// Reduced motion: 3 visible updates per second, each doing the same 60 training steps
				// that 1/3 s of full-speed training would, so the map and chart jump between updates.
				const reduced = ctx.reducedMotion;
				acc += fr.dt * (reduced ? 3 : STEPS_PER_SECOND);
				let n = Math.floor(acc);
				acc -= n;
				const cap = reduced ? 1 : MAX_STEPS_PER_FRAME;
				if (n > cap) {
					n = cap;
					acc = 0;
				}
				n *= reduced ? STEPS_PER_SECOND / 3 : 1;
				for (let i = 0; i < n && running; i++) doStep();
			}
			if (dirty) refresh();

			// Washes: light ink so the class regions read as tinted paper; the iso-line is the boundary.
			// Two layers, each fading to transparent in its own ink, so unsure regions read as paper.
			const neg = tone('neg', 0.22 * mul('neg'));
			const pos = tone('data', 0.22 * mul('data'));
			field.colorLo = neg;
			field.colorHi = [neg[0], neg[1], neg[2], 0];
			fieldB.colorLo = [pos[0], pos[1], pos[2], 0];
			fieldB.colorHi = pos;
			field.upload();
			fieldB.upload();
			field.isoColor = tone('text', 1);
			drawFrame();
			drawDots();
			updateDiagram(fr.dt);

			const labels: Label[] = [];
			if (!free) drawChart(labels);
			else {
				chart.count = 0;
				chart.upload();
			}
			const [mx, my] = v.project(mapOx(), -EXTENT);
			labels.push({ id: 'map-caption', x: mx, y: my + 16, text: 'what the network predicts', tone: 'muted' });
			if (diagram) labels.push(...diagram.labels(v));
			if (blewUp) {
				const [cx, cy] = v.project(mapOx(), 0);
				labels.push({ id: 'blowup', x: cx, y: cy, text: 'Training blew up — press Reset', tone: 'neg' });
			}
			ctx.labels.set(labels);

			if (Number.isFinite(loss) && loss !== lastLoss) {
				lastLoss = loss;
				ctx.emit('loss', loss);
			}
			const accText = `${Math.round(accuracy * 100)}%`;
			if (accText !== lastAcc) {
				lastAcc = accText;
				ctx.emit('acc', accText);
			}
			if (steps !== lastSteps) {
				lastSteps = steps;
				ctx.emit('steps', steps);
			}
			const runFlag = running ? 1 : 0;
			if (runFlag !== lastRunning) {
				lastRunning = runFlag;
				ctx.emit('running', runFlag);
			}
		},
		draw(pass) {
			fieldB.draw(pass);
			field.draw(pass);
			frame.draw(pass);
			dots.draw(pass);
			diagram?.draw(pass);
			chart.draw(pass);
		},
		command(id) {
			if (id === 'run') {
				if (blewUp) return;
				running = !running;
				acc = 0;
				if (running) diagram?.clearPulses();
			} else if (id === 'step') {
				running = false;
				doStep();
				diagram?.pulse('forward', { layerDuration: 0.3, onDone: () => diagram?.pulse('backward', { layerDuration: 0.3 }) });
			} else if (id === 'reset') {
				reset(true);
			}
		},
		highlight(t) {
			hl = t;
		},
		dispose() {
			field.destroy();
			fieldB.destroy();
			frame.destroy();
			dots.destroy();
			chart.destroy();
			diagram?.destroy();
			diagram = null;
		}
	};
	return scene;
};
