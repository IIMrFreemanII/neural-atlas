import { Camera2D } from '../../../gpu/camera.ts';
import { paperFill, tone } from '../../../gpu/colors.ts';
import { Field } from '../../../gpu/primitives/field.ts';
import { Lines } from '../../../gpu/primitives/lines.ts';
import { Points } from '../../../gpu/primitives/points.ts';
import type { Label, Scene, SceneContext, SceneFactory, Tone } from '../../../gpu/types.ts';
import type { Dataset } from '../../../ml/datasets.ts';
import { knnNeighbors, knnProb } from '../../../ml/knn.ts';
import { logisticProb, logisticStep, type Logistic } from '../../../ml/logistic.ts';
import { buildTree, countLeaves, treeProb, treeSplits, type SplitSegment, type TreeNode } from '../../../ml/tree.ts';
import { fitCamera, lessonInsets, type WorldRect } from '../../fit.ts';
import { makeBoundaryData, N_POINTS } from './data.ts';

const DATA_SEED = 11;
const FIELD_N = 80;
const EXTENT = 1.4; // the decision map covers inputs [-EXTENT, EXTENT] on both axes
const LR = 1.0;
const STEPS_PER_SECOND = 60;
const MAX_STEPS_PER_FRAME = 4;
const REDUCED_TICKS_PER_SECOND = 3; // reduced motion: the boundary updates this often while training
const PRETRAIN_STEPS = 500;
const MAX_K = 25;
const WORLD: WorldRect = { x0: -1.5, x1: 1.5, y0: -1.5, y1: 1.5 };

const clamp = (v: number, a: number, b: number) => Math.min(b, Math.max(a, v));

export const createDecisionBoundariesScene: SceneFactory = () => {
	let ctx: SceneContext;
	let camera: Camera2D;
	let field: Field; // class B (data ink) fading to paper at low probability, plus the boundary line
	let fieldNeg: Field; // class A (neg ink) fading to paper at high probability
	let frame: Lines;
	let splitLines: Lines;
	let queryLines: Lines;
	let dots: Points;

	// params
	let model = 0;
	let datasetId = 0;
	let k = 7;
	let depth = 3;
	let noise = 0.05;
	let trained = 1;
	let scenario = 1;
	let paramsSeen = false;
	let noiseUsed = 0.05;
	let hl: Tone | null = null;

	// state
	let data: Dataset = makeBoundaryData(0, 0.05, DATA_SEED);
	let lm: Logistic = { w: [0.6, -0.8], b: 0.5 };
	let tree: TreeNode | null = null;
	let splits: SplitSegment[] = [];
	let leaves = 1;
	let steps = 0;
	let running = false;
	let acc = 0;
	let accuracy = 0;
	let dirty = true;
	let splitsDirty = true;
	const wrong = new Uint8Array(N_POINTS);

	// hover query (k-NN only)
	let qActive = false;
	let qx = 0;
	let qy = 0;
	let qNeighbors: number[] = [];

	let lastAcc = '';
	let lastInfo = '';

	const mul = (t: Tone) => (hl === null || hl === t ? 1 : 0.3);

	function prob(x0: number, x1: number): number {
		if (model === 1) return knnProb(data, k, x0, x1);
		if (model === 2 && tree) return treeProb(tree, x0, x1);
		return logisticProb(lm, x0, x1);
	}

	function resetModel() {
		running = false;
		acc = 0;
		steps = 0;
		lm = { w: [0.6, -0.8], b: 0.5 };
		if (trained >= 0.5) {
			for (let i = 0; i < PRETRAIN_STEPS; i++) logisticStep(lm, data, LR);
			steps = PRETRAIN_STEPS;
		}
		dirty = true;
	}

	function rebuildTree() {
		tree = buildTree(data, depth);
		leaves = countLeaves(tree);
		splits = treeSplits(tree, [-EXTENT, -EXTENT, EXTENT, EXTENT]);
		splitsDirty = true;
	}

	function refresh() {
		dirty = false;
		if (model === 2) rebuildTree();
		const span = 2 * EXTENT;
		for (let r = 0; r < FIELD_N; r++) {
			const y = -EXTENT + ((r + 0.5) / FIELD_N) * span;
			for (let c = 0; c < FIELD_N; c++) {
				const x = -EXTENT + ((c + 0.5) / FIELD_N) * span;
				field.data[r * FIELD_N + c] = prob(x, y);
			}
		}
		let correct = 0;
		for (let i = 0; i < N_POINTS; i++) {
			const p = prob(data.x[2 * i], data.x[2 * i + 1]);
			const ok = (p >= 0.5 ? 1 : 0) === data.y[i];
			wrong[i] = ok ? 0 : 1;
			if (ok) correct++;
		}
		accuracy = correct / N_POINTS;
		fieldNeg.data.set(field.data);
		fieldNeg.upload();
		field.upload();
		if (model === 1 && qActive) qNeighbors = knnNeighbors(data, k, qx, qy);
		else qNeighbors = [];
	}

	function loadData() {
		data = makeBoundaryData(datasetId, noise, DATA_SEED);
		noiseUsed = noise;
		dirty = true;
	}

	function drawFrame() {
		const c = tone('muted', 1, 0.6);
		const e = EXTENT;
		frame.count = frame.setPolyline(
			0,
			[
				[-e, -e],
				[e, -e],
				[e, e],
				[-e, e],
				[-e, -e]
			],
			{ width: 1, color: c }
		);
		frame.upload();
	}

	function drawSplits() {
		splitsDirty = false;
		let n = 0;
		if (model === 2) {
			const m = mul('text');
			for (const s of splits) {
				const a = 0.65 - 0.07 * s.depth;
				const color = tone('text', 1, clamp(a, 0.25, 0.65) * m);
				const width = s.depth === 0 ? 1.8 : s.depth < 3 ? 1.3 : 1;
				const pa: [number, number] = s.axis === 0 ? [s.threshold, s.lo] : [s.lo, s.threshold];
				const pb: [number, number] = s.axis === 0 ? [s.threshold, s.hi] : [s.hi, s.threshold];
				splitLines.set(n++, { a: pa, b: pb, width, color });
			}
		}
		splitLines.count = n;
		splitLines.upload();
	}

	function draw(labels: Label[]) {
		const mData = mul('data');
		const mNeg = mul('neg');
		const mGrad = mul('grad');
		let n = 0;
		for (let pass = 0; pass < 3; pass++) {
			for (let i = 0; i < N_POINTS; i++) {
				const cls = data.y[i] >= 0.5;
				const t: Tone = cls ? 'data' : 'neg';
				const m = cls ? mData : mNeg;
				const x = data.x[2 * i];
				const y = data.x[2 * i + 1];
				if (pass === 0) dots.set(n++, { x, y, radius: 5.2, color: paperFill(m) });
				else if (pass === 1) dots.set(n++, { x, y, radius: 3.5, color: tone(t, m) });
				else if (wrong[i]) dots.set(n++, { x, y, radius: 6.5, ring: 1.4, color: tone('grad', mGrad) });
			}
		}

		let ql = 0;
		if (model === 1 && qActive && qNeighbors.length > 0) {
			let ones = 0;
			for (const i of qNeighbors) {
				const cls = data.y[i] >= 0.5;
				if (cls) ones++;
				const t: Tone = cls ? 'data' : 'neg';
				const m = cls ? mData : mNeg;
				queryLines.set(ql++, {
					a: [qx, qy],
					b: [data.x[2 * i], data.x[2 * i + 1]],
					width: 1,
					color: tone(t, 0.9, 0.7 * m)
				});
				dots.set(n++, { x: data.x[2 * i], y: data.x[2 * i + 1], radius: 6, ring: 1.3, color: tone(t, m) });
			}
			dots.set(n++, { x: qx, y: qy, radius: 8, color: paperFill(0.9) });
			dots.set(n++, { x: qx, y: qy, radius: 8, ring: 1.5, color: tone('text') });
			const total = qNeighbors.length;
			const aWins = ones * 2 > total;
			const [px, py] = ctx.view.project(qx, qy);
			const flip = px > ctx.view.width - 170;
			labels.push({
				id: 'knn-vote',
				x: flip ? px - 18 : px + 18,
				y: py - 30,
				text: `${aWins ? ones : total - ones} of ${total} are ${aWins ? 'A' : 'B'}`,
				tone: aWins ? 'data' : 'neg',
				mono: true,
				align: flip ? 'right' : 'left'
			});
		}
		queryLines.count = ql;
		queryLines.upload();
		dots.count = n;
		dots.upload();
	}

	const scene: Scene = {
		init(c) {
			ctx = c;
			camera = new Camera2D();
			field = new Field(c, { cols: FIELD_N, rows: FIELD_N });
			field.rect = [-EXTENT, -EXTENT, EXTENT, EXTENT];
			field.iso = 0.5;
			field.isoWidth = 2;
			fieldNeg = new Field(c, { cols: FIELD_N, rows: FIELD_N });
			fieldNeg.rect = field.rect;
			fieldNeg.iso = -1;
			frame = new Lines(c, { capacity: 8, blend: 'alpha' });
			splitLines = new Lines(c, { capacity: 300, blend: 'alpha' });
			queryLines = new Lines(c, { capacity: MAX_K + 4, blend: 'alpha' });
			dots = new Points(c, { capacity: N_POINTS * 3 + MAX_K * 2 + 4, blend: 'alpha' });
			drawFrame();
		},
		setParams(p) {
			const prev = { model, datasetId, k, depth, scenario };
			model = Math.round(p.model ?? model);
			datasetId = Math.round(p.dataset ?? datasetId);
			const kr = clamp(Math.round(p.k ?? k), 1, MAX_K);
			k = kr % 2 === 0 ? kr + 1 : kr;
			depth = clamp(Math.round(p.depth ?? depth), 0, 8);
			trained = p.trained ?? trained;
			scenario = Math.round(p.scenario ?? scenario);
			noise = p.noise ?? noise;
			const noiseChanged = Math.abs(noise - noiseUsed) > 0.001;
			const dataChanged = prev.datasetId !== datasetId || noiseChanged || !paramsSeen;
			if (dataChanged) loadData();
			if (!paramsSeen || prev.scenario !== scenario || prev.model !== model || dataChanged) {
				resetModel();
				if (prev.model !== model) {
					qActive = false;
					qNeighbors = [];
				}
			}
			if (prev.k !== k || prev.depth !== depth) dirty = true;
			if (model !== 2 || prev.model !== model) splitsDirty = true;
			paramsSeen = true;
		},
		update(fr) {
			const v = ctx.view;
			fitCamera(camera, v, WORLD, lessonInsets(v));
			v.setViewProj(camera.matrix(v.aspect));

			if (running && model === 0 && fr.dt > 0) {
				// Reduced motion: the same training speed, shown as 3 jumps a second of 20 steps each.
				const reduced = ctx.reducedMotion;
				acc += fr.dt * (reduced ? REDUCED_TICKS_PER_SECOND : STEPS_PER_SECOND);
				let n = Math.floor(acc);
				acc -= n;
				if (n > MAX_STEPS_PER_FRAME) {
					n = MAX_STEPS_PER_FRAME;
					acc = 0;
				}
				if (reduced) n *= STEPS_PER_SECOND / REDUCED_TICKS_PER_SECOND;
				for (let i = 0; i < n; i++) {
					logisticStep(lm, data, LR);
					steps++;
				}
				if (n > 0) dirty = true;
			}
			if (dirty) refresh();
			if (splitsDirty || model === 2) {
				// splits are rebuilt on refresh; redraw when flagged or when highlight changes
				drawSplits();
			}

			// Class washes stay faint so the dots and the boundary carry the picture.
			// Two layers that each fade to transparent, so uncertain areas read as paper, not grey.
			const negC = tone('neg', 0.2 * mul('neg'));
			const dataC = tone('data', 0.2 * mul('data'));
			fieldNeg.colorLo = negC;
			fieldNeg.colorHi = [negC[0], negC[1], negC[2], 0];
			field.colorLo = [dataC[0], dataC[1], dataC[2], 0];
			field.colorHi = dataC;
			field.isoColor = tone('text', mul('text'));
			fieldNeg.upload();
			field.upload();

			const labels: Label[] = [];
			draw(labels);
			ctx.labels.set(labels);

			const accText = `${Math.round(accuracy * 100)}%`;
			if (accText !== lastAcc) {
				lastAcc = accText;
				ctx.emit('acc', accText);
			}
			const info =
				model === 0
					? `step ${steps}`
					: model === 1
						? `k = ${k}`
						: `depth ${depth} · ${leaves} ${leaves === 1 ? 'leaf' : 'leaves'}`;
			if (info !== lastInfo) {
				lastInfo = info;
				ctx.emit('info', info);
			}
		},
		draw(pass) {
			fieldNeg.draw(pass);
			field.draw(pass);
			frame.draw(pass);
			splitLines.draw(pass);
			queryLines.draw(pass);
			dots.draw(pass);
		},
		command(id) {
			if (model !== 0) return;
			if (id === 'run') {
				running = !running;
				acc = 0;
			} else if (id === 'reset') {
				resetModel();
			}
		},
		pointer(e) {
			if (model !== 1 || e.type === 'leave' || e.type === 'up') {
				if (e.type === 'leave' && qActive) {
					qActive = false;
					qNeighbors = [];
				}
				return;
			}
			const [x, y] = ctx.view.unproject(e.x, e.y);
			if (Math.abs(x) > EXTENT || Math.abs(y) > EXTENT) {
				qActive = false;
				qNeighbors = [];
				return;
			}
			qActive = true;
			qx = x;
			qy = y;
			qNeighbors = knnNeighbors(data, k, x, y);
		},
		highlight(t) {
			hl = t;
			splitsDirty = true;
		},
		dispose() {
			field.destroy();
			fieldNeg.destroy();
			frame.destroy();
			splitLines.destroy();
			queryLines.destroy();
			dots.destroy();
		}
	};
	return scene;
};
