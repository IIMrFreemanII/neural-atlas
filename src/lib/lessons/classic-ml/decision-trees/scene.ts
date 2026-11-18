import { Camera2D } from '../../../gpu/camera.ts';
import { mixColor, tone, VOID } from '../../../gpu/colors.ts';
import { Field } from '../../../gpu/primitives/field.ts';
import { Lines, type SegmentStyle } from '../../../gpu/primitives/lines.ts';
import { Points, type PointStyle } from '../../../gpu/primitives/points.ts';
import { Quads, type QuadStyle } from '../../../gpu/primitives/quads.ts';
import type { Label, Scene, SceneContext, SceneFactory, ScenePointer, Tone } from '../../../gpu/types.ts';
import type { Dataset } from '../../../ml/datasets.ts';
import {
	accuracy,
	bestSplit,
	buildForest,
	buildTree,
	forestProb,
	layoutTree,
	splitCurve,
	splitImpurity,
	treeBoxes,
	treePath,
	treeProb,
	treeSplits,
	type CurvePoint,
	type LaidOutNode,
	type LeafBox,
	type SplitSegment,
	type TreeNode
} from '../../../ml/tree.ts';
import { fitCamera, type Insets } from '../../fit.ts';
import { BOUNDS, FOREST_SEED, MAX_DEPTH, TEST, depthCurve, trainingData, type DepthPoint } from './data.ts';

// World layout, y up. Everything is in world units; node boxes and dots are sized in CSS px.
// The plane (data units mapped to a square) sits above the tree diagram. When a chart opens, the plane
// shrinks and moves to the left, the composition widens, and the chart takes the column to the right.
const PLANE = 3.3; // plane side with no chart
const PLANE_CHART = 3.0; // plane side with a chart open
const COMP = 5.4; // composition width, the same with or without a chart
const CHART_GAP = 0.45;
const PLANE_B = -3.3; // plane bottom edge
const TREE_TOP = -3.75; // tree diagram panel
const TREE_BOT = -5.05;
const VIEW_TOP = 0.02;
const VIEW_BOT = -5.15;
const INSETS: Insets = { top: 64, right: 24, bottom: 110, left: 24 };

const FIELD_N = 96;
const N_FOREST = 50;
const N_SAMPLES = 9;
const ROW = 10; // forest silhouettes per row
const MAX_LABELLED_LEAVES = 8;
const MAX_LABELLED_DEPTH = 3;
const IMPURITY_MAX = 0.5;
const ACC_MIN = 0.5;
const CELL_PAD_X = 28; // px
const PANEL_PAD_TOP = 14;
const PANEL_PAD_BOT = 24;

type Rgba = [number, number, number, number];

const clamp = (v: number, a: number, b: number) => Math.min(b, Math.max(a, v));
const lerp = (a: number, b: number, t: number) => a + (b - a) * t;
const fmt = (v: number) => {
	const s = v.toFixed(2);
	return s === '-0.00' ? '0.00' : s.replace('-', '−');
};
const percent = (v: number) => `${Math.round(v * 100)}%`;
const leafColor = (prob: number, alpha: number, intensity = 1): Rgba =>
	mixColor(tone('neg', intensity, alpha), tone('data', intensity, alpha), prob);
const classCounts = (n: TreeNode): [number, number] => {
	const a = Math.round(n.prob * n.n);
	return [a, n.n - a];
};
const question = (n: TreeNode) => `x${n.axis === 0 ? '₁' : '₂'} < ${fmt(n.threshold ?? 0)}?`;

/** A tree of one question: the root cut at `t` on `axis`, with two leaves. Used for the "best first question" page. */
function makeStump(d: Dataset, axis: 0 | 1, t: number): TreeNode {
	let nl = 0;
	let leftOnes = 0;
	let ones = 0;
	for (let i = 0; i < d.n; i++) {
		const one = d.y[i] > 0.5 ? 1 : 0;
		ones += one;
		if (d.x[2 * i + axis] < t) {
			nl++;
			leftOnes += one;
		}
	}
	const leaf = (n: number, o: number): TreeNode => ({ depth: 1, n, prob: n === 0 ? 0.5 : o / n });
	return {
		depth: 0,
		n: d.n,
		prob: d.n === 0 ? 0.5 : ones / d.n,
		axis,
		threshold: t,
		left: leaf(nl, leftOnes),
		right: leaf(d.n - nl, ones - leftOnes)
	};
}

interface Batch<S> {
	add(s: S): void;
}
const batch = <S>(p: { capacity: number; count: number; set(i: number, s: S): void }): Batch<S> => ({
	add(s) {
		if (p.count < p.capacity) p.set(p.count++, s);
	}
});

interface Geo {
	comp: number;
	ps: number;
	pl: number;
	pb: number;
	pr: number;
	pt: number;
	cl: number;
	cr: number;
	cb: number;
	ct: number;
	tl: number;
	tr: number;
}

function geometry(s: number): Geo {
	const comp = COMP;
	const ps = lerp(PLANE, PLANE_CHART, s);
	const pl = lerp(-ps / 2, -comp / 2, s);
	const pr = pl + ps;
	return {
		comp,
		ps,
		pl,
		pb: PLANE_B,
		pr,
		pt: PLANE_B + ps,
		cl: pr + CHART_GAP,
		cr: comp / 2 - 0.05,
		cb: PLANE_B,
		ct: PLANE_B + ps,
		tl: -comp / 2,
		tr: comp / 2
	};
}

export const createDecisionTreesScene: SceneFactory = () => {
	let ctx: SceneContext;
	let camera: Camera2D;
	let field: Field;
	let quads: Quads;
	let ghostLines: Lines;
	let planeLines: Lines;
	let treeLines: Lines;
	let chartLines: Lines;
	let dots: Points;

	// params
	let depth = 2;
	let split = 0;
	let axis: 0 | 1 = 0;
	let cut = 0;
	let probeOn = 0;
	let showTest = 0;
	let chart = 0;
	let sample = 0;
	let ghosts = 0;
	let forest = 0;
	let nTrees = 25;
	let scenario = 1;
	let paramsSeen = false;
	let hl: Tone | null = null;

	// model
	let train: Dataset = trainingData(0);
	let tree: TreeNode = { depth: 0, n: 0, prob: 0.5 };
	let boxes: LeafBox[] = [];
	let splits: SplitSegment[] = [];
	let lay: LaidOutNode[] = [];
	let leafIdx: number[] = [];
	let nodeIdx = new Map<TreeNode, number>();
	let ghostSplits: SplitSegment[][] = [];
	let forestSample = -1;
	let forestTrees: TreeNode[] = [];
	let forestLay: LaidOutNode[][] = [];
	let curve: CurvePoint[] = [];
	let dCurve: DepthPoint[] = [];
	let accTrain = 0;
	let accTest = 0;
	let impurity = 0;
	let structKey = '';
	let probe: [number, number] | null = null;
	let hover = -1;
	let dragging = false;
	let answer = '—';

	// what is on screen
	let s = 0; // eased chart openness, 0..1
	let kind: 'split' | 'depth' = 'split'; // which chart is (or was last) shown
	let fade = 1; // fade-in of a freshly built tree
	let snap = true;
	let geo: Geo = geometry(0);
	let pxu = 90; // CSS px per world unit
	let rev = 0;
	let builtKey = '';
	let labels: Label[] = [];
	let leafHits: { x: number; y: number; r: number }[] = [];

	const last = { acc: '', test: '' };

	const mul = (t: Tone) => (hl === null || hl === t ? 1 : 0.3);
	const u = (px: number) => px / pxu;

	// caches
	const trainCache = new Map<number, Dataset>();
	const getTrain = (k: number) => {
		let d = trainCache.get(k);
		if (!d) {
			d = trainingData(k);
			trainCache.set(k, d);
		}
		return d;
	};
	const curveCache = new Map<string, CurvePoint[]>();
	const depthCache = new Map<number, DepthPoint[]>();
	const ghostCache = new Map<string, SplitSegment[]>();

	function ghostFor(k: number, d: number): SplitSegment[] {
		const key = `${k}|${d}`;
		let g = ghostCache.get(key);
		if (!g) {
			g = treeSplits(buildTree(getTrain(k), d), BOUNDS);
			ghostCache.set(key, g);
		}
		return g;
	}

	function rasterForest(ft: TreeNode[]) {
		const [x0, y0, x1, y1] = BOUNDS;
		for (let r = 0; r < FIELD_N; r++) {
			const y = y0 + ((r + 0.5) / FIELD_N) * (y1 - y0);
			for (let c = 0; c < FIELD_N; c++) {
				const x = x0 + ((c + 0.5) / FIELD_N) * (x1 - x0);
				field.data[r * FIELD_N + c] = forestProb(ft, x, y);
			}
		}
	}

	/** A point inside a class-A leaf: the mean of the dots of the leaf with the most class-A dots among the nearly pure ones. */
	function startProbe(): [number, number] {
		let best: LeafBox | null = null;
		const dotsA = (b: LeafBox) => Math.round(b.leaf.prob * b.leaf.n);
		for (const b of boxes) if (b.leaf.prob >= 0.9 && (best === null || dotsA(b) > dotsA(best))) best = b;
		if (!best) for (const b of boxes) if (b.leaf.prob > 0.5 && (best === null || dotsA(b) > dotsA(best))) best = b;
		if (!best) return [0, 0];
		const [x0, y0, x1, y1] = best.box;
		let sx = 0;
		let sy = 0;
		let n = 0;
		for (let i = 0; i < train.n; i++) {
			const x = train.x[2 * i];
			const y = train.x[2 * i + 1];
			if (x >= x0 && x < x1 && y >= y0 && y < y1) {
				sx += x;
				sy += y;
				n++;
			}
		}
		return n > 0 ? [sx / n, sy / n] : [(x0 + x1) / 2, (y0 + y1) / 2];
	}

	function computeModel(scenarioChanged: boolean) {
		train = getTrain(sample);
		const eff = split >= 0.5 ? 0 : depth;
		const key = `${forest}|${split}|${eff}|${sample}|${nTrees}`;
		const structChanged = key !== structKey;
		structKey = key;

		if (forest >= 0.5) {
			if (forestSample !== sample) {
				forestTrees = buildForest(train, N_FOREST, MAX_DEPTH, FOREST_SEED);
				forestLay = forestTrees.map(layoutTree);
				forestSample = sample;
			}
			const ft = forestTrees.slice(0, nTrees);
			if (structChanged) rasterForest(ft);
			const p = (a: number, b: number) => forestProb(ft, a, b);
			accTrain = accuracy(train, p);
			accTest = accuracy(TEST, p);
		} else {
			tree = split >= 0.5 ? makeStump(train, axis, cut) : buildTree(train, eff);
			boxes = treeBoxes(tree, BOUNDS);
			splits = treeSplits(tree, BOUNDS);
			lay = layoutTree(tree);
			leafIdx = [];
			nodeIdx = new Map();
			lay.forEach((n, i) => {
				nodeIdx.set(n.node, i);
				if (!n.node.left) leafIdx.push(i);
			});
			const p = (a: number, b: number) => treeProb(tree, a, b);
			accTrain = accuracy(train, p);
			accTest = accuracy(TEST, p);
			if (structChanged) hover = -1;
		}

		impurity = splitImpurity(train, axis, cut);
		const ck = `${sample}|${axis}`;
		let c = curveCache.get(ck);
		if (!c) {
			c = splitCurve(train, axis);
			curveCache.set(ck, c);
		}
		curve = c;
		let dc = depthCache.get(sample);
		if (!dc) {
			dc = depthCurve(sample);
			depthCache.set(sample, dc);
		}
		dCurve = dc;

		ghostSplits = [];
		if (ghosts >= 0.5 && forest < 0.5) {
			for (let k = 0; k < N_SAMPLES; k++) if (k !== sample) ghostSplits.push(ghostFor(k, eff));
		}

		if (structChanged && !ctx.reducedMotion) fade = 0.25;
		if (probeOn >= 0.5 && forest < 0.5 && (scenarioChanged || probe === null)) probe = startProbe();
	}

	// ---------------------------------------------------------------- drawing

	function build() {
		const v = ctx.view;
		const g = geo;
		const mT = mul('text');
		const mD = mul('data');
		const mN = mul('neg');
		const mG = mul('grad');
		const mO = mul('ok');
		const isForest = forest >= 0.5;
		const isSplit = split >= 0.5;
		const eff = isSplit ? 0 : depth;

		quads.count = 0;
		ghostLines.count = 0;
		planeLines.count = 0;
		treeLines.count = 0;
		chartLines.count = 0;
		dots.count = 0;
		const Q = batch<QuadStyle>(quads);
		const GL = batch<SegmentStyle>(ghostLines);
		const PL = batch<SegmentStyle>(planeLines);
		const TL = batch<SegmentStyle>(treeLines);
		const CL = batch<SegmentStyle>(chartLines);
		const D = batch<PointStyle>(dots);
		const labs: Label[] = [];

		const seg = (b: Batch<SegmentStyle>, ax: number, ay: number, bx: number, by: number, width: number, color: Rgba) =>
			b.add({ a: [ax, ay], b: [bx, by], width, color });
		const poly = (b: Batch<SegmentStyle>, pts: [number, number][], width: number, color: Rgba) => {
			for (let i = 0; i + 1 < pts.length; i++) seg(b, pts[i][0], pts[i][1], pts[i + 1][0], pts[i + 1][1], width, color);
		};
		const rect = (b: Batch<SegmentStyle>, x0: number, y0: number, x1: number, y1: number, width: number, color: Rgba) =>
			poly(b, [[x0, y0], [x1, y0], [x1, y1], [x0, y1], [x0, y0]], width, color);
		const label = (id: string, wx: number, wy: number, text: string, o: Partial<Label> = {}) => {
			const [x, y] = v.project(wx, wy);
			labs.push({ id, x: x + (o.x ?? 0), y: y + (o.y ?? 0), text, tone: o.tone ?? 'muted', mono: true, size: o.size ?? 10, align: o.align, opacity: o.opacity });
		};

		// ---- the plane
		const toW = (x: number, y: number): [number, number] => [
			g.pl + ((x - BOUNDS[0]) / (BOUNDS[2] - BOUNDS[0])) * g.ps,
			g.pb + ((y - BOUNDS[1]) / (BOUNDS[3] - BOUNDS[1])) * g.ps
		];

		let path: number[] = [];
		let probeLeaf = -1;
		const probeSet = new Set<number>();
		answer = '—';
		if (probeOn >= 0.5 && !isForest && probe) {
			const nodes = treePath(tree, probe[0], probe[1]);
			path = nodes.map((n) => nodeIdx.get(n) ?? 0);
			for (const i of path) probeSet.add(i);
			const leaf = nodes[nodes.length - 1];
			probeLeaf = boxes.findIndex((b) => b.leaf === leaf);
			const [a, b] = classCounts(leaf);
			answer = leaf.prob >= 0.5 ? `A · ${a} of ${leaf.n}` : `B · ${b} of ${leaf.n}`;
		}
		const hoverSet = new Set<number>();
		if (hover >= 0 && hover < leafIdx.length) {
			for (let i = leafIdx[hover]; i >= 0; i = lay[i].parent) hoverSet.add(i);
		}

		if (isForest) {
			field.rect = [g.pl, g.pb, g.pr, g.pt];
			field.colorLo = tone('neg', 0.3, 0.55 * mN);
			field.colorHi = tone('data', 0.3, 0.55 * mD);
			field.isoColor = tone('text', 1, 0.9 * mT);
			field.upload();
		} else {
			for (const b of boxes) {
				const [x0, y0] = toW(b.box[0], b.box[1]);
				const [x1, y1] = toW(b.box[2], b.box[3]);
				const hot = b.index === hover;
				const a = (hot ? 0.36 : 0.17) * fade * mul(b.leaf.prob >= 0.5 ? 'data' : 'neg');
				Q.add({ x: (x0 + x1) / 2, y: (y0 + y1) / 2, w: x1 - x0, h: y1 - y0, color: leafColor(b.leaf.prob, a) });
				if (hot) rect(PL, x0, y0, x1, y1, 1.5, tone('text', 1, 0.9 * mT));
				if (b.index === probeLeaf) rect(PL, x0, y0, x1, y1, 2, tone('grad', 1, 0.95 * mG));
			}
		}
		rect(PL, g.pl, g.pb, g.pr, g.pt, 1, tone('muted', 1, 0.55));

		// ghosts: the trees of the other samples, behind the current one
		for (const gs of ghostSplits) {
			for (const sp of gs) {
				const [ax, ay] = sp.axis === 0 ? toW(sp.threshold, sp.lo) : toW(sp.lo, sp.threshold);
				const [bx, by] = sp.axis === 0 ? toW(sp.threshold, sp.hi) : toW(sp.hi, sp.threshold);
				seg(GL, ax, ay, bx, by, 1, tone('muted', 1.3, 0.3));
			}
		}

		if (!isForest) {
			const lineTone: Tone = isSplit ? 'grad' : 'text';
			const mL = isSplit ? mG : mT;
			for (const sp of splits) {
				const a = clamp(0.7 - 0.07 * sp.depth, 0.28, 0.7) * mL * (isSplit ? 1 : fade);
				const width = sp.depth === 0 ? (isSplit ? 2.5 : 1.8) : sp.depth < 3 ? 1.3 : 1;
				const [ax, ay] = sp.axis === 0 ? toW(sp.threshold, sp.lo) : toW(sp.lo, sp.threshold);
				const [bx, by] = sp.axis === 0 ? toW(sp.threshold, sp.hi) : toW(sp.hi, sp.threshold);
				seg(PL, ax, ay, bx, by, width, tone(lineTone, 1, a));
			}
			if (!isSplit && eff <= MAX_LABELLED_DEPTH) {
				splits.forEach((sp, i) => {
					const txt = `x${sp.axis === 0 ? '₁' : '₂'} < ${fmt(sp.threshold)}?`;
					if (sp.axis === 0) {
						const [wx, wy] = toW(sp.threshold, sp.lo + 0.82 * (sp.hi - sp.lo));
						const [sx] = v.project(wx, wy);
						const flip = sx + 74 > v.project(g.pr, 0)[0];
						label(`q${i}`, wx, wy, txt, { tone: 'text', size: 12, align: flip ? 'right' : 'left', x: flip ? -5 : 5, opacity: mT * fade });
					} else {
						const [wx, wy] = toW((sp.lo + sp.hi) / 2, sp.threshold);
						label(`q${i}`, wx, wy, txt, { tone: 'text', size: 12, y: -9, opacity: mT * fade });
					}
				});
			}
		}

		// ---- points: test rings, training dots, probe
		if (showTest >= 0.5) {
			for (let i = 0; i < TEST.n; i++) {
				const [x, y] = toW(TEST.x[2 * i], TEST.x[2 * i + 1]);
				D.add({ x, y, radius: 3, ring: 1, color: tone('ok', 1.1, hl === 'ok' ? 0.95 : 0.45 * mO) });
			}
		}
		for (let pass = 0; pass < 2; pass++) {
			for (let i = 0; i < train.n; i++) {
				const cls = train.y[i] > 0.5;
				const [x, y] = toW(train.x[2 * i], train.x[2 * i + 1]);
				const t: Tone = cls ? 'data' : 'neg';
				const m = cls ? mD : mN;
				if (pass === 0) D.add({ x, y, radius: 3.5, color: tone(t, 1.3, 0.95 * m) });
				else D.add({ x, y, radius: 3.5, ring: 1, color: tone(t, 0.3, 0.9 * m) });
			}
		}
		if (probeOn >= 0.5 && !isForest && probe) {
			const [x, y] = toW(probe[0], probe[1]);
			D.add({ x, y, radius: 10, ring: 2, color: tone('grad', 1.1, 0.95 * mG) });
			D.add({ x, y, radius: 4, color: tone('grad', 1.2, mG) });
			const flip = v.project(x, y)[0] + 110 > v.project(g.pr, 0)[0];
			label('probe', x, y, answer, { tone: 'grad', size: 11, align: flip ? 'right' : 'left', x: flip ? -16 : 16, opacity: mG });
		}

		// ---- the tree diagram (or the forest's silhouettes) under the plane
		const padX = u(CELL_PAD_X);
		leafHits = [];
		if (!isForest) {
			const nL = leafIdx.length;
			const many = nL > MAX_LABELLED_LEAVES;
			const dMax = lay.reduce((m, n) => Math.max(m, n.depth), 0);
			const usableW = g.tr - g.tl - 2 * padX;
			const usableH = TREE_TOP - TREE_BOT - u(PANEL_PAD_TOP) - u(PANEL_PAD_BOT);
			const ls = dMax > 0 ? Math.min(u(many ? 30 : 46), usableH / dMax) : 0;
			const pitchPx = (usableW * pxu) / Math.max(1, nL);
			const leafPx = many ? clamp(Math.min(ls * pxu * 0.85, pitchPx * 0.85), 3, 8) : 14;
			const boxW = 76;
			const boxH = 18;
			const pos = lay.map((n): [number, number] => [g.tl + padX + n.x * usableW, TREE_TOP - u(PANEL_PAD_TOP) - n.depth * ls]);
			const isLeaf = (i: number) => !lay[i].node.left;
			const top = (i: number) => (isLeaf(i) ? u(leafPx / 2) : many ? 0 : u(boxH / 2));

			// edges
			for (let i = 1; i < lay.length; i++) {
				const p = lay[i].parent;
				const a = hoverSet.has(i) ? 0.95 : 0.4;
				seg(TL, pos[p][0], pos[p][1] - (many ? 0 : u(boxH / 2)), pos[i][0], pos[i][1] + top(i), hoverSet.has(i) ? 2.2 : 1.2, tone('text', 1, a * mT));
			}
			for (const i of path) {
				if (i === 0) continue;
				const p = lay[i].parent;
				seg(TL, pos[p][0], pos[p][1] - (many ? 0 : u(boxH / 2)), pos[i][0], pos[i][1] + top(i), 2.6, tone('grad', 1.1, 0.95 * mG));
			}

			// nodes
			lay.forEach((n, i) => {
				const [x, y] = pos[i];
				const onPath = probeSet.has(i);
				if (isLeaf(i)) {
					const k = leafIdx.indexOf(i);
					const hot = k === hover;
					const mc = mul(n.node.prob >= 0.5 ? 'data' : 'neg');
					const side = u(leafPx);
					if (leafPx >= 8 || hot || onPath) {
						const o = u(hot || onPath ? 3.5 : 2);
						const oc = onPath ? tone('grad', 1.1, 0.95 * mG) : tone('text', 1, hot ? 0.95 : 0.35);
						Q.add({ x, y, w: side + o, h: side + o, radius: 2, color: oc });
					}
					Q.add({ x, y, w: side, h: side, radius: 1.5, color: leafColor(n.node.prob, 0.95 * mc * fade, 1.1) });
					leafHits.push({ x, y, r: Math.max(8, leafPx / 2 + 4) });
					if (!many) {
						const [a, b] = classCounts(n.node);
						label(`n${i}`, x, y - side / 2, `${a} · ${b}`, { tone: 'text', y: 11, opacity: (hot ? 1 : 0.75) * mT });
					}
				} else if (many) {
					D.add({ x, y, radius: onPath ? 3.6 : 2.4, color: onPath ? tone('grad', 1.1, 0.95 * mG) : tone('text', 1, 0.6 * mT) });
				} else {
					const w = u(boxW);
					const h = u(boxH);
					const o = u(onPath ? 3.5 : 2);
					const oc = onPath ? tone('grad', 1.1, 0.95 * mG) : tone('text', 1, 0.5 * mT);
					Q.add({ x, y, w: w + o, h: h + o, radius: 6, color: oc });
					Q.add({ x, y, w, h, radius: 5, color: [VOID[0], VOID[1], VOID[2], 0.97] });
					label(`n${i}`, x, y, question(n.node), { tone: 'text', opacity: mT * fade });
				}
			});
			if (!many && lay.length > 1) {
				const kids = lay.map((n, i) => (n.parent === 0 ? i : -1)).filter((i) => i > 0);
				kids.forEach((i, j) => {
					const mx = (pos[0][0] + pos[i][0]) / 2;
					const my = (pos[0][1] - u(boxH / 2) + pos[i][1] + top(i)) / 2;
					label(`e${j}`, mx, my, j === 0 ? 'yes' : 'no', { tone: 'muted', x: j === 0 ? -14 : 14, opacity: 0.9 });
				});
			}
		} else {
			const rows = Math.ceil(nTrees / ROW);
			const cellW = (g.tr - g.tl - 2 * padX) / ROW;
			const cellH = (TREE_TOP - TREE_BOT - u(8)) / rows;
			for (let k = 0; k < nTrees; k++) {
				const row = Math.floor(k / ROW);
				const col = k % ROW;
				const inRow = Math.min(ROW, nTrees - row * ROW);
				const x0 = g.tl + padX + ((ROW - inRow) * cellW) / 2 + col * cellW;
				const yTop = TREE_TOP - u(5) - row * cellH;
				const mx = u(5);
				const my = u(4);
				const ln = forestLay[k];
				const dm = ln.reduce((m, n) => Math.max(m, n.depth), 0);
				const at = (n: LaidOutNode): [number, number] => [
					x0 + mx + n.x * (cellW - 2 * mx),
					yTop - my - (dm > 0 ? (n.depth / dm) * (cellH - 2 * my) : 0)
				];
				for (let i = 1; i < ln.length; i++) {
					const a = at(ln[ln[i].parent]);
					const b = at(ln[i]);
					seg(TL, a[0], a[1], b[0], b[1], 1, tone('text', 1, 0.5 * mT));
				}
			}
		}

		// ---- the chart column
		if (s > 0.02) {
			const a = s;
			const cw = g.cr - g.cl;
			const ch = g.ct - g.cb;
			const axisC = tone('muted', 1, 0.7 * a);
			const gridC = tone('muted', 1, 0.2 * a);
			seg(CL, g.cl, g.cb, g.cl, g.ct, 1, axisC);
			seg(CL, g.cl, g.cb, g.cr, g.cb, 1, axisC);
			const tickY = (yy: number, text: string, id: string) => {
				seg(CL, g.cl, yy, g.cr, yy, 1, gridC);
				label(id, g.cl, yy, text, { x: -6, align: 'right', opacity: a });
			};
			const tickX = (xx: number, text: string, id: string) =>
				label(id, xx, g.cb, text, { y: 12, opacity: a });
			if (kind === 'split') {
				const mc = mG;
				const cx = (t: number) => g.cl + ((t - BOUNDS[0]) / (BOUNDS[2] - BOUNDS[0])) * cw;
				const cy = (i: number) => g.cb + clamp(i / IMPURITY_MAX, 0, 1) * ch;
				tickY(cy(0), '0', 'yt0');
				tickY(cy(0.25), '0.25', 'yt1');
				tickY(cy(0.5), '0.5', 'yt2');
				for (const t of [-1, 0, 1]) tickX(cx(t), fmt(t).replace('.00', ''), `xt${t}`);
				poly(CL, curve.map((p): [number, number] => [cx(p.threshold), cy(p.impurity)]), 2.2, tone('grad', 1.2, 0.95 * a * mc));
				const mx = cx(clamp(cut, BOUNDS[0], BOUNDS[2]));
				const my = cy(impurity);
				seg(CL, mx, g.cb, mx, my, 1, tone('grad', 1, 0.5 * a * mc));
				D.add({ x: mx, y: my, radius: 8, ring: 1.6, color: tone('grad', 1.1, 0.9 * a * mc) });
				D.add({ x: mx, y: my, radius: 3.6, color: tone('grad', 1.3, a * mc) });
				// Above the plot, over the marker, so the value never meets the curve or the ring.
				// Kept clear of the 'impurity' caption at the top-left corner and inside the right edge.
				const impX = v.project(mx, g.ct)[0];
				const impMin = v.project(g.cl, g.ct)[0] + 76;
				const impMax = v.project(g.cr, g.ct)[0] - 16;
				label('imp', mx, g.ct, impurity.toFixed(3), { tone: 'grad', size: 11, y: -22, x: clamp(impX, impMin, impMax) - impX, opacity: a * mc });
				label('cx', (g.cl + g.cr) / 2, g.cb, `cut position along x${axis + 1}`, { y: 32, opacity: a });
				label('cy', g.cl, g.ct, 'impurity', { y: -14, align: 'left', opacity: a });
			} else {
				const cx = (d: number) => g.cl + (d / MAX_DEPTH) * cw;
				const cy = (v2: number) => g.cb + clamp((v2 - ACC_MIN) / (1 - ACC_MIN), 0, 1) * ch;
				tickY(cy(0.5), '50%', 'yt0');
				tickY(cy(0.75), '75%', 'yt1');
				tickY(cy(1), '100%', 'yt2');
				for (let d = 0; d <= MAX_DEPTH; d += 2) tickX(cx(d), String(d), `xt${d}`);
				const trainPts = dCurve.map((p): [number, number] => [cx(p.depth), cy(p.train)]);
				const testPts = dCurve.map((p): [number, number] => [cx(p.depth), cy(p.test)]);
				poly(CL, testPts, 2.2, tone('ok', 1.3, 0.95 * a * mO));
				poly(CL, trainPts, 2.2, tone('data', 1.3, 0.95 * a * mD));
				const dm = clamp(eff, 0, MAX_DEPTH);
				seg(CL, cx(dm), g.cb, cx(dm), g.ct, 1.5, tone('text', 1, 0.45 * a * mT));
				for (let d = 0; d <= MAX_DEPTH; d++) {
					D.add({ x: testPts[d][0], y: testPts[d][1], radius: d === dm ? 4.5 : 2.4, color: tone('ok', 1.3, 0.95 * a * mO) });
					D.add({ x: trainPts[d][0], y: trainPts[d][1], radius: d === dm ? 4.5 : 2.4, color: tone('data', 1.3, 0.95 * a * mD) });
				}
				label('cx', (g.cl + g.cr) / 2, g.cb, 'tree depth', { y: 32, opacity: a });
				label('cy', g.cl, g.ct, 'accuracy', { y: -14, align: 'left', opacity: a });
			}
		}

		quads.upload();
		ghostLines.upload();
		planeLines.upload();
		treeLines.upload();
		chartLines.upload();
		dots.upload();
		labels = labs;
	}

	function emitReadouts() {
		const acc = percent(accTrain);
		if (acc !== last.acc) {
			last.acc = acc;
			ctx.emit('acc', acc);
		}
		const test = showTest >= 0.5 ? percent(accTest) : '—';
		if (test !== last.test) {
			last.test = test;
			ctx.emit('test', test);
		}
	}

	function planeData(wx: number, wy: number): [number, number] {
		const g = geo;
		return [
			BOUNDS[0] + ((wx - g.pl) / g.ps) * (BOUNDS[2] - BOUNDS[0]),
			BOUNDS[1] + ((wy - g.pb) / g.ps) * (BOUNDS[3] - BOUNDS[1])
		];
	}

	const scene: Scene = {
		init(c) {
			ctx = c;
			camera = new Camera2D();
			field = new Field(c, { cols: FIELD_N, rows: FIELD_N });
			field.iso = 0.5;
			field.isoWidth = 2;
			quads = new Quads(c, { capacity: 900 });
			ghostLines = new Lines(c, { capacity: 1000, blend: 'alpha' });
			planeLines = new Lines(c, { capacity: 400, blend: 'alpha' });
			treeLines = new Lines(c, { capacity: 4096, blend: 'alpha' });
			chartLines = new Lines(c, { capacity: 400, blend: 'alpha' });
			dots = new Points(c, { capacity: 1400, blend: 'alpha' });
		},
		setParams(p) {
			const prevScenario = scenario;
			depth = clamp(Math.round(p.depth ?? depth), 0, MAX_DEPTH);
			split = p.split ?? split;
			axis = Math.round(p.axis ?? axis) === 1 ? 1 : 0;
			cut = p.cut ?? cut;
			probeOn = p.probe ?? probeOn;
			showTest = p.test ?? showTest;
			chart = p.chart ?? chart;
			sample = clamp(Math.round(p.sample ?? sample), 0, N_SAMPLES - 1);
			ghosts = p.ghosts ?? ghosts;
			forest = p.forest ?? forest;
			nTrees = clamp(Math.round(p.trees ?? nTrees), 1, N_FOREST);
			scenario = Math.round(p.scenario ?? scenario);
			const changed = !paramsSeen || prevScenario !== scenario;
			if (changed) {
				hover = -1;
				dragging = false;
				structKey = '';
			}
			computeModel(changed);
			paramsSeen = true;
			rev++;
		},
		update(fr) {
			const v = ctx.view;
			const instant = ctx.reducedMotion || snap;
			snap = false;

			// chart column: opens for either chart, never both
			if (split >= 0.5) kind = 'split';
			else if (chart >= 0.5) kind = 'depth';
			const target = forest < 0.5 && (split >= 0.5 || chart >= 0.5) ? 1 : 0;
			if (s !== target) {
				const k = instant ? 1 : 1 - Math.exp(-fr.dt / 0.16);
				const nx = s + (target - s) * k;
				s = Math.abs(target - nx) < 0.002 ? target : nx;
				rev++;
			}
			if (fade < 1) {
				fade = instant ? 1 : Math.min(1, fade + fr.dt / 0.3);
				rev++;
			}

			geo = geometry(s);
			fitCamera(camera, v, { x0: -geo.comp / 2 - 0.05, x1: geo.comp / 2 + 0.05, y0: VIEW_BOT, y1: VIEW_TOP }, INSETS);
			v.setViewProj(camera.matrix(v.aspect));
			pxu = Math.max(1, v.project(1, 0)[0] - v.project(0, 0)[0]);

			const key = `${rev}|${v.width}|${v.height}`;
			if (key !== builtKey) {
				builtKey = key;
				build();
			}
			ctx.labels.set(labels);
			emitReadouts();
		},
		draw(pass) {
			if (forest >= 0.5) field.draw(pass);
			quads.draw(pass);
			ghostLines.draw(pass);
			planeLines.draw(pass);
			treeLines.draw(pass);
			chartLines.draw(pass);
			dots.draw(pass);
		},
		command(id) {
			if (id === 'best') {
				const b = bestSplit(getTrain(sample));
				if (!b) return;
				ctx.emit('param:axis', b.axis);
				ctx.emit('param:cut', Math.round(b.threshold * 100) / 100);
			} else if (id === 'grow') {
				if (depth < MAX_DEPTH) ctx.emit('param:depth', depth + 1);
			} else if (id === 'resample') {
				ctx.emit('param:sample', (sample + 1) % N_SAMPLES);
			}
		},
		highlight(t) {
			hl = t;
			rev++;
		},
		pointer(e: ScenePointer) {
			if (e.type === 'up' || e.type === 'leave') {
				dragging = false;
				if (e.type === 'leave' && hover !== -1) {
					hover = -1;
					rev++;
				}
				return;
			}
			const [wx, wy] = ctx.view.unproject(e.x, e.y);
			const g = geo;
			const inPlane = wx >= g.pl - 0.05 && wx <= g.pr + 0.05 && wy >= g.pb - 0.05 && wy <= g.pt + 0.05;
			const probing = probeOn >= 0.5 && forest < 0.5;
			if (e.type === 'down' && probing && inPlane) dragging = true;
			if (dragging && probing) {
				const [x, y] = planeData(wx, wy);
				const m = 0.02;
				probe = [clamp(x, BOUNDS[0] + m, BOUNDS[2] - m), clamp(y, BOUNDS[1] + m, BOUNDS[3] - m)];
				rev++;
				return;
			}
			if (e.type !== 'move' || forest >= 0.5) return;
			let h = -1;
			if (inPlane) {
				const [x, y] = planeData(wx, wy);
				h = boxes.findIndex((b) => x >= b.box[0] && x < b.box[2] && y >= b.box[1] && y < b.box[3]);
			} else {
				for (let k = 0; k < leafHits.length; k++) {
					const hit = leafHits[k];
					if (Math.hypot((wx - hit.x) * pxu, (wy - hit.y) * pxu) <= hit.r) {
						h = k;
						break;
					}
				}
			}
			if (h !== hover) {
				hover = h;
				rev++;
			}
		},
		dispose() {
			field.destroy();
			quads.destroy();
			ghostLines.destroy();
			planeLines.destroy();
			treeLines.destroy();
			chartLines.destroy();
			dots.destroy();
		}
	};
	return scene;
};
