import { describe, expect, it } from 'vitest';
import { makeBoundaryData } from '../lessons/classic-ml/decision-boundaries/data.ts';
import { makeDataset, type Dataset } from './datasets.ts';
import { mulberry32 } from './rng.ts';
import {
	accuracy,
	bestSplit,
	bootstrap,
	buildForest,
	buildTree,
	countLeaves,
	forestProb,
	layoutTree,
	splitCurve,
	splitImpurity,
	treeBoxes,
	treePath,
	treeProb,
	treeSplits,
	type TreeNode
} from './tree.ts';

const treeAccuracy = (t: TreeNode, d: Dataset) => {
	let ok = 0;
	for (let k = 0; k < d.n; k++) if ((treeProb(t, d.x[2 * k], d.x[2 * k + 1]) >= 0.5 ? 1 : 0) === d.y[k]) ok++;
	return ok / d.n;
};

describe('tree', () => {
	it('maxDepth 0 is a single leaf with the class fraction', () => {
		const d = makeBoundaryData(1, 0.05, 1);
		const t = buildTree(d, 0);
		expect(countLeaves(t)).toBe(1);
		expect(t.prob).toBeCloseTo(0.5, 12);
		expect(t.axis).toBeUndefined();
	});

	it('solves a vertically separable set at depth 1 on the x axis', () => {
		const x: number[] = [];
		const y: number[] = [];
		for (let i = 0; i < 20; i++) {
			x.push(i < 10 ? -0.5 + i * 0.01 : 0.5 + i * 0.01, (i % 5) * 0.2 - 0.4);
			y.push(i < 10 ? 0 : 1);
		}
		const d: Dataset = { n: 20, x: Float64Array.from(x), y: Float64Array.from(y) };
		const t = buildTree(d, 1);
		expect(t.axis).toBe(0);
		expect(t.threshold!).toBeGreaterThan(-0.41);
		expect(t.threshold!).toBeLessThan(0.5);
		expect(treeAccuracy(t, d)).toBe(1);
		expect(countLeaves(t)).toBe(2);
	});

	it('reaches 100% training accuracy when deep enough', () => {
		for (const id of [1, 2, 3]) {
			const d = makeBoundaryData(id, 0.1, 4);
			expect(treeAccuracy(buildTree(d, 40), d)).toBe(1);
		}
	});

	it('split segments stay inside bounds and parent regions', () => {
		const d = makeBoundaryData(1, 0.1, 2);
		const t = buildTree(d, 4);
		const b: [number, number, number, number] = [-1.4, -1.4, 1.4, 1.4];
		const segs = treeSplits(t, b);
		expect(segs.length).toBe(countLeaves(t) - 1);
		for (const s of segs) {
			const [lo, hi] = s.axis === 0 ? [b[1], b[3]] : [b[0], b[2]];
			expect(s.lo).toBeGreaterThanOrEqual(lo);
			expect(s.hi).toBeLessThanOrEqual(hi);
			expect(s.lo).toBeLessThan(s.hi);
			expect(s.threshold).toBeGreaterThan(s.axis === 0 ? b[0] : b[1]);
			expect(s.threshold).toBeLessThan(s.axis === 0 ? b[2] : b[3]);
		}
		const root = segs.find((s) => s.depth === 0)!;
		expect(root.lo).toBe(root.axis === 0 ? b[1] : b[0]);
		for (const s of segs.filter((q) => q.depth > 0)) {
			expect(s.hi - s.lo).toBeLessThan(2.8 + 1e-9);
		}
	});

	it('counts leaves', () => {
		const leaf = (depth: number): TreeNode => ({ depth, n: 1, prob: 0 });
		const t: TreeNode = { depth: 0, n: 3, prob: 0.3, axis: 0, threshold: 0, left: leaf(1), right: { depth: 1, n: 2, prob: 0.5, axis: 1, threshold: 0, left: leaf(2), right: leaf(2) } };
		expect(countLeaves(t)).toBe(3);
	});
});

describe('tree helpers', () => {
	const moons = () => makeDataset(1, 120, 0.2, 5);
	const all = (t: TreeNode): TreeNode[] => (t.left && t.right ? [t, ...all(t.left), ...all(t.right)] : [t]);
	const leaves = (t: TreeNode): TreeNode[] => all(t).filter((n) => !n.left);

	it('splitImpurity is 0 for a perfect cut and weights an empty side as 0', () => {
		const d: Dataset = { n: 4, x: Float64Array.from([0, 0, 1, 0, 2, 0, 3, 0]), y: Float64Array.from([0, 0, 1, 1]) };
		expect(splitImpurity(d, 0, 1.5)).toBe(0);
		expect(splitImpurity(d, 0, 10)).toBeCloseTo(0.5, 12);
		expect(splitImpurity(d, 0, 1.5, [0, 1])).toBe(0);
	});

	it('splitCurve thresholds ascend and its minimum is the best split', () => {
		const d = moons();
		const curves = ([0, 1] as const).map((a) => splitCurve(d, a));
		for (const c of curves) for (let i = 1; i < c.length; i++) expect(c[i].threshold).toBeGreaterThan(c[i - 1].threshold);
		const min = Math.min(...curves.flat().map((p) => p.impurity));
		expect(min).toBeCloseTo(bestSplit(d)!.impurity, 12);
		for (const p of curves[0].slice(0, 5)) expect(p.impurity).toBeCloseTo(splitImpurity(d, 0, p.threshold), 12);
	});

	it('bestSplit matches the root of buildTree and is null for a pure node', () => {
		const d = moons();
		const t = buildTree(d, 3);
		const s = bestSplit(d)!;
		expect(s.axis).toBe(t.axis);
		expect(s.threshold).toBe(t.threshold);
		expect(bestSplit(d, [0])).toBeNull();
	});

	it('random-axis trees are deterministic and still memorise when deep', () => {
		const d = moons();
		const a = buildTree(d, 40, { rand: mulberry32(3) });
		const b = buildTree(d, 40, { rand: mulberry32(3) });
		expect(a).toEqual(b);
		expect(accuracy(d, (x, y) => treeProb(a, x, y))).toBe(1);
	});

	it('bootstrap keeps n and only original points', () => {
		const d = moons();
		const s = bootstrap(d, mulberry32(1));
		expect(s.n).toBe(d.n);
		const orig = new Set<string>();
		for (let k = 0; k < d.n; k++) orig.add(`${d.x[2 * k]},${d.x[2 * k + 1]},${d.y[k]}`);
		for (let k = 0; k < s.n; k++) expect(orig.has(`${s.x[2 * k]},${s.x[2 * k + 1]},${s.y[k]}`)).toBe(true);
	});

	it('forest prefix property and forestProb of one tree', () => {
		const d = moons();
		const big = buildForest(d, 25, 6, 7);
		const small = buildForest(d, 10, 6, 7);
		expect(big.slice(0, 10)).toEqual(small);
		expect(forestProb([big[0]], 0.1, -0.2)).toBe(treeProb(big[0], 0.1, -0.2));
		expect(forestProb([], 0, 0)).toBe(0.5);
	});

	it('treePath ends in the leaf treeProb returns', () => {
		const t = buildTree(moons(), 5);
		for (const [x, y] of [[0, 0], [-0.7, 0.3], [0.8, -0.4]]) {
			const p = treePath(t, x, y);
			expect(p[0]).toBe(t);
			expect(p[p.length - 1].left).toBeUndefined();
			expect(p[p.length - 1].prob).toBe(treeProb(t, x, y));
		}
	});

	it('treeBoxes tile the bounds and match treeProb', () => {
		const t = buildTree(moons(), 5);
		const b: [number, number, number, number] = [-1.5, -1.2, 1.5, 1.2];
		const boxes = treeBoxes(t, b);
		expect(boxes.length).toBe(countLeaves(t));
		const area = boxes.reduce((s, q) => s + (q.box[2] - q.box[0]) * (q.box[3] - q.box[1]), 0);
		expect(area).toBeCloseTo(3 * 2.4, 9);
		boxes.forEach((q, i) => {
			expect(q.index).toBe(i);
			const cx = (q.box[0] + q.box[2]) / 2;
			const cy = (q.box[1] + q.box[3]) / 2;
			expect(treeProb(t, cx, cy)).toBe(q.leaf.prob);
		});
		expect(boxes.map((q) => q.leaf)).toEqual(leaves(t));
	});

	it('layoutTree is pre-order with valid parents and x inside (0, 1)', () => {
		const t = buildTree(moons(), 4);
		const lay = layoutTree(t);
		expect(lay.length).toBe(all(t).length);
		expect(lay[0].node).toBe(t);
		expect(lay[0].parent).toBe(-1);
		lay.forEach((n, i) => {
			expect(n.x).toBeGreaterThan(0);
			expect(n.x).toBeLessThan(1);
			expect(n.depth).toBe(n.node.depth);
			if (i > 0) {
				expect(n.parent).toBeGreaterThanOrEqual(0);
				expect(n.parent).toBeLessThan(i);
				expect(lay[n.parent].depth).toBe(n.depth - 1);
			}
		});
		const k = countLeaves(t);
		const ls = lay.filter((n) => !n.node.left).map((n) => n.x);
		ls.forEach((x, i) => expect(x).toBeCloseTo((i + 0.5) / k, 12));
		const single = layoutTree(buildTree(moons(), 0));
		expect(single).toHaveLength(1);
		expect(single[0].x).toBe(0.5);
	});

	it('accuracy counts matching labels', () => {
		const d: Dataset = { n: 4, x: new Float64Array(8), y: Float64Array.from([0, 1, 1, 0]) };
		expect(accuracy(d, () => 0.9)).toBe(0.5);
		expect(accuracy(d, () => 0.2)).toBe(0.5);
	});
});
