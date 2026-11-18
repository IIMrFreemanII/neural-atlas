import { describe, expect, it } from 'vitest';
import { makeBoundaryData } from '../lessons/classic-ml/decision-boundaries/data.ts';
import type { Dataset } from './datasets.ts';
import { buildTree, countLeaves, treeProb, treeSplits, type TreeNode } from './tree.ts';

const accuracy = (t: TreeNode, d: Dataset) => {
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
		expect(accuracy(t, d)).toBe(1);
		expect(countLeaves(t)).toBe(2);
	});

	it('reaches 100% training accuracy when deep enough', () => {
		for (const id of [1, 2, 3]) {
			const d = makeBoundaryData(id, 0.1, 4);
			expect(accuracy(buildTree(d, 40), d)).toBe(1);
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
