import { describe, expect, it } from 'vitest';
import type { Dataset } from '../../../ml/datasets.ts';
import { accuracy, bestSplit, buildForest, buildTree, forestProb, splitImpurity, treeProb } from '../../../ml/tree.ts';
import {
	BOUNDS,
	FOREST_SEED,
	MAX_DEPTH,
	N_TEST,
	N_TRAIN,
	START_CUT,
	TEST,
	bestTestDepth,
	depthCurve,
	trainingData
} from './data.ts';

const treeAcc = (d: Dataset, depth: number, train: Dataset) => {
	const t = buildTree(train, depth);
	return accuracy(d, (a, b) => treeProb(t, a, b));
};

describe('decision-trees data', () => {
	it('is deterministic and sized as declared', () => {
		expect(trainingData(0).n).toBe(N_TRAIN);
		expect(TEST.n).toBe(N_TEST);
		expect(Array.from(trainingData(3).x)).toEqual(Array.from(trainingData(3).x));
		expect(Array.from(trainingData(1).x)).not.toEqual(Array.from(trainingData(2).x));
	});

	it('keeps every point strictly inside BOUNDS', () => {
		const sets = [TEST, ...Array.from({ length: 9 }, (_, s) => trainingData(s))];
		for (const d of sets) {
			for (let k = 0; k < d.n; k++) {
				expect(d.x[2 * k]).toBeGreaterThan(BOUNDS[0]);
				expect(d.x[2 * k]).toBeLessThan(BOUNDS[2]);
				expect(d.x[2 * k + 1]).toBeGreaterThan(BOUNDS[1]);
				expect(d.x[2 * k + 1]).toBeLessThan(BOUNDS[3]);
			}
		}
	});

	it('1: a depth-MAX_DEPTH tree memorises the training sample', () => {
		const curve = depthCurve(0);
		expect(curve).toHaveLength(MAX_DEPTH + 1);
		expect(curve[MAX_DEPTH].train).toBe(1);
	});

	it('2: the best test depth is shallow and the deepest tree overfits', () => {
		const curve = depthCurve(0);
		const d = bestTestDepth();
		expect(d).toBeGreaterThanOrEqual(2);
		expect(d).toBeLessThanOrEqual(6);
		expect(curve[d].test - curve[MAX_DEPTH].test).toBeGreaterThanOrEqual(0.03);
		for (const p of curve) expect(p.test).toBeLessThanOrEqual(curve[d].test);
	});

	it('3: training accuracy never decreases with depth', () => {
		const curve = depthCurve(0);
		for (let i = 1; i < curve.length; i++) expect(curve[i].train).toBeGreaterThanOrEqual(curve[i - 1].train);
	});

	it('4: the forest beats every single tree, and more trees help', () => {
		const best = Math.max(...depthCurve(0).map((p) => p.test));
		const trees = buildForest(trainingData(0), 25, MAX_DEPTH, FOREST_SEED);
		const forest = accuracy(TEST, (a, b) => forestProb(trees, a, b));
		const one = accuracy(TEST, (a, b) => forestProb(trees.slice(0, 1), a, b));
		expect(forest).toBeGreaterThan(best);
		expect(one).toBeLessThan(forest);
	});

	it('5: new samples give different deep trees', () => {
		const roots: { axis: number | undefined; threshold: number | undefined }[] = [];
		const accs: number[] = [];
		for (let s = 1; s <= 8; s++) {
			const tr = trainingData(s);
			const t = buildTree(tr, MAX_DEPTH);
			roots.push({ axis: t.axis, threshold: t.threshold });
			accs.push(treeAcc(TEST, MAX_DEPTH, tr));
		}
		const differ = roots.some(
			(r) => r.axis !== roots[0].axis || Math.abs((r.threshold ?? 0) - (roots[0].threshold ?? 0)) > 0.05
		);
		expect(differ).toBe(true);
		expect(Math.max(...accs) - Math.min(...accs)).toBeGreaterThanOrEqual(0.03);
	});

	it('6: the starting cut is clearly worse than the best root split', () => {
		const tr = trainingData(0);
		const best = bestSplit(tr)!;
		expect(splitImpurity(tr, START_CUT.axis, START_CUT.threshold) - best.impurity).toBeGreaterThanOrEqual(0.1);
		expect(START_CUT.threshold).toBeGreaterThan(BOUNDS[0]);
		expect(START_CUT.threshold).toBeLessThan(BOUNDS[2]);
	});

	it('7: test accuracy rises to a peak, then falls without jumps back up', () => {
		const test = depthCurve(0).map((p) => p.test);
		const b = bestTestDepth();
		for (let d = 0; d < MAX_DEPTH; d++) {
			if (d < b) expect(test[d + 1]).toBeGreaterThanOrEqual(test[d] - 0.01);
			else expect(test[d + 1]).toBeLessThanOrEqual(test[d] + 0.015);
		}
	});
});
