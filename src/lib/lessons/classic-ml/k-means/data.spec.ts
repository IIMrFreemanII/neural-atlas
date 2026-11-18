import { describe, expect, it } from 'vitest';
import { initCenters, runKMeans } from '../../../ml/kmeans.ts';
import { N, makeCloud } from './data.ts';

const FIVE = [
	[-0.85, 0.7],
	[0, 0.75],
	[0.85, 0.55],
	[-0.6, -0.7],
	[0.55, -0.6]
];

/** How many of the five blobs have their own center within 0.2. */
function blobsFound(c: Float64Array): number {
	const used = new Set<number>();
	for (const [tx, ty] of FIVE) {
		let best = -1;
		let bd = Infinity;
		for (let i = 0; i < 5; i++) {
			const d = Math.hypot(c[i * 2] - tx, c[i * 2 + 1] - ty);
			if (d < bd) {
				bd = d;
				best = i;
			}
		}
		if (bd < 0.2) used.add(best);
	}
	return used.size;
}

describe('clouds', () => {
	it('are deterministic, sized N and inside the map', () => {
		for (let id = 0; id < 4; id++) {
			const a = makeCloud(id, 1);
			expect(a.length).toBe(N * 2);
			expect(makeCloud(id, 1)).toEqual(a);
			for (const v of a) expect(Math.abs(v)).toBeLessThanOrEqual(1.35);
		}
	});

	// Lesson step 6 hard-codes seed 1: a random start gets stuck, k-means++ finds all five blobs.
	it('seed 1 on five blobs: random start is stuck, k-means++ is not', () => {
		const pts = makeCloud(1, 1);
		const bad = runKMeans(pts, initCenters(pts, 5, 1, 'random'), 100, 1e-4);
		const good = runKMeans(pts, initCenters(pts, 5, 1, 'plusplus'), 100, 1e-4);
		expect(blobsFound(good.centers)).toBe(5);
		expect(blobsFound(bad.centers)).toBeLessThan(5);
		expect(bad.inertia).toBeGreaterThan(good.inertia * 1.3);
	});
});
