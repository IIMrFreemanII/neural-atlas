import { describe, expect, it } from 'vitest';
import { gaussian, mulberry32 } from './rng.ts';
import { assign, initCenters, runKMeans, updateCenters } from './kmeans.ts';

function blobs(centers: number[][], per: number, sd: number, seed: number): Float64Array {
	const r = mulberry32(seed);
	const out = new Float64Array(centers.length * per * 2);
	let o = 0;
	for (const [cx, cy] of centers) {
		for (let i = 0; i < per; i++) {
			out[o++] = cx + sd * gaussian(r);
			out[o++] = cy + sd * gaussian(r);
		}
	}
	return out;
}

const TRUE = [
	[-1, -0.8],
	[1, -0.6],
	[0, 1]
];

describe('kmeans', () => {
	it('never increases inertia between iterations', () => {
		const pts = blobs(TRUE, 200, 0.3, 3);
		let centers = initCenters(pts, 3, 7, 'random');
		const a = new Int32Array(pts.length / 2);
		let last = Infinity;
		for (let it = 0; it < 15; it++) {
			const inertia = assign(pts, centers, a);
			expect(inertia).toBeLessThanOrEqual(last + 1e-9);
			last = inertia;
			centers = updateCenters(pts, a, centers).centers;
		}
	});

	it('recovers three separated blobs with k-means++ seeding', () => {
		const pts = blobs(TRUE, 300, 0.1, 5);
		const res = runKMeans(pts, initCenters(pts, 3, 1, 'plusplus'));
		for (const [tx, ty] of TRUE) {
			let best = Infinity;
			for (let c = 0; c < 3; c++) best = Math.min(best, Math.hypot(res.centers[c * 2] - tx, res.centers[c * 2 + 1] - ty));
			expect(best).toBeLessThan(0.05);
		}
	});

	it('k = 1 gives the mean', () => {
		const pts = Float64Array.from([0, 0, 2, 0, 4, 6]);
		const res = runKMeans(pts, Float64Array.from([1, 1]));
		expect(res.centers[0]).toBeCloseTo(2);
		expect(res.centers[1]).toBeCloseTo(2);
	});

	it('an empty cluster keeps its center', () => {
		const pts = Float64Array.from([0, 0, 1, 0]);
		const a = Int32Array.from([0, 0]);
		const u = updateCenters(pts, a, Float64Array.from([0, 0, 9, 9]));
		expect(Array.from(u.centers)).toEqual([0.5, 0, 9, 9]);
		expect(Array.from(u.counts)).toEqual([2, 0]);
	});

	it('ties go to the lowest index', () => {
		const pts = Float64Array.from([0, 0]);
		const out = new Int32Array(1);
		assign(pts, Float64Array.from([1, 0, -1, 0, 0, 1]), out);
		expect(out[0]).toBe(0);
	});

	it('initCenters is deterministic and random picks distinct points', () => {
		const pts = blobs(TRUE, 100, 0.3, 2);
		for (const m of ['random', 'plusplus'] as const) {
			expect(initCenters(pts, 4, 9, m)).toEqual(initCenters(pts, 4, 9, m));
		}
		const c = initCenters(pts, 5, 4, 'random');
		const keys = new Set<string>();
		for (let i = 0; i < 5; i++) keys.add(`${c[i * 2]},${c[i * 2 + 1]}`);
		expect(keys.size).toBe(5);
	});
});
