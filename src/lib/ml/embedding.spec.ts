import { describe, expect, it } from 'vitest';
import { analogy, nearest, pca3 } from './embedding.ts';
import { dot } from './vec.ts';
import { gaussian, mulberry32 } from './rng.ts';

describe('nearest', () => {
	const vectors = [
		[1, 0],
		[0.9, 0.1],
		[0, 1],
		[-1, 0],
		[0.7, 0.7]
	];
	it('orders by cosine similarity, best first', () => {
		const r = nearest(vectors, [1, 0], 3);
		expect(r.map((x) => x.index)).toEqual([0, 1, 4]);
		expect(r[0].similarity).toBeCloseTo(1, 10);
		expect(r[1].similarity).toBeGreaterThan(r[2].similarity);
	});
	it('skips excluded indices', () => {
		const r = nearest(vectors, [1, 0], 2, [0, 1]);
		expect(r.map((x) => x.index)).toEqual([4, 2]);
	});
	it('returns fewer rows when k is larger than what is left', () => {
		expect(nearest(vectors, [1, 0], 10, [0])).toHaveLength(4);
	});
});

describe('analogy', () => {
	it('computes b - a + c', () => {
		expect(analogy([1, 2, 3], [4, 6, 8], [0, 1, 0])).toEqual([3, 5, 5]);
	});
});

function random(n: number, d: number, seed: number): number[][] {
	const rand = mulberry32(seed);
	return Array.from({ length: n }, () => Array.from({ length: d }, () => gaussian(rand)));
}

describe('pca3', () => {
	it('finds that points in a plane need only two components', () => {
		const rand = mulberry32(3);
		// a plane spanned by two orthonormal-ish directions in 6 dimensions, offset from the origin
		const u = [1, 0, 1, 0, 0, 0].map((x) => x / Math.SQRT2);
		const w = [0, 1, 0, 1, 1, 0].map((x) => x / Math.sqrt(3));
		const pts = Array.from({ length: 60 }, () => {
			const s = 3 * gaussian(rand);
			const t = gaussian(rand);
			return u.map((x, i) => s * x + t * w[i] + 5);
		});
		const r = pca3(pts);
		expect(r.explained[0] + r.explained[1]).toBeGreaterThan(1 - 1e-9);
		expect(r.explained[2]).toBeLessThan(1e-9);
		expect(r.explained[0]).toBeGreaterThan(r.explained[1]);
	});

	it('returns orthonormal components', () => {
		const r = pca3(random(50, 8, 4));
		for (let i = 0; i < 3; i++) {
			for (let j = 0; j < 3; j++) {
				expect(dot(r.components[i], r.components[j])).toBeCloseTo(i === j ? 1 : 0, 8);
			}
		}
	});

	it('preserves pairwise distances when the data is truly 3-dimensional', () => {
		const rand = mulberry32(8);
		const basis = [
			[1, 2, 0, 1, 0],
			[0, 1, 1, 0, 2],
			[1, 0, 0, 2, 1]
		];
		const pts = Array.from({ length: 20 }, () => {
			const c = [gaussian(rand), gaussian(rand), gaussian(rand)];
			return [0, 1, 2, 3, 4].map((j) => c[0] * basis[0][j] + c[1] * basis[1][j] + c[2] * basis[2][j] + 1);
		});
		const r = pca3(pts);
		const dist = (a: number[], b: number[]) => Math.hypot(...a.map((x, i) => x - b[i]));
		for (let i = 0; i < pts.length; i++) {
			for (let j = i + 1; j < pts.length; j++) {
				expect(dist(r.points[i], r.points[j])).toBeCloseTo(dist(pts[i], pts[j]), 8);
			}
		}
		expect(r.explained[0] + r.explained[1] + r.explained[2]).toBeCloseTo(1, 9);
	});

	it('is deterministic and fixes each sign so the biggest entry is positive', () => {
		const data = random(40, 6, 12);
		const a = pca3(data);
		const b = pca3(data);
		expect(a.components).toEqual(b.components);
		expect(a.points).toEqual(b.points);
		for (const comp of a.components) {
			let big = 0;
			for (let j = 1; j < comp.length; j++) if (Math.abs(comp[j]) > Math.abs(comp[big])) big = j;
			expect(comp[big]).toBeGreaterThan(0);
		}
	});

	it('projects the mean to the origin', () => {
		const r = pca3(random(30, 5, 2));
		const sum = [0, 1, 2].map((k) => r.points.reduce((s, p) => s + p[k], 0));
		for (const s of sum) expect(Math.abs(s)).toBeLessThan(1e-9);
	});
});
