import { describe, expect, it } from 'vitest';
import { mulberry32 } from './rng.ts';
import { chebyshev, evalPoly, fitPoly, polyMse } from './polyfit.ts';

function sample(n: number, seed: number, f: (x: number) => number) {
	const r = mulberry32(seed);
	const x = new Float64Array(n);
	const y = new Float64Array(n);
	for (let i = 0; i < n; i++) {
		x[i] = -0.97 + 1.94 * r();
		y[i] = f(x[i]);
	}
	return { x, y };
}

describe('polyfit', () => {
	it('chebyshev matches cos(k acos x)', () => {
		const t = chebyshev(6, 0.3);
		for (let k = 0; k <= 6; k++) expect(t[k]).toBeCloseTo(Math.cos(k * Math.acos(0.3)), 12);
	});

	it('recovers an exact cubic', () => {
		const f = (x: number) => 0.5 - 1.2 * x + 0.7 * x * x + 2 * x * x * x;
		const { x, y } = sample(20, 3, f);
		const m = fitPoly(x, y, 3, 0);
		for (const q of [-0.9, -0.3, 0, 0.45, 0.95]) expect(evalPoly(m, q)).toBeCloseTo(f(q), 6);
	});

	it('degree n - 1 interpolates n points', () => {
		const r = mulberry32(8);
		const { x, y } = sample(10, 5, () => r() - 0.5);
		const m = fitPoly(x, y, 9, 0);
		for (let i = 0; i < 10; i++) expect(Math.abs(evalPoly(m, x[i]) - y[i])).toBeLessThan(1e-3);
	});

	it('training MSE never increases with degree', () => {
		const r = mulberry32(21);
		const { x, y } = sample(20, 4, (v) => Math.sin(3 * v) + 0.2 * (r() - 0.5));
		let prev = Infinity;
		for (let d = 1; d <= 12; d++) {
			const e = polyMse(fitPoly(x, y, d, 0), x, y);
			expect(e).toBeLessThanOrEqual(prev + 1e-9);
			prev = e;
		}
	});

	it('a larger lambda shrinks the coefficients', () => {
		const { x, y } = sample(15, 6, (v) => Math.sin(4 * v));
		const norm = (lam: number) => {
			const c = fitPoly(x, y, 10, lam).coef;
			let s = 0;
			for (let j = 1; j < c.length; j++) s += c[j] * c[j];
			return s;
		};
		expect(norm(0.1)).toBeLessThan(norm(0.001));
		expect(norm(1)).toBeLessThan(norm(0.1));
	});

	it('stays finite for degree 14 with 10 points', () => {
		const { x, y } = sample(10, 9, (v) => Math.cos(5 * v));
		const m = fitPoly(x, y, 14, 0);
		for (const c of m.coef) expect(Number.isFinite(c)).toBe(true);
		for (const q of [-1, -0.5, 0, 0.5, 1]) expect(Number.isFinite(evalPoly(m, q))).toBe(true);
	});
});
