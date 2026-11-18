import { describe, expect, it } from 'vitest';
import { gradient, leastSquares, makeLinearData, mse } from './linreg.ts';

describe('leastSquares', () => {
	it('recovers an exact line', () => {
		const x = [-1, -0.3, 0.2, 0.9];
		const y = x.map((v) => 0.7 * v - 0.2);
		const f = leastSquares(x, y);
		expect(f.w).toBeCloseTo(0.7, 10);
		expect(f.b).toBeCloseTo(-0.2, 10);
	});

	it('handles equal x', () => {
		expect(leastSquares([0.5, 0.5, 0.5], [1, 2, 3])).toEqual({ w: 0, b: 2 });
	});

	it('is not beaten by nearby lines', () => {
		const { x, y } = makeLinearData(20, 0.8, 0.1, 0.2, 3);
		const f = leastSquares(x, y);
		const best = mse(x, y, f.w, f.b);
		for (const [dw, db] of [[0.05, 0], [-0.05, 0], [0, 0.05], [0, -0.05], [0.03, -0.03]]) {
			expect(mse(x, y, f.w + dw, f.b + db)).toBeGreaterThanOrEqual(best);
		}
	});
});

describe('gradient', () => {
	it('matches a numerical gradient', () => {
		const { x, y } = makeLinearData(12, 0.5, 0.2, 0.3, 9);
		const w = 0.3;
		const b = -0.4;
		const e = 1e-6;
		const g = gradient(x, y, w, b);
		expect(g.dw).toBeCloseTo((mse(x, y, w + e, b) - mse(x, y, w - e, b)) / (2 * e), 5);
		expect(g.db).toBeCloseTo((mse(x, y, w, b + e) - mse(x, y, w, b - e)) / (2 * e), 5);
	});

	it('is zero at the least-squares solution', () => {
		const { x, y } = makeLinearData(15, -0.6, 0.3, 0.25, 5);
		const f = leastSquares(x, y);
		const g = gradient(x, y, f.w, f.b);
		expect(Math.abs(g.dw)).toBeLessThan(1e-10);
		expect(Math.abs(g.db)).toBeLessThan(1e-10);
	});
});

describe('makeLinearData', () => {
	it('is deterministic and keeps x in range', () => {
		const a = makeLinearData(14, 0.8, 0.1, 0.2, 1);
		const b = makeLinearData(14, 0.8, 0.1, 0.2, 1);
		expect(Array.from(a.x)).toEqual(Array.from(b.x));
		expect(Array.from(a.y)).toEqual(Array.from(b.y));
		for (const v of a.x) {
			expect(v).toBeGreaterThanOrEqual(-1);
			expect(v).toBeLessThanOrEqual(1);
		}
		expect(mse([], [], 1, 1)).toBe(0);
	});
});
