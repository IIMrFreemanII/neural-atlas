import { describe, expect, it } from 'vitest';
import { RANGE, bestLine, data, loss, lossGrid, predict } from './model.ts';

describe('loss landscape model', () => {
	it('data is deterministic and has the expected size', () => {
		expect(data(0)).toEqual(data(0));
		expect(data(0)).toHaveLength(10);
		expect(data(1)).toHaveLength(12);
	});

	it('bestLine is near the generating line and beats perturbations', () => {
		const { w, b } = bestLine();
		expect(Math.abs(w - 0.8)).toBeLessThan(0.2);
		expect(Math.abs(b + 0.2)).toBeLessThan(0.2);
		const best = loss(0, w, b);
		for (const [dw, db] of [[0.05, 0], [-0.05, 0], [0, 0.05], [0, -0.05], [0.03, -0.04]]) {
			expect(loss(0, w + dw, b + db)).toBeGreaterThan(best);
		}
	});

	it('loss is 0 for a perfect fit', () => {
		for (const model of [0, 1] as const) {
			const pts = data(model);
			const w = 0.7;
			const b = -0.3;
			const exact = pts.map((p) => ({ x: p.x, y: predict(model, w, b, p.x) }));
			let s = 0;
			for (const p of exact) s += (predict(model, w, b, p.x) - p.y) ** 2;
			expect(s / exact.length).toBe(0);
		}
		expect(loss(0, 0.8, -0.2)).toBeGreaterThan(0);
	});

	it('lossGrid min is the smallest value and argmin is near bestLine', () => {
		const cols = 72;
		const rows = 72;
		const g = lossGrid(0, cols, rows);
		expect(g.min).toBe(Math.min(...g.values));
		expect(g.max).toBe(Math.max(...g.values));
		const cell = (RANGE.w[1] - RANGE.w[0]) / (cols - 1);
		const best = bestLine();
		expect(Math.abs(g.argmin.w - best.w)).toBeLessThanOrEqual(cell);
		expect(Math.abs(g.argmin.b - best.b)).toBeLessThanOrEqual(cell);
	});
});
