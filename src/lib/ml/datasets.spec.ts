import { describe, expect, it } from 'vitest';
import { makeDataset, type DatasetId } from './datasets.ts';

const ids: DatasetId[] = [0, 1, 2, 3];

describe('makeDataset', () => {
	it('has exact size and balance, and stays in range', () => {
		for (const id of ids) {
			const d = makeDataset(id, 200, 0.1, 3);
			expect(d.n).toBe(200);
			expect(d.x.length).toBe(400);
			expect(d.y.length).toBe(200);
			expect(d.y.reduce((s, v) => s + v, 0)).toBe(100);
			for (const v of d.x) expect(Math.abs(v)).toBeLessThanOrEqual(1.6);
		}
	});

	it('is deterministic and differs across ids', () => {
		for (const id of ids) {
			expect(Array.from(makeDataset(id, 50, 0.05, 9).x)).toEqual(
				Array.from(makeDataset(id, 50, 0.05, 9).x)
			);
		}
		expect(Array.from(makeDataset(0, 50, 0, 1).x)).not.toEqual(Array.from(makeDataset(1, 50, 0, 1).x));
		expect(Array.from(makeDataset(2, 50, 0, 1).x)).not.toEqual(Array.from(makeDataset(3, 50, 0, 1).x));
	});

	it('separates circles by radius without noise', () => {
		const d = makeDataset(0, 200, 0, 2);
		for (let k = 0; k < d.n; k++) {
			const r = Math.hypot(d.x[2 * k], d.x[2 * k + 1]);
			if (d.y[k] === 0) expect(r).toBeLessThanOrEqual(0.36);
			else expect(r).toBeGreaterThan(0.7);
		}
	});
});
