import { describe, expect, it } from 'vitest';
import { MAX_DEPTH, nextTokenProbs, TARGETS, TRACES } from './model.ts';

describe('toy model', () => {
	it('has one trace per block with finite, bounded values', () => {
		expect(TRACES).toHaveLength(12);
		for (const t of TRACES) {
			for (const m of [t.x, t.norm1, t.attnOut, t.mid, t.norm2, t.hidden, t.mlpOut, t.out]) {
				for (const row of m) for (const v of row) expect(Number.isFinite(v)).toBe(true);
			}
			for (const row of t.out) for (const v of row) expect(Math.abs(v)).toBeLessThanOrEqual(12);
		}
	});
	it('reproduces the target probabilities at full depth', () => {
		const p = nextTokenProbs(MAX_DEPTH);
		TARGETS.forEach((t, i) => expect(Math.abs(p[i] - t)).toBeLessThan(1e-9));
	});
	it('always returns a distribution', () => {
		for (let d = 1; d <= MAX_DEPTH; d++) {
			const p = nextTokenProbs(d);
			expect(p.reduce((a, b) => a + b, 0)).toBeCloseTo(1, 10);
		}
	});
});
