import { describe, expect, it } from 'vitest';
import { gelu, layerNorm, positional, randomBlock, runBlock } from './block.ts';
import type { Mat } from './vec.ts';

const inputs = (n: number, d: number, seed: number): Mat => {
	const w = randomBlock(d, seed);
	// Reuse seeded Gaussian rows as test input.
	return w.wq.slice(0, n).map((r) => r.map((v) => v * 3));
};

describe('layerNorm', () => {
	const x = [1, -2, 3, 0.5, 4, -1, 2, 7];
	it('has mean 0 and variance about 1', () => {
		const y = layerNorm(x);
		const mean = y.reduce((a, b) => a + b, 0) / y.length;
		const variance = y.reduce((a, b) => a + (b - mean) ** 2, 0) / y.length;
		expect(mean).toBeCloseTo(0, 10);
		expect(variance).toBeCloseTo(1, 4);
	});
	it('ignores scaling and shifting of the input', () => {
		const a = layerNorm(x);
		const b = layerNorm(x.map((v) => 5 * v + 3));
		for (let i = 0; i < x.length; i++) expect(b[i]).toBeCloseTo(a[i], 4);
	});
});

describe('gelu', () => {
	it('matches known values', () => {
		expect(gelu(0)).toBe(0);
		expect(gelu(3)).toBeCloseTo(2.996, 3);
		expect(gelu(-3)).toBeCloseTo(-0.004, 3);
	});
});

describe('positional', () => {
	it('starts with [0, 1, 0, 1, ...] and has shape n x d', () => {
		const p = positional(5, 8);
		expect(p).toHaveLength(5);
		expect(p[0]).toEqual([0, 1, 0, 1, 0, 1, 0, 1]);
		expect(p[3]).toHaveLength(8);
		expect(p[1][0]).toBeCloseTo(Math.sin(1), 10);
	});
});

describe('runBlock', () => {
	const d = 8;
	const w = randomBlock(d, 3);
	const x = inputs(5, d, 9);
	const t = runBlock(x, w);

	it('adds updates exactly', () => {
		for (let i = 0; i < 5; i++) {
			for (let k = 0; k < d; k++) {
				expect(t.mid[i][k]).toBe(t.x[i][k] + t.attnOut[i][k]);
				expect(t.out[i][k]).toBe(t.mid[i][k] + t.mlpOut[i][k]);
			}
		}
	});
	it('has hidden rows of length 4d', () => {
		for (const r of t.hidden) expect(r).toHaveLength(4 * d);
	});
	it('respects the causal mask', () => {
		const x2 = x.map((r) => [...r]);
		x2[4] = x2[4].map((v, k) => v + (k % 2 === 0 ? 2 : -1));
		const c = runBlock(x2, w, true);
		for (let i = 0; i < 4; i++) for (let k = 0; k < d; k++) expect(c.out[i][k]).toBe(t.out[i][k]);
		const a = runBlock(x, w, false);
		const b = runBlock(x2, w, false);
		let diff = 0;
		for (let i = 0; i < 4; i++) for (let k = 0; k < d; k++) diff += Math.abs(a.out[i][k] - b.out[i][k]);
		expect(diff).toBeGreaterThan(1e-6);
	});
});

describe('randomBlock', () => {
	it('is deterministic in the seed', () => {
		expect(randomBlock(8, 4)).toEqual(randomBlock(8, 4));
		expect(randomBlock(8, 4).wq).not.toEqual(randomBlock(8, 5).wq);
	});
	it('has the documented shapes', () => {
		const w = randomBlock(8, 1);
		expect(w.w1).toHaveLength(32);
		expect(w.w1[0]).toHaveLength(8);
		expect(w.w2).toHaveLength(8);
		expect(w.w2[0]).toHaveLength(32);
	});
});
