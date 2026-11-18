import { describe, expect, it } from 'vitest';
import { attention } from './attention.ts';
import { cosine, matVec, softmax } from './vec.ts';

describe('softmax', () => {
	it('sums to 1 and keeps order', () => {
		const p = softmax([1, 3, 2]);
		expect(p.reduce((a, b) => a + b, 0)).toBeCloseTo(1, 12);
		expect(p[1]).toBeGreaterThan(p[2]);
		expect(p[2]).toBeGreaterThan(p[0]);
	});
	it('is stable for large inputs and sharpens at low temperature', () => {
		const p = softmax([1000, 1001]);
		expect(p[1]).toBeCloseTo(1 / (1 + Math.exp(-1)), 10);
		expect(softmax([1, 2], 0.01)[1]).toBeGreaterThan(0.999);
		expect(softmax([1, 2], 100)[1]).toBeLessThan(0.51);
	});
	it('gives masked entries zero', () => {
		expect(softmax([0, -Infinity, 0])).toEqual([0.5, 0, 0.5]);
		expect(softmax([-Infinity, -Infinity])).toEqual([0, 0]);
	});
});

describe('vec helpers', () => {
	it('matVec and cosine', () => {
		expect(matVec([[1, 2], [0, -1]], [3, 4])).toEqual([11, -4]);
		expect(cosine([1, 0], [0, 2])).toBe(0);
		expect(cosine([1, 1], [2, 2])).toBeCloseTo(1, 12);
		expect(cosine([0, 0], [1, 2])).toBe(0);
	});
});

describe('attention', () => {
	const q = [[1, 0], [0, 1], [1, 1]];
	const k = [[4, 0], [0, 4], [-4, -4]];
	const v = [[1, 0, 0], [0, 1, 0], [0, 0, 1]];

	it('rows of weights sum to 1 and out mixes the values', () => {
		const r = attention(q, k, v, { scale: 1 });
		for (const row of r.weights) expect(row.reduce((a, b) => a + b, 0)).toBeCloseTo(1, 12);
		expect(r.scores[0]).toEqual([4, 0, -4]);
		expect(r.weights[0][0]).toBeGreaterThan(0.95);
		expect(r.weights[1][1]).toBeGreaterThan(0.95);
		for (let i = 0; i < 3; i++) for (let d = 0; d < 3; d++) expect(r.out[i][d]).toBeCloseTo(r.weights[i][d], 12);
	});
	it('default scale is 1 / sqrt(dk)', () => {
		const r = attention(q, k, v);
		expect(r.scores[0][0]).toBeCloseTo(4 / Math.SQRT2, 12);
	});
	it('causal mask hides later tokens', () => {
		const r = attention(q, k, v, { causal: true });
		expect(r.weights[0]).toEqual([1, 0, 0]);
		expect(r.weights[1][2]).toBe(0);
		expect(r.scores[0][1]).toBe(-Infinity);
		expect(r.weights[2].reduce((a, b) => a + b, 0)).toBeCloseTo(1, 12);
	});
});
