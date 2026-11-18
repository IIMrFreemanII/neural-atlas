import { describe, expect, it } from 'vitest';
import { gelu, layerNorm } from './block.ts';

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
