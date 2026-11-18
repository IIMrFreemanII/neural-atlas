import { describe, expect, it } from 'vitest';
import { diverging, hexToLinear, mixColor, sequential, tone, VOID } from './colors.ts';

describe('colors', () => {
	it('converts known sRGB values to linear', () => {
		const [r, g, b] = hexToLinear('#808080');
		expect(r).toBeCloseTo(0.2158605, 5);
		expect(g).toBeCloseTo(0.2158605, 5);
		expect(b).toBeCloseTo(0.2158605, 5);
		expect(hexToLinear('#ffffff')).toEqual([1, 1, 1]);
		expect(hexToLinear('#000000')).toEqual([0, 0, 0]);
	});

	it('scales tone intensity but not alpha', () => {
		const [r] = hexToLinear('#4de2ff');
		const t = tone('data', 2, 0.5);
		expect(t[0]).toBeCloseTo(r * 2, 6);
		expect(t[3]).toBe(0.5);
		expect(VOID[2]).toBeGreaterThan(VOID[0]);
	});

	it('mixColor lerps componentwise', () => {
		expect(mixColor([0, 0, 0, 0], [2, 4, 6, 1], 0.5)).toEqual([1, 2, 3, 0.5]);
		expect(mixColor([1, 2, 3, 1], [5, 5, 5, 1], 0)).toEqual([1, 2, 3, 1]);
	});

	it('diverging picks the tone by sign', () => {
		const neg = diverging(-1);
		const pos = diverging(1);
		expect(neg).toEqual(tone('neg'));
		expect(pos).toEqual(tone('data'));
		expect(neg[0]).toBeGreaterThan(neg[1]);
		expect(pos[2]).toBeGreaterThan(pos[0]);
	});

	it('diverging is nearly black at zero and clamps beyond 1', () => {
		const z = diverging(0);
		expect(Math.max(z[0], z[1], z[2])).toBeLessThan(0.1);
		expect(diverging(5)).toEqual(diverging(1));
		expect(diverging(-5)).toEqual(diverging(-1));
		expect(diverging(1, 2, 0.5)[3]).toBe(0.5);
		expect(diverging(1, 2)[2]).toBeCloseTo(tone('data', 2)[2], 6);
	});

	it('sequential runs from dim param to full data and clamps', () => {
		expect(sequential(0)).toEqual(tone('param', 0.25));
		const hi = sequential(1);
		const d = tone('data');
		d.forEach((x, i) => expect(hi[i]).toBeCloseTo(x, 6));
		expect(sequential(-3)).toEqual(sequential(0));
		expect(sequential(9)).toEqual(sequential(1));
		expect(sequential(1, 0.5, 0.3)[0]).toBeCloseTo(d[0] * 0.5, 6);
		expect(sequential(1, 0.5, 0.3)[3]).toBe(0.3);
	});
});
