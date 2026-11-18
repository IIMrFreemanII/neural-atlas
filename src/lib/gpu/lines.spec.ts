import { describe, expect, it } from 'vitest';
import { bezier, lengthFractions } from './primitives/lines.ts';

describe('bezier', () => {
	it('returns segments + 1 points including both ends', () => {
		const pts = bezier([0, 0], [1, 0], [2, 1], [3, 1], 10);
		expect(pts).toHaveLength(11);
		expect(pts[0]).toEqual([0, 0]);
		expect(pts[10][0]).toBeCloseTo(3, 10);
		expect(pts[10][1]).toBeCloseTo(1, 10);
	});
});

describe('lengthFractions', () => {
	it('is continuous from 0 to 1 and proportional to length', () => {
		const f = lengthFractions([
			[0, 0],
			[1, 0],
			[1, 3]
		]);
		expect(f).toEqual([0, 0.25, 1]);
	});

	it('measures arc length in 3D', () => {
		const f = lengthFractions([
			[0, 0, 0],
			[0, 0, 1],
			[0, 3, 1]
		]);
		expect(f).toEqual([0, 0.25, 1]);
		const g = lengthFractions([
			[0, 0, 0],
			[1, 2, 2],
			[1, 2, 5]
		]);
		expect(g[1]).toBeCloseTo(3 / 6, 12);
		expect(g[2]).toBe(1);
	});

	it('gives 2D input and the same points lifted to z = 0 identical results', () => {
		const flat = [[0, 0], [2, 1], [3, 5], [-1, 4]] as Array<[number, number]>;
		const lifted = flat.map(([x, y]) => [x, y, 0] as [number, number, number]);
		expect(lengthFractions(lifted)).toEqual(lengthFractions(flat));
	});

	it('handles degenerate input', () => {
		expect(lengthFractions([[1, 1], [1, 1]])).toEqual([0, 0]);
	});
});
