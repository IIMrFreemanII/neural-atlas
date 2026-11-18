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

	it('handles degenerate input', () => {
		expect(lengthFractions([[1, 1], [1, 1]])).toEqual([0, 0]);
	});
});
