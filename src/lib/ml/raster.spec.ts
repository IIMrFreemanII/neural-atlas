import { describe, expect, it } from 'vitest';
import { drawShape, segmentDistance, shapePaths, strokeSegment, SHAPE_KINDS } from './raster.ts';

describe('raster', () => {
	it('measures distance to a segment, including past its ends', () => {
		expect(segmentDistance(1, 1, 0, 0, 2, 0)).toBeCloseTo(1, 12);
		expect(segmentDistance(5, 0, 0, 0, 2, 0)).toBeCloseTo(3, 12);
		expect(segmentDistance(3, 4, 0, 0, 0, 0)).toBeCloseTo(5, 12);
	});
	it('strokes a horizontal line with a soft edge and stays in bounds', () => {
		const n = 8;
		const d = new Float32Array(n * n);
		strokeSegment(d, n, 1.5, 3.5, 6.5, 3.5, 0.5);
		expect(d[3 * n + 4]).toBe(1); // on the line
		expect(d[2 * n + 4]).toBe(0); // one pixel away: distance 1, coverage 0
		expect(d[0]).toBe(0);
		strokeSegment(d, n, -5, -5, 20, 20, 1); // crosses the whole image, must not throw or write outside
		expect(d.length).toBe(n * n);
		expect(d[0]).toBe(1);
	});
	it('erases', () => {
		const n = 6;
		const d = new Float32Array(n * n).fill(1);
		strokeSegment(d, n, 2.5, 2.5, 2.5, 2.5, 0.5, true);
		expect(d[2 * n + 2]).toBe(0);
		expect(d[0]).toBe(1);
	});
	it('draws every shape inside the image with values in 0..1', () => {
		for (const kind of SHAPE_KINDS) {
			const img = drawShape(kind, 16, 8, 8, 5, 0.3);
			let sum = 0;
			for (const v of img) {
				expect(v).toBeGreaterThanOrEqual(0);
				expect(v).toBeLessThanOrEqual(1);
				sum += v;
			}
			expect(sum).toBeGreaterThan(8);
		}
	});
	it('closes ring shapes and crosses two strokes', () => {
		const sq = shapePaths('square', 0, 0, 1);
		expect(sq.length).toBe(1);
		expect(sq[0].length).toBe(5);
		expect(sq[0][0][0]).toBeCloseTo(sq[0][4][0], 10);
		expect(shapePaths('cross', 0, 0, 1).length).toBe(2);
	});
});
