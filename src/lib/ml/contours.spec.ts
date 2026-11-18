import { describe, expect, it } from 'vitest';
import { contourSegments } from './contours.ts';

describe('contourSegments', () => {
	const cols = 60;
	const rows = 60;
	const cx = 29.5;
	const cy = 29.5;
	const values = new Float32Array(cols * rows);
	for (let r = 0; r < rows; r++)
		for (let c = 0; c < cols; c++) values[r * cols + c] = (c - cx) ** 2 + (r - cy) ** 2;

	it('traces a circle within half a cell', () => {
		const radius = 15;
		const segs = contourSegments(values, cols, rows, radius * radius);
		expect(segs.length).toBeGreaterThan(40);
		for (const [x0, y0, x1, y1] of segs) {
			expect(Math.abs(Math.hypot(x0 - cx, y0 - cy) - radius)).toBeLessThan(0.5);
			expect(Math.abs(Math.hypot(x1 - cx, y1 - cy) - radius)).toBeLessThan(0.5);
		}
	});
	it('returns nothing for a level outside the data range', () => {
		expect(contourSegments(values, cols, rows, -5)).toEqual([]);
		expect(contourSegments(values, cols, rows, 1e9)).toEqual([]);
	});
	it('resolves a saddle cell into two segments', () => {
		const v = new Float32Array([1, 0, 0, 1]); // tl, tr / bl, br
		expect(contourSegments(v, 2, 2, 0.5)).toHaveLength(2);
	});
});
