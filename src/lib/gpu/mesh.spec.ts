import { describe, expect, it } from 'vitest';
import { buildGridIndices } from './primitives/mesh.ts';

describe('buildGridIndices', () => {
	it('has (cols-1)*(rows-1)*6 indices, all inside the grid', () => {
		for (const [cols, rows] of [[2, 2], [5, 3], [64, 64]]) {
			const idx = buildGridIndices(cols, rows);
			expect(idx.length).toBe((cols - 1) * (rows - 1) * 6);
			for (const i of idx) expect(i).toBeLessThan(cols * rows);
		}
	});

	it('builds the expected first cell', () => {
		expect(Array.from(buildGridIndices(2, 2))).toEqual([0, 1, 2, 1, 3, 2]);
	});
});
