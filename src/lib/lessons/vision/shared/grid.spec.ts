import { describe, expect, it } from 'vitest';
import { cellAt, cellCenter, gridSize, toPixel, type GridLayout } from './grid.ts';

const g: GridLayout = { x: -1, y: 2, cell: 0.5, cols: 4, rows: 3 };

describe('grid', () => {
	it('places cells with row 0 on top', () => {
		expect(cellCenter(g, 0, 0)).toEqual([-0.75, 1.75]);
		expect(cellCenter(g, 3, 2)).toEqual([0.75, 0.75]);
		expect(gridSize(g)).toEqual([2, 1.5]);
	});
	it('hit-tests and converts to pixel coordinates', () => {
		expect(cellAt(g, -0.9, 1.9)).toEqual({ col: 0, row: 0 });
		expect(cellAt(g, 0.9, 0.6)).toEqual({ col: 3, row: 2 });
		expect(cellAt(g, 1.1, 1)).toBeNull();
		expect(cellAt(g, 0, 2.1)).toBeNull();
		expect(toPixel(g, -0.75, 1.75)).toEqual([0.5, 0.5]);
	});
});
