import type { Quads } from '../../../gpu/primitives/quads.ts';

/** A pixel grid drawn with Quads: `cols` x `rows` square cells, row 0 at the top. */
export interface GridLayout {
	/** World position of the grid's top-left corner. */
	x: number;
	y: number;
	/** World size of one cell (pitch, gap included). */
	cell: number;
	cols: number;
	rows: number;
	/** Fraction of the pitch left empty between cells, 0..0.5. Default 0.08. */
	gap?: number;
	/** Corner radius in CSS px. Default 0. */
	radius?: number;
}

export type Rgba = [number, number, number, number];

/** World center of cell (col, row). */
export function cellCenter(g: GridLayout, col: number, row: number): [number, number] {
	return [g.x + (col + 0.5) * g.cell, g.y - (row + 0.5) * g.cell];
}

/** World width and height of the whole grid. */
export function gridSize(g: GridLayout): [number, number] {
	return [g.cols * g.cell, g.rows * g.cell];
}

/** The cell under a world position, or null when outside the grid. */
export function cellAt(g: GridLayout, wx: number, wy: number): { col: number; row: number } | null {
	const col = Math.floor((wx - g.x) / g.cell);
	const row = Math.floor((g.y - wy) / g.cell);
	return col >= 0 && col < g.cols && row >= 0 && row < g.rows ? { col, row } : null;
}

/** A world position as continuous pixel coordinates of the grid (0..cols, 0..rows), for raster strokes. Not clamped. */
export function toPixel(g: GridLayout, wx: number, wy: number): [number, number] {
	return [(wx - g.x) / g.cell, (g.y - wy) / g.cell];
}

/**
 * Write one quad per cell into `quads`, starting at instance `start`. `values` is row-major (row 0 on top) and
 * `color(value, index)` gives each cell's color. Returns the next free instance index. The caller sets quads.count
 * and calls upload(). Colors are ink on paper: encode value as density (alpha), so an empty cell is simply paper.
 */
export function writeGrid(
	quads: Quads,
	start: number,
	values: ArrayLike<number>,
	g: GridLayout,
	color: (value: number, index: number) => Rgba
): number {
	const size = g.cell * (1 - (g.gap ?? 0.08));
	let i = start;
	for (let row = 0; row < g.rows; row++) {
		for (let col = 0; col < g.cols; col++) {
			const k = row * g.cols + col;
			quads.set(i++, {
				x: g.x + (col + 0.5) * g.cell,
				y: g.y - (row + 0.5) * g.cell,
				w: size,
				h: size,
				color: color(values[k], k),
				radius: g.radius ?? 0
			});
		}
	}
	return i;
}
