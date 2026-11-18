import { diverging } from '../../../gpu/colors.ts';
import type { Quads } from '../../../gpu/primitives/quads.ts';

export interface StripStyle {
	/** World center of the whole strip. */
	x: number;
	y: number;
	/** World size of one cell along the strip and across it. */
	cell: number;
	/** Cell size across the strip. Default = cell. */
	thickness?: number;
	/** World gap between cells. Default cell * 0.12. */
	gap?: number;
	/** Cells run top to bottom instead of left to right. Default false. */
	vertical?: boolean;
	/** Value that maps to full color. Default 1. */
	maxAbs?: number;
	/** Brightness multiplier. Default 0.9 (keeps a full cell below the bloom threshold). */
	intensity?: number;
	/** Default 1. */
	alpha?: number;
	/** Corner radius in CSS px. Default 2. */
	radius?: number;
}

/** World length of a strip of n cells. */
export function stripLength(n: number, cell: number, gap = cell * 0.12): number {
	return n * cell + Math.max(0, n - 1) * gap;
}

/**
 * Write a vector as a row (or column) of colored cells into `quads`, starting at instance `start`.
 * Positive values are 'data' cyan, negative 'neg' rose, brightness = |value| / maxAbs.
 * Returns the next free instance index. The caller sets quads.count and calls upload().
 */
export function writeStrip(quads: Quads, start: number, values: ArrayLike<number>, s: StripStyle): number {
	const n = values.length;
	const gap = s.gap ?? s.cell * 0.12;
	const len = stripLength(n, s.cell, gap);
	const maxAbs = s.maxAbs ?? 1;
	const thick = s.thickness ?? s.cell;
	for (let i = 0; i < n; i++) {
		const off = -len / 2 + s.cell / 2 + i * (s.cell + gap);
		quads.set(start + i, {
			x: s.vertical ? s.x : s.x + off,
			y: s.vertical ? s.y - off : s.y,
			w: s.vertical ? thick : s.cell,
			h: s.vertical ? s.cell : thick,
			color: diverging(values[i] / maxAbs, s.intensity ?? 0.9, s.alpha ?? 1),
			radius: s.radius ?? 2
		});
	}
	return start + n;
}

/**
 * Write a vector as a grid of `cols` columns of colored cells (row-major, first row on top), centered at (x, y).
 * Colors match writeStrip. `cell` is the cell size in both directions unless `thickness` is given for the height.
 * Returns the next free instance index. The caller sets quads.count and calls upload().
 */
export function writeGrid(quads: Quads, start: number, values: ArrayLike<number>, cols: number, s: StripStyle): number {
	const n = values.length;
	const rows = Math.ceil(n / cols);
	const gap = s.gap ?? s.cell * 0.12;
	const thick = s.thickness ?? s.cell;
	const width = stripLength(cols, s.cell, gap);
	const height = rows * thick + Math.max(0, rows - 1) * gap;
	const maxAbs = s.maxAbs ?? 1;
	for (let i = 0; i < n; i++) {
		const r = Math.floor(i / cols);
		const c = i % cols;
		quads.set(start + i, {
			x: s.x - width / 2 + s.cell / 2 + c * (s.cell + gap),
			y: s.y + height / 2 - thick / 2 - r * (thick + gap),
			w: s.cell,
			h: thick,
			color: diverging(values[i] / maxAbs, s.intensity ?? 0.9, s.alpha ?? 1),
			radius: s.radius ?? 2
		});
	}
	return start + n;
}
