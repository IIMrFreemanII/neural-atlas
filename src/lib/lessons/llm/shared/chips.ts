import type { Camera2D } from '../../../gpu/camera.ts';
import type { View } from '../../../gpu/types.ts';

/**
 * Token chips: a rounded rectangle (Quads, world units) with a monospace label on top (HTML, CSS px).
 * The label font is JetBrains Mono, whose every glyph is 0.6 em wide, so text width is known without measuring.
 */
export const MONO_ADVANCE = 0.6;

/** Width in CSS px of `text` set in the mono label font at `fontPx`. */
export function textWidthPx(text: string, fontPx: number): number {
	return [...text].length * fontPx * MONO_ADVANCE;
}

/** CSS px per world unit of a Camera2D. Call after fitCamera. */
export function pxPerWorld(view: Pick<View, 'height'>, camera: Camera2D): number {
	return Math.max(1, view.height) / camera.viewHeight;
}

/** How a token is written on a chip: spaces become a visible middle dot, line breaks a return arrow. */
export function showToken(token: string): string {
	return token.replace(/ /g, '·').replace(/\n/g, '↵');
}

export interface ChipBox {
	/** Index into the input array. */
	index: number;
	/** Text to put in the label (already passed through showToken). */
	text: string;
	/** World center. */
	x: number;
	y: number;
	/** World size. */
	w: number;
	h: number;
	/** 0-based row after wrapping. */
	row: number;
}

export interface ChipLayout {
	/** World x range the rows may use. */
	x0: number;
	x1: number;
	/** World y of the center of the first row. Later rows go down (smaller y). */
	y: number;
	/** CSS px per world unit (pxPerWorld). */
	scale: number;
	/** Label font size in CSS px. */
	fontPx: number;
	/** Horizontal padding inside a chip, CSS px. Default 6. */
	padPx?: number;
	/** Gap between chips, CSS px. Default 4. */
	gapPx?: number;
	/** Chip height, CSS px. Default fontPx + 12. */
	heightPx?: number;
	/** Gap between rows, CSS px. Default 8. */
	rowGapPx?: number;
	/** Default 'center'. */
	align?: 'left' | 'center';
	/** Chips never get narrower than this many characters. Default 1. */
	minChars?: number;
}

/**
 * Lay tokens out left to right, wrapping to a new row when the next chip would pass x1.
 * Returns one box per token, in input order. Pure: call again when the stage size (scale) changes.
 */
export function layoutChips(tokens: readonly string[], o: ChipLayout): ChipBox[] {
	const pad = o.padPx ?? 6;
	const gap = (o.gapPx ?? 4) / o.scale;
	const h = (o.heightPx ?? o.fontPx + 12) / o.scale;
	const rowStep = h + (o.rowGapPx ?? 8) / o.scale;
	const minChars = o.minChars ?? 1;
	const maxW = o.x1 - o.x0;
	const boxes: ChipBox[] = [];
	let row = 0;
	let cursor = 0;
	let rowStart = 0;
	const finishRow = (end: number) => {
		if (end <= rowStart) return;
		const used = cursor - gap;
		const shift = o.x0 + (o.align === 'left' ? 0 : (maxW - used) / 2);
		for (let i = rowStart; i < end; i++) boxes[i].x += shift;
	};
	tokens.forEach((token, index) => {
		const text = showToken(token);
		const chars = Math.max(minChars, [...text].length);
		const w = (chars * o.fontPx * MONO_ADVANCE + pad * 2) / o.scale;
		if (cursor > 0 && cursor + w > maxW + 1e-9) {
			finishRow(index);
			row++;
			cursor = 0;
			rowStart = index;
		}
		boxes.push({ index, text, x: cursor + w / 2, y: o.y - row * rowStep, w, h, row });
		cursor += w + gap;
	});
	finishRow(tokens.length);
	return boxes;
}
