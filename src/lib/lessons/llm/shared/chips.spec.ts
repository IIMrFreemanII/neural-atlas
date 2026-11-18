import { describe, expect, it } from 'vitest';
import { layoutChips, showToken, textWidthPx } from './chips.ts';

describe('chips', () => {
	it('measures mono text and shows spaces', () => {
		expect(textWidthPx('abcd', 10)).toBeCloseTo(24, 10);
		expect(showToken(' the')).toBe('·the');
	});
	it('centers a single row', () => {
		const b = layoutChips(['ab', 'c'], { x0: -1, x1: 1, y: 0, scale: 100, fontPx: 10, padPx: 5, gapPx: 10 });
		// widths: (12 + 10) / 100 = 0.22 and (6 + 10) / 100 = 0.16, gap 0.1 -> total 0.48
		expect(b[0].w).toBeCloseTo(0.22, 10);
		expect(b[1].w).toBeCloseTo(0.16, 10);
		expect(b[0].x - b[0].w / 2).toBeCloseTo(-0.24, 10);
		expect(b[1].x + b[1].w / 2).toBeCloseTo(0.24, 10);
		expect(b.every((c) => c.row === 0 && c.y === 0)).toBe(true);
	});
	it('wraps and keeps every chip inside the range', () => {
		const tokens = Array.from({ length: 30 }, (_, i) => 'tok' + i);
		const b = layoutChips(tokens, { x0: 0, x1: 3, y: 1, scale: 100, fontPx: 12 });
		expect(b.length).toBe(30);
		expect(b[b.length - 1].row).toBeGreaterThan(0);
		for (const c of b) {
			expect(c.x - c.w / 2).toBeGreaterThanOrEqual(-1e-9);
			expect(c.x + c.w / 2).toBeLessThanOrEqual(3 + 1e-9);
		}
		expect(b[b.length - 1].y).toBeLessThan(b[0].y);
	});
});
