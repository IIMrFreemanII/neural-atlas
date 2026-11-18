import { afterEach, describe, expect, it } from 'vitest';
import { diverging, hexToLinear, mixColor, paperFill, sequential, sheetFill, tone, toneHex } from './colors.ts';
import { PALETTES, paletteName, setPalette } from './theme.ts';

afterEach(() => setPalette('paper'));

describe('colors', () => {
	it('converts known sRGB values to linear', () => {
		const [r, g, b] = hexToLinear('#808080');
		expect(r).toBeCloseTo(0.2158605, 5);
		expect(g).toBeCloseTo(0.2158605, 5);
		expect(b).toBeCloseTo(0.2158605, 5);
		expect(hexToLinear('#ffffff')).toEqual([1, 1, 1]);
		expect(hexToLinear('#000000')).toEqual([0, 0, 0]);
	});

	it('tone density is the alpha, multiplied by alpha, and clamps', () => {
		const [r, g, b] = hexToLinear(PALETTES.paper.tones.data);
		const t = tone('data', 0.5, 0.5);
		expect([t[0], t[1], t[2]]).toEqual([r, g, b]);
		expect(t[3]).toBeCloseTo(0.25, 6);
		expect(tone('data')[3]).toBe(1);
		expect(tone('data', 3)[3]).toBe(1);
		expect(tone('data', -1)[3]).toBe(0);
	});

	it('follows the active palette', () => {
		expect(paletteName()).toBe('paper');
		const paper = tone('text');
		setPalette('night');
		expect(paletteName()).toBe('night');
		const night = tone('text');
		expect(night.slice(0, 3)).toEqual(hexToLinear(PALETTES.night.tones.text));
		expect(night[0]).toBeGreaterThan(paper[0]);
		expect(toneHex('data')).toBe(PALETTES.night.tones.data);
	});

	it('paperFill and sheetFill use the palette backgrounds', () => {
		const [r, g, b] = hexToLinear(PALETTES.paper.pg);
		expect(paperFill()).toEqual([r, g, b, 1]);
		expect(paperFill(0.3)[3]).toBe(0.3);
		expect(sheetFill().slice(0, 3)).toEqual(hexToLinear(PALETTES.paper.sheet));
		setPalette('night');
		expect(paperFill().slice(0, 3)).toEqual(hexToLinear(PALETTES.night.pg));
	});

	it('mixColor lerps componentwise', () => {
		expect(mixColor([0, 0, 0, 0], [2, 4, 6, 1], 0.5)).toEqual([1, 2, 3, 0.5]);
		expect(mixColor([1, 2, 3, 1], [5, 5, 5, 1], 0)).toEqual([1, 2, 3, 1]);
	});

	it('diverging picks the hue by sign and the alpha by magnitude', () => {
		const neg = diverging(-1);
		const pos = diverging(1);
		expect(neg).toEqual(tone('neg'));
		expect(pos).toEqual(tone('data'));
		expect(neg[0]).toBeGreaterThan(neg[1]);
		expect(pos[2]).toBeGreaterThan(pos[0]);
		expect(diverging(0.5)[3]).toBeCloseTo(0.54, 6);
	});

	it('diverging is nearly transparent at zero and clamps beyond 1', () => {
		expect(diverging(0)[3]).toBeCloseTo(0.08, 6);
		expect(diverging(5)).toEqual(diverging(1));
		expect(diverging(-5)).toEqual(diverging(-1));
		expect(diverging(1, 1, 0.5)[3]).toBe(0.5);
		expect(diverging(1, 0.5)[3]).toBeCloseTo(0.5, 6);
	});

	it('sequential runs from light param ink to full data ink and clamps', () => {
		expect(sequential(0)).toEqual(tone('param', 0.25));
		const hi = sequential(1);
		const d = tone('data');
		d.forEach((x, i) => expect(hi[i]).toBeCloseTo(x, 6));
		expect(sequential(-3)).toEqual(sequential(0));
		expect(sequential(9)).toEqual(sequential(1));
		expect(sequential(1, 0.5, 0.3)[0]).toBeCloseTo(d[0], 6);
		expect(sequential(1, 0.5, 0.3)[3]).toBeCloseTo(0.15, 6);
	});
});
