import type { Tone } from './types.ts';

function srgbToLinear(c: number): number {
	return c <= 0.04045 ? c / 12.92 : Math.pow((c + 0.055) / 1.055, 2.4);
}

/** '#rrggbb' (sRGB) -> linear RGB. */
export function hexToLinear(hex: string): [number, number, number] {
	const h = hex.replace('#', '');
	const n = parseInt(h.length === 3 ? h.replace(/./g, '$&$&') : h, 16);
	return [
		srgbToLinear(((n >> 16) & 255) / 255),
		srgbToLinear(((n >> 8) & 255) / 255),
		srgbToLinear((n & 255) / 255)
	];
}

export const TONE_HEX: Record<Tone, string> = {
	text: '#e8eaf2',
	muted: '#5c647a',
	data: '#4de2ff',
	param: '#9b7bff',
	grad: '#ffb347',
	neg: '#ff5d8f',
	ok: '#5df2a6'
};

export function tone(t: Tone, intensity = 1, alpha = 1): [number, number, number, number] {
	const [r, g, b] = hexToLinear(TONE_HEX[t]);
	return [r * intensity, g * intensity, b * intensity, alpha];
}

export const VOID: [number, number, number] = hexToLinear('#07080d');

/** Componentwise lerp of two rgba colors. */
export function mixColor(
	a: readonly [number, number, number, number],
	b: readonly [number, number, number, number],
	t: number
): [number, number, number, number] {
	return [
		a[0] + (b[0] - a[0]) * t,
		a[1] + (b[1] - a[1]) * t,
		a[2] + (b[2] - a[2]) * t,
		a[3] + (b[3] - a[3]) * t
	];
}

/** Signed value color: v in [-1, 1] (clamped). Negative -> 'neg' tone, positive -> 'data' tone, brightness = (0.08 + 0.92 * |v|) * intensity. */
export function diverging(v: number, intensity = 1, alpha = 1): [number, number, number, number] {
	const c = Math.max(-1, Math.min(1, v));
	return tone(c < 0 ? 'neg' : 'data', (0.08 + 0.92 * Math.abs(c)) * intensity, alpha);
}

/** Unsigned value color: v in [0, 1] (clamped). Lerps from 'param' tone at 0.25 brightness to 'data' tone at full brightness, times intensity. */
export function sequential(v: number, intensity = 1, alpha = 1): [number, number, number, number] {
	const c = Math.max(0, Math.min(1, v));
	const m = mixColor(tone('param', 0.25), tone('data'), c);
	return [m[0] * intensity, m[1] * intensity, m[2] * intensity, alpha];
}
