import { palette } from './theme.ts';
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

const linearCache = new Map<string, [number, number, number]>();

function linear(hex: string): [number, number, number] {
	let v = linearCache.get(hex);
	if (!v) {
		v = hexToLinear(hex);
		linearCache.set(hex, v);
	}
	return v;
}

const clamp01 = (x: number) => Math.max(0, Math.min(1, x));

/** The active palette's sRGB hex for a tone (for CSS and labels). */
export function toneHex(t: Tone): string {
	return palette().tones[t];
}

/**
 * Ink of a tone, as a straight (non-premultiplied) linear RGBA color.
 * `density` is how much ink is laid down: it scales opacity, not brightness. Values above 1 clamp to 1.
 */
export function tone(t: Tone, density = 1, alpha = 1): [number, number, number, number] {
	const [r, g, b] = linear(palette().tones[t]);
	return [r, g, b, alpha * clamp01(density)];
}

/** Opaque stage background color, for punching shapes out of what is behind them. */
export function paperFill(alpha = 1): [number, number, number, number] {
	const [r, g, b] = linear(palette().pg);
	return [r, g, b, alpha];
}

/** Raised-surface color (the card/sheet color). */
export function sheetFill(alpha = 1): [number, number, number, number] {
	const [r, g, b] = linear(palette().sheet);
	return [r, g, b, alpha];
}

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

/** Signed value color: v in [-1, 1] (clamped). Negative -> 'neg' ink, positive -> 'data' ink, ink density = (0.08 + 0.92 * |v|) * density, so zero reads as bare paper. */
export function diverging(v: number, density = 1, alpha = 1): [number, number, number, number] {
	const c = Math.max(-1, Math.min(1, v));
	return tone(c < 0 ? 'neg' : 'data', (0.08 + 0.92 * Math.abs(c)) * density, alpha);
}

/** Unsigned value color: v in [0, 1] (clamped). Lerps from light 'param' ink (density 0.25) to full 'data' ink; `density` scales the result's opacity. */
export function sequential(v: number, density = 1, alpha = 1): [number, number, number, number] {
	const m = mixColor(tone('param', 0.25), tone('data'), clamp01(v));
	return [m[0], m[1], m[2], m[3] * clamp01(density) * alpha];
}
