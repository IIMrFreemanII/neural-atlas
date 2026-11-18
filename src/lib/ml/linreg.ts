import { gaussian, mulberry32 } from './rng.ts';

export interface LineFit {
	w: number;
	b: number;
}

export interface XY {
	x: Float64Array;
	y: Float64Array;
}

/** Mean of (w*x + b - y)^2. Returns 0 for empty input. */
export function mse(x: ArrayLike<number>, y: ArrayLike<number>, w: number, b: number): number {
	const n = x.length;
	if (n === 0) return 0;
	let s = 0;
	for (let i = 0; i < n; i++) {
		const r = w * x[i] + b - y[i];
		s += r * r;
	}
	return s / n;
}

/** Closed-form least squares. If all x are equal: { w: 0, b: mean(y) }. */
export function leastSquares(x: ArrayLike<number>, y: ArrayLike<number>): LineFit {
	const n = x.length;
	if (n === 0) return { w: 0, b: 0 };
	let mx = 0;
	let my = 0;
	for (let i = 0; i < n; i++) {
		mx += x[i];
		my += y[i];
	}
	mx /= n;
	my /= n;
	let sxx = 0;
	let sxy = 0;
	for (let i = 0; i < n; i++) {
		sxx += (x[i] - mx) * (x[i] - mx);
		sxy += (x[i] - mx) * (y[i] - my);
	}
	if (sxx < 1e-12) return { w: 0, b: my };
	const w = sxy / sxx;
	return { w, b: my - w * mx };
}

/** Gradient of mse with respect to w and b. */
export function gradient(
	x: ArrayLike<number>,
	y: ArrayLike<number>,
	w: number,
	b: number
): { dw: number; db: number } {
	const n = x.length;
	if (n === 0) return { dw: 0, db: 0 };
	let dw = 0;
	let db = 0;
	for (let i = 0; i < n; i++) {
		const r = w * x[i] + b - y[i];
		dw += r * x[i];
		db += r;
	}
	return { dw: (2 * dw) / n, db: (2 * db) / n };
}

/** n points: x evenly stratified over [-1, 1] with jitter inside each stratum, y = slope*x + intercept + noise*gaussian. */
export function makeLinearData(n: number, slope: number, intercept: number, noise: number, seed: number): XY {
	const rand = mulberry32(seed);
	const x = new Float64Array(n);
	const y = new Float64Array(n);
	for (let i = 0; i < n; i++) {
		x[i] = -1 + (2 * (i + 0.15 + 0.7 * rand())) / n;
		y[i] = slope * x[i] + intercept + noise * gaussian(rand);
	}
	return { x, y };
}
