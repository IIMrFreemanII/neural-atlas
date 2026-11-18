import { gaussian, mulberry32 } from '../../../ml/rng.ts';

/** 0 = line: y = w*x + b.  1 = tanh neuron: y = tanh(w*x + b). */
export type ModelId = 0 | 1;

export const RANGE = { w: [-2, 2], b: [-2, 2] } as const;

export interface Point {
	x: number;
	y: number;
}

const cache = new Map<ModelId, readonly Point[]>();

/** Fixed, seeded data for each model. */
export function data(model: ModelId): readonly Point[] {
	const hit = cache.get(model);
	if (hit) return hit;
	const rand = mulberry32(model === 0 ? 7 : 11);
	const pts: Point[] = [];
	if (model === 0) {
		const n = 10;
		for (let i = 0; i < n; i++) {
			const x = -1 + (2 * i) / (n - 1);
			pts.push({ x, y: 0.8 * x - 0.2 + 0.15 * gaussian(rand) });
		}
	} else {
		const n = 12;
		for (let i = 0; i < n; i++) {
			const x = -1.5 + (3 * i) / (n - 1);
			pts.push({ x, y: Math.tanh(1.5 * x - 0.5) + 0.08 * gaussian(rand) });
		}
	}
	cache.set(model, pts);
	return pts;
}

export function predict(model: ModelId, w: number, b: number, x: number): number {
	return model === 0 ? w * x + b : Math.tanh(w * x + b);
}

/** Mean squared error. */
export function loss(model: ModelId, w: number, b: number): number {
	const pts = data(model);
	let s = 0;
	for (const p of pts) {
		const e = predict(model, w, b, p.x) - p.y;
		s += e * e;
	}
	return s / pts.length;
}

/** Closed-form least squares for model 0. */
export function bestLine(): { w: number; b: number } {
	const pts = data(0);
	const n = pts.length;
	let mx = 0;
	let my = 0;
	for (const p of pts) {
		mx += p.x;
		my += p.y;
	}
	mx /= n;
	my /= n;
	let sxy = 0;
	let sxx = 0;
	for (const p of pts) {
		sxy += (p.x - mx) * (p.y - my);
		sxx += (p.x - mx) * (p.x - mx);
	}
	const w = sxy / sxx;
	return { w, b: my - w * mx };
}

/** Loss on a cols x rows grid over RANGE (col -> w, row -> b), index = row * cols + col. */
export function lossGrid(
	model: ModelId,
	cols: number,
	rows: number
): { values: Float32Array; min: number; max: number; argmin: { w: number; b: number } } {
	const values = new Float32Array(cols * rows);
	let min = Infinity;
	let max = -Infinity;
	let am = { w: 0, b: 0 };
	for (let r = 0; r < rows; r++) {
		const b = RANGE.b[0] + ((RANGE.b[1] - RANGE.b[0]) * r) / (rows - 1);
		for (let c = 0; c < cols; c++) {
			const w = RANGE.w[0] + ((RANGE.w[1] - RANGE.w[0]) * c) / (cols - 1);
			const v = Math.fround(loss(model, w, b));
			values[r * cols + c] = v;
			if (v < min) {
				min = v;
				am = { w, b };
			}
			if (v > max) max = v;
		}
	}
	return { values, min, max, argmin: am };
}
