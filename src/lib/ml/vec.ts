/** Small dense vector and matrix helpers. A matrix is an array of rows. */
export type Vec = number[];
export type Mat = number[][];

export function dot(a: ArrayLike<number>, b: ArrayLike<number>): number {
	let s = 0;
	for (let i = 0; i < a.length; i++) s += a[i] * b[i];
	return s;
}

export function norm(a: ArrayLike<number>): number {
	return Math.sqrt(dot(a, a));
}

/** M v, where M has one row per output component. */
export function matVec(m: Mat, v: ArrayLike<number>): Vec {
	return m.map((row) => dot(row, v));
}

/** Cosine of the angle between a and b. 0 when either vector is all zeros. */
export function cosine(a: ArrayLike<number>, b: ArrayLike<number>): number {
	const d = norm(a) * norm(b);
	return d > 0 ? dot(a, b) / d : 0;
}

/**
 * Softmax of x / temperature, computed stably. Entries equal to -Infinity get probability 0.
 * temperature must be > 0. If every entry is -Infinity the result is all zeros.
 */
export function softmax(x: ArrayLike<number>, temperature = 1): Vec {
	let max = -Infinity;
	for (let i = 0; i < x.length; i++) if (x[i] > max) max = x[i];
	const out = new Array<number>(x.length).fill(0);
	if (max === -Infinity) return out;
	let sum = 0;
	for (let i = 0; i < x.length; i++) {
		out[i] = x[i] === -Infinity ? 0 : Math.exp((x[i] - max) / temperature);
		sum += out[i];
	}
	for (let i = 0; i < x.length; i++) out[i] /= sum;
	return out;
}
