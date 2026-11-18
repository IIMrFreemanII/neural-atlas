/**
 * Polynomial curve fitting on x in [-1, 1], in the Chebyshev basis for numerical stability.
 * Pure TypeScript, no DOM.
 */
export interface PolyModel {
	degree: number;
	/** Chebyshev coefficients c0..c_degree */
	coef: Float64Array;
}

/** Floor added to the penalty of every coefficient j >= 1 so the problem stays solvable when degree >= n - 1. */
const RIDGE_FLOOR = 1e-12;

/** T_0(x)..T_degree(x) by the three-term recurrence. */
export function chebyshev(degree: number, x: number, out?: Float64Array): Float64Array {
	const t = out ?? new Float64Array(degree + 1);
	t[0] = 1;
	if (degree >= 1) t[1] = x;
	for (let j = 2; j <= degree; j++) t[j] = 2 * x * t[j - 1] - t[j - 2];
	return t;
}

/**
 * Ridge least squares: minimize sum (y_i - sum_j c_j T_j(x_i))^2 + lambda * sum_{j>=1} c_j^2 (c0 is not penalized).
 * Solved by Householder QR of the stacked system [A; sqrt(penalty) I].
 */
export function fitPoly(x: ArrayLike<number>, y: ArrayLike<number>, degree: number, lambda = 0): PolyModel {
	const n = x.length;
	const p = degree + 1;
	const rows = n + degree;
	const a = new Float64Array(rows * p);
	const b = new Float64Array(rows);
	const basis = new Float64Array(p);
	for (let i = 0; i < n; i++) {
		chebyshev(degree, x[i], basis);
		for (let j = 0; j < p; j++) a[i * p + j] = basis[j];
		b[i] = y[i];
	}
	const root = Math.sqrt(Math.max(0, lambda) + RIDGE_FLOOR);
	for (let j = 1; j < p; j++) a[(n + j - 1) * p + j] = root;

	// Householder QR, applying each reflection to b as well.
	for (let k = 0; k < p; k++) {
		let norm = 0;
		for (let i = k; i < rows; i++) norm += a[i * p + k] * a[i * p + k];
		norm = Math.sqrt(norm);
		if (norm === 0) continue;
		const alpha = a[k * p + k] > 0 ? -norm : norm;
		const v = new Float64Array(rows - k);
		for (let i = k; i < rows; i++) v[i - k] = a[i * p + k];
		v[0] -= alpha;
		let vv = 0;
		for (let i = 0; i < v.length; i++) vv += v[i] * v[i];
		if (vv === 0) continue;
		for (let j = k; j < p; j++) {
			let dot = 0;
			for (let i = k; i < rows; i++) dot += v[i - k] * a[i * p + j];
			const f = (2 * dot) / vv;
			for (let i = k; i < rows; i++) a[i * p + j] -= f * v[i - k];
		}
		let dot = 0;
		for (let i = k; i < rows; i++) dot += v[i - k] * b[i];
		const f = (2 * dot) / vv;
		for (let i = k; i < rows; i++) b[i] -= f * v[i - k];
	}

	// Back substitution on the upper triangle.
	const coef = new Float64Array(p);
	for (let k = p - 1; k >= 0; k--) {
		let s = b[k];
		for (let j = k + 1; j < p; j++) s -= a[k * p + j] * coef[j];
		const d = a[k * p + k];
		coef[k] = Math.abs(d) > 1e-300 ? s / d : 0;
	}
	return { degree, coef };
}

export function evalPoly(m: PolyModel, x: number): number {
	// Clenshaw recurrence.
	let b1 = 0;
	let b2 = 0;
	for (let j = m.degree; j >= 1; j--) {
		const b0 = m.coef[j] + 2 * x * b1 - b2;
		b2 = b1;
		b1 = b0;
	}
	return m.coef[0] + x * b1 - b2;
}

/** Mean squared error of the model on (x, y). */
export function polyMse(m: PolyModel, x: ArrayLike<number>, y: ArrayLike<number>): number {
	const n = x.length;
	if (n === 0) return 0;
	let s = 0;
	for (let i = 0; i < n; i++) {
		const d = y[i] - evalPoly(m, x[i]);
		s += d * d;
	}
	return s / n;
}
