import { activate, type ActivationId } from '../../../ml/activations.ts';

export const X_MIN = -3,
	X_MAX = 3;

const SLOPE = 1.5;
const SAMPLES = 121;
const RIDGE = 1e-6;

/** The curve the tiny network tries to match: a smooth wave. */
export function target(x: number): number {
	return Math.sin(1.6 * x) * 0.8 + 0.25 * x;
}

/** Center of hidden unit k of n. */
export function center(n: number, k: number): number {
	return X_MIN + ((k + 1) * (X_MAX - X_MIN)) / (n + 1);
}

/** Hidden unit k of n: h_k(x) = activate(act, s * (x - c_k)). */
export function unit(act: ActivationId, n: number, k: number, x: number): number {
	return activate(act, SLOPE * (x - center(n, k)));
}

export interface BasisFit {
	v: number[];
	bias: number;
	error: number;
}

/** Solve A x = b in place by Gaussian elimination with partial pivoting. */
function solve(A: number[][], b: number[]): number[] {
	const m = b.length;
	for (let c = 0; c < m; c++) {
		let p = c;
		for (let r = c + 1; r < m; r++) if (Math.abs(A[r][c]) > Math.abs(A[p][c])) p = r;
		[A[c], A[p]] = [A[p], A[c]];
		[b[c], b[p]] = [b[p], b[c]];
		const d = A[c][c];
		if (Math.abs(d) < 1e-14) continue;
		for (let r = c + 1; r < m; r++) {
			const f = A[r][c] / d;
			if (f === 0) continue;
			for (let k = c; k < m; k++) A[r][k] -= f * A[c][k];
			b[r] -= f * b[c];
		}
	}
	const x = new Array<number>(m).fill(0);
	for (let r = m - 1; r >= 0; r--) {
		let s = b[r];
		for (let k = r + 1; k < m; k++) s -= A[r][k] * x[k];
		x[r] = Math.abs(A[r][r]) < 1e-14 ? 0 : s / A[r][r];
	}
	return x;
}

/** Least-squares fit of y = bias + sum_k v[k] * unit(act, n, k, x) to target on 121 samples. */
export function fitBasis(act: ActivationId, n: number): BasisFit {
	const m = n + 1; // columns: units then constant
	const AtA: number[][] = Array.from({ length: m }, () => new Array<number>(m).fill(0));
	const Atb = new Array<number>(m).fill(0);
	const row = new Array<number>(m);
	const xs: number[] = [];
	for (let i = 0; i < SAMPLES; i++) {
		const x = X_MIN + ((X_MAX - X_MIN) * i) / (SAMPLES - 1);
		xs.push(x);
		for (let k = 0; k < n; k++) row[k] = unit(act, n, k, x);
		row[n] = 1;
		const y = target(x);
		for (let a = 0; a < m; a++) {
			Atb[a] += row[a] * y;
			for (let b = 0; b < m; b++) AtA[a][b] += row[a] * row[b];
		}
	}
	for (let a = 0; a < m; a++) AtA[a][a] += RIDGE;
	const sol = solve(AtA, Atb);
	const fit: BasisFit = { v: sol.slice(0, n), bias: sol[n], error: 0 };
	let se = 0;
	for (const x of xs) {
		const d = evalFit(act, n, fit, x) - target(x);
		se += d * d;
	}
	fit.error = Math.sqrt(se / SAMPLES);
	return fit;
}

export function evalFit(act: ActivationId, n: number, fit: BasisFit, x: number): number {
	let y = fit.bias;
	for (let k = 0; k < n; k++) y += fit.v[k] * unit(act, n, k, x);
	return y;
}
