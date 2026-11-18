import { gaussian } from './rng.ts';

/** An equal-weight mixture of K round Gaussians: the "data" of the toy diffusion world. */
export interface Mixture {
	/** K x 2 centers, xy pairs. */
	centers: Float64Array;
	sd: number;
}

/** Variance-preserving noise schedule for t in [0, 1]. */
export function schedule(t: number): { alpha: number; sigma: number } {
	return { alpha: Math.cos((Math.PI * t) / 2), sigma: Math.sin((Math.PI * t) / 2) };
}

/** n samples of the data: a random center plus sd * Gaussian noise. Returns xy pairs. */
export function sampleData(m: Mixture, n: number, rand: () => number): Float64Array {
	const k = m.centers.length / 2;
	const out = new Float64Array(n * 2);
	for (let i = 0; i < n; i++) {
		const c = Math.min(k - 1, Math.floor(rand() * k));
		out[i * 2] = m.centers[c * 2] + m.sd * gaussian(rand);
		out[i * 2 + 1] = m.centers[c * 2 + 1] + m.sd * gaussian(rand);
	}
	return out;
}

/** x_t = alpha x0 + sigma eps, elementwise, into out (same length). */
export function noise(x0: Float64Array, eps: Float64Array, t: number, out: Float64Array): void {
	const { alpha, sigma } = schedule(t);
	for (let i = 0; i < x0.length; i++) out[i] = alpha * x0[i] + sigma * eps[i];
}

/** Variance of every component of the noised cloud at time t. */
function variance(m: Mixture, alpha: number, sigma: number): number {
	return alpha * alpha * m.sd * m.sd + sigma * sigma;
}

function weights(m: Mixture, x: number, y: number, alpha: number, v: number, out: Float64Array): void {
	const k = m.centers.length / 2;
	let max = -Infinity;
	for (let c = 0; c < k; c++) {
		const dx = x - alpha * m.centers[c * 2];
		const dy = y - alpha * m.centers[c * 2 + 1];
		const l = -(dx * dx + dy * dy) / (2 * v);
		out[c] = l;
		if (l > max) max = l;
	}
	let sum = 0;
	for (let c = 0; c < k; c++) {
		out[c] = Math.exp(out[c] - max);
		sum += out[c];
	}
	for (let c = 0; c < k; c++) out[c] /= sum;
}

/** Posterior weights of the K components for a point (x, y) at time t (log-sum-exp stable), into out (length K). */
export function responsibilities(m: Mixture, x: number, y: number, t: number, out: Float64Array): void {
	const { alpha, sigma } = schedule(t);
	weights(m, x, y, alpha, variance(m, alpha, sigma), out);
}

function denoisedWith(m: Mixture, x: number, y: number, alpha: number, sigma: number, r: Float64Array): [number, number] {
	const k = m.centers.length / 2;
	const v = variance(m, alpha, sigma);
	weights(m, x, y, alpha, v, r);
	const g = (alpha * m.sd * m.sd) / v;
	let ex = 0;
	let ey = 0;
	for (let c = 0; c < k; c++) {
		const mx = m.centers[c * 2];
		const my = m.centers[c * 2 + 1];
		ex += r[c] * (mx + g * (x - alpha * mx));
		ey += r[c] * (my + g * (y - alpha * my));
	}
	return [ex, ey];
}

/** E[x_0 | x_t = (x, y)]: the best guess of the clean point. */
export function denoised(m: Mixture, x: number, y: number, t: number): [number, number] {
	const { alpha, sigma } = schedule(t);
	return denoisedWith(m, x, y, alpha, sigma, new Float64Array(m.centers.length / 2));
}

/** Gradient of the log density of the noised cloud at (x, y): sum_k r_k (alpha mu_k - x) / v. */
export function score(m: Mixture, x: number, y: number, t: number): [number, number] {
	const { alpha, sigma } = schedule(t);
	const k = m.centers.length / 2;
	const v = variance(m, alpha, sigma);
	const r = new Float64Array(k);
	weights(m, x, y, alpha, v, r);
	let sx = 0;
	let sy = 0;
	for (let c = 0; c < k; c++) {
		sx += (r[c] * (alpha * m.centers[c * 2] - x)) / v;
		sy += (r[c] * (alpha * m.centers[c * 2 + 1] - y)) / v;
	}
	return [sx, sy];
}

/** Log density of the noised cloud at (x, y) (for tests). */
export function logDensity(m: Mixture, x: number, y: number, t: number): number {
	const { alpha, sigma } = schedule(t);
	const k = m.centers.length / 2;
	const v = variance(m, alpha, sigma);
	const l = new Float64Array(k);
	let max = -Infinity;
	for (let c = 0; c < k; c++) {
		const dx = x - alpha * m.centers[c * 2];
		const dy = y - alpha * m.centers[c * 2 + 1];
		l[c] = -(dx * dx + dy * dy) / (2 * v);
		if (l[c] > max) max = l[c];
	}
	let sum = 0;
	for (let c = 0; c < k; c++) sum += Math.exp(l[c] - max);
	return max + Math.log(sum / k) - Math.log(2 * Math.PI * v);
}

/**
 * One deterministic reverse step (DDIM) for every point in pts (xy pairs, in place), from time tFrom to a smaller tTo.
 * With tFrom so small that sigma(tFrom) < 1e-6 the point is set to x0.
 */
export function reverseStep(m: Mixture, pts: Float64Array, tFrom: number, tTo: number): void {
	const from = schedule(tFrom);
	const to = schedule(tTo);
	const r = new Float64Array(m.centers.length / 2);
	for (let i = 0; i < pts.length; i += 2) {
		const [x0x, x0y] = denoisedWith(m, pts[i], pts[i + 1], from.alpha, from.sigma, r);
		if (from.sigma < 1e-6) {
			pts[i] = x0x;
			pts[i + 1] = x0y;
			continue;
		}
		const ex = (pts[i] - from.alpha * x0x) / from.sigma;
		const ey = (pts[i + 1] - from.alpha * x0y) / from.sigma;
		pts[i] = to.alpha * x0x + to.sigma * ex;
		pts[i + 1] = to.alpha * x0y + to.sigma * ey;
	}
}
