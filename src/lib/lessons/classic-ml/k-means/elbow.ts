import { assign, initCenters, runKMeans } from '../../../ml/kmeans.ts';

/** How many points the search for good centers uses. The spread itself is measured on every point. */
const SAMPLE = 2000;
/** k-means++ starts per k. One start is not enough: with five blobs it often merges two of them. */
const STARTS = 10;
const SEED = 11;

/**
 * The points of the elbow chart: for k = 1..maxK, the spread (inertia over all of `points`) of the best
 * clustering found. k-means can stop in a poor result, so every k is run from several starts and the one
 * with the lowest spread is kept. Otherwise one unlucky start puts a false bend in the curve.
 * The runs use an evenly spaced sample of the points, which must be in random order. Deterministic.
 */
export function elbowCurve(points: Float64Array, maxK: number): Float64Array {
	const n = points.length / 2;
	const m = Math.min(n, SAMPLE);
	const stride = Math.max(1, Math.floor(n / m));
	const sub = new Float64Array(m * 2);
	for (let i = 0; i < m; i++) {
		sub[i * 2] = points[i * stride * 2];
		sub[i * 2 + 1] = points[i * stride * 2 + 1];
	}
	const out = new Float64Array(maxK);
	const scratch = new Int32Array(n);
	for (let k = 1; k <= maxK; k++) {
		let best: Float64Array | undefined;
		let bestInertia = Infinity;
		for (let s = 0; s < STARTS; s++) {
			const r = runKMeans(sub, initCenters(sub, k, SEED + s, 'plusplus'), 60, 1e-5);
			if (r.inertia < bestInertia) {
				bestInertia = r.inertia;
				best = r.centers;
			}
		}
		out[k - 1] = best ? assign(points, best, scratch) : 0;
	}
	return out;
}
