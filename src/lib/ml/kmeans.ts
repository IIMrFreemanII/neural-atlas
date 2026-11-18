import { mulberry32 } from './rng.ts';

/** Squared distance from point i of `points` to center c of `centers`. */
function dist2(points: Float64Array, i: number, centers: Float64Array, c: number): number {
	const dx = points[i * 2] - centers[c * 2];
	const dy = points[i * 2 + 1] - centers[c * 2 + 1];
	return dx * dx + dy * dy;
}

/** k starting centers (k * 2 floats). 'random' = k distinct points chosen uniformly; 'plusplus' = k-means++ seeding. Deterministic in seed. */
export function initCenters(
	points: Float64Array,
	k: number,
	seed: number,
	method: 'random' | 'plusplus'
): Float64Array {
	const n = points.length / 2;
	const rand = mulberry32(seed);
	const out = new Float64Array(k * 2);
	const chosen: number[] = [];
	const take = (idx: number) => {
		chosen.push(idx);
		out[(chosen.length - 1) * 2] = points[idx * 2];
		out[(chosen.length - 1) * 2 + 1] = points[idx * 2 + 1];
	};
	if (n === 0) return out;
	if (method === 'random') {
		// partial Fisher-Yates over indices: distinct points
		const idx = Uint32Array.from({ length: n }, (_, i) => i);
		for (let i = 0; i < Math.min(k, n); i++) {
			const j = i + Math.floor(rand() * (n - i));
			const t = idx[i];
			idx[i] = idx[j];
			idx[j] = t;
			take(idx[i]);
		}
		return out;
	}
	take(Math.min(n - 1, Math.floor(rand() * n)));
	const d2 = new Float64Array(n).fill(Infinity);
	for (let c = 1; c < k; c++) {
		let total = 0;
		const last = c - 1;
		for (let i = 0; i < n; i++) {
			const d = dist2(points, i, out, last);
			if (d < d2[i]) d2[i] = d;
			total += d2[i];
		}
		if (total <= 0) {
			take(Math.min(n - 1, Math.floor(rand() * n)));
			continue;
		}
		let r = rand() * total;
		let pick = n - 1;
		for (let i = 0; i < n; i++) {
			r -= d2[i];
			if (r < 0) {
				pick = i;
				break;
			}
		}
		take(pick);
	}
	return out;
}

/** Writes the nearest-center index of every point into out (ties -> lowest index). Returns the inertia: sum of squared distances to the assigned center. */
export function assign(points: Float64Array, centers: Float64Array, out: Int32Array): number {
	const n = points.length / 2;
	const k = centers.length / 2;
	let inertia = 0;
	for (let i = 0; i < n; i++) {
		let best = 0;
		let bd = Infinity;
		for (let c = 0; c < k; c++) {
			const d = dist2(points, i, centers, c);
			if (d < bd) {
				bd = d;
				best = c;
			}
		}
		out[i] = best;
		inertia += bd;
	}
	return inertia;
}

/** Mean of each cluster's points. A cluster with no points keeps its previous center. `moved` = largest distance any center moved. */
export function updateCenters(
	points: Float64Array,
	assignment: Int32Array,
	prev: Float64Array
): { centers: Float64Array; counts: Int32Array; moved: number } {
	const n = points.length / 2;
	const k = prev.length / 2;
	const sums = new Float64Array(k * 2);
	const counts = new Int32Array(k);
	for (let i = 0; i < n; i++) {
		const c = assignment[i];
		sums[c * 2] += points[i * 2];
		sums[c * 2 + 1] += points[i * 2 + 1];
		counts[c]++;
	}
	const centers = new Float64Array(k * 2);
	let moved = 0;
	for (let c = 0; c < k; c++) {
		if (counts[c] > 0) {
			centers[c * 2] = sums[c * 2] / counts[c];
			centers[c * 2 + 1] = sums[c * 2 + 1] / counts[c];
		} else {
			centers[c * 2] = prev[c * 2];
			centers[c * 2 + 1] = prev[c * 2 + 1];
		}
		moved = Math.max(moved, Math.hypot(centers[c * 2] - prev[c * 2], centers[c * 2 + 1] - prev[c * 2 + 1]));
	}
	return { centers, counts, moved };
}

/** Alternate assign and update until moved < tol or maxIter. Returns the final centers, assignment, inertia and the number of iterations done. */
export function runKMeans(
	points: Float64Array,
	centers0: Float64Array,
	maxIter = 100,
	tol = 1e-6
): { centers: Float64Array; assignment: Int32Array; inertia: number; iterations: number } {
	const assignment = new Int32Array(points.length / 2);
	let centers: Float64Array = new Float64Array(centers0);
	let iterations = 0;
	for (; iterations < maxIter; ) {
		assign(points, centers, assignment);
		const u = updateCenters(points, assignment, centers);
		centers = u.centers;
		iterations++;
		if (u.moved < tol) break;
	}
	const inertia = assign(points, centers, assignment);
	return { centers, assignment, inertia, iterations };
}
