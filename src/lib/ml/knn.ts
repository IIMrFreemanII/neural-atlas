import type { Dataset } from './datasets.ts';

/** Indices of the k nearest training points to (x0, x1), nearest first. k is clamped to [1, n]. Ties broken by lower index. */
export function knnNeighbors(data: Dataset, k: number, x0: number, x1: number): number[] {
	const kk = Math.max(1, Math.min(data.n, Math.round(k)));
	const idx: number[] = [];
	const dist: number[] = [];
	for (let i = 0; i < data.n; i++) {
		const dx = data.x[2 * i] - x0;
		const dy = data.x[2 * i + 1] - x1;
		const d = dx * dx + dy * dy;
		if (idx.length === kk && d >= dist[kk - 1]) continue;
		// insertion position: after all entries with distance <= d (keeps lower index first on ties)
		let j = idx.length < kk ? idx.length : kk - 1;
		if (idx.length < kk) {
			idx.push(i);
			dist.push(d);
		} else {
			idx[j] = i;
			dist[j] = d;
		}
		while (j > 0 && dist[j - 1] > d) {
			dist[j] = dist[j - 1];
			idx[j] = idx[j - 1];
			j--;
			dist[j] = d;
			idx[j] = i;
		}
	}
	return idx;
}

/** Fraction of class 1 among the k nearest. */
export function knnProb(data: Dataset, k: number, x0: number, x1: number): number {
	const nb = knnNeighbors(data, k, x0, x1);
	let ones = 0;
	for (const i of nb) if (data.y[i] > 0.5) ones++;
	return ones / nb.length;
}
