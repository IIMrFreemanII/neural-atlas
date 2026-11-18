import { gaussian, mulberry32 } from '../../../ml/rng.ts';
import { makeDataset, type Dataset, type DatasetId } from '../../../ml/datasets.ts';

export const N_POINTS = 200;

/** 0 = two blobs (linearly separable), 1 = moons, 2 = circles, 3 = XOR. Deterministic in seed. */
export function makeBoundaryData(id: number, noise: number, seed: number): Dataset {
	const kind = Math.max(0, Math.min(3, Math.round(id)));
	if (kind === 0) {
		const rand = mulberry32(seed * 7919 + 31);
		const spread = 0.2 + noise;
		const n = N_POINTS;
		const x = new Float64Array(n * 2);
		const y = new Float64Array(n);
		for (let k = 0; k < n; k++) {
			const cls = k < n / 2 ? 0 : 1;
			const s = cls === 1 ? 1 : -1;
			x[2 * k] = s * 0.45 + spread * gaussian(rand);
			x[2 * k + 1] = s * 0.35 + spread * gaussian(rand);
			y[k] = cls;
		}
		return { n, x, y };
	}
	const map: Record<number, DatasetId> = { 1: 1, 2: 0, 3: 2 };
	return makeDataset(map[kind], N_POINTS, noise, seed);
}
