import { gaussian, mulberry32 } from './rng.ts';

export interface Dataset {
	n: number;
	x: Float64Array;
	y: Float64Array;
}

export type DatasetId = 0 | 1 | 2 | 3;

export const DATASET_NAMES: Record<DatasetId, string> = {
	0: 'Circles',
	1: 'Moons',
	2: 'XOR',
	3: 'Spiral'
};

export function makeDataset(id: DatasetId, n: number, noise: number, seed: number): Dataset {
	const rand = mulberry32(seed * 7919 + id * 104729 + 1);
	const x = new Float64Array(n * 2);
	const y = new Float64Array(n);
	const half = n / 2;
	const jitter = () => (noise > 0 ? noise * gaussian(rand) : 0);
	for (let k = 0; k < n; k++) {
		const cls = k < half ? 0 : 1;
		const m = cls === 0 ? k : k - half; // index within class
		let px = 0;
		let py = 0;
		if (id === 0) {
			const a = rand() * 2 * Math.PI;
			const r = cls === 0 ? 0.35 * Math.sqrt(rand()) : 0.8;
			px = r * Math.cos(a);
			py = r * Math.sin(a);
		} else if (id === 1) {
			const t = rand() * Math.PI;
			const s = 0.65;
			if (cls === 0) {
				px = (Math.cos(t) - 0.5) * s;
				py = (Math.sin(t) - 0.25) * s;
			} else {
				px = (1 - Math.cos(t) - 0.5) * s;
				py = (0.5 - Math.sin(t) - 0.25) * s;
			}
		} else if (id === 2) {
			// Same-sign quadrants are class 0: (+,+) and (-,-); cross-sign are class 1.
			const sx = rand() < 0.5 ? 1 : -1;
			const sy = cls === 0 ? sx : -sx;
			px = sx * 0.5 + 0.12 * gaussian(rand);
			py = sy * 0.5 + 0.12 * gaussian(rand);
		} else {
			const t = (m + rand()) / half;
			const r = 0.1 + 0.85 * t;
			const a = t * 1.5 * 2 * Math.PI + (cls === 0 ? 0 : Math.PI);
			px = r * Math.cos(a);
			py = r * Math.sin(a);
		}
		x[2 * k] = px + jitter();
		x[2 * k + 1] = py + jitter();
		y[k] = cls;
	}
	return { n, x, y };
}
