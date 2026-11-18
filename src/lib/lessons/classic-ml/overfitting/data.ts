import { gaussian, mulberry32 } from '../../../ml/rng.ts';

/** The true pattern on [-1, 1]: 0.6 * sin(2.6 * x) + 0.25 * x. */
export function truth(x: number): number {
	return 0.6 * Math.sin(2.6 * x) + 0.25 * x;
}

export interface Sample {
	x: Float64Array;
	y: Float64Array;
}

/** n points, x evenly stratified over [-0.97, 0.97] (one jittered point per stratum), y = truth(x) + noise * gaussian. Deterministic in seed. Sorted by x. */
export function makeSample(n: number, noise: number, seed: number): Sample {
	const rand = mulberry32(seed);
	const pts: Array<[number, number]> = [];
	for (let i = 0; i < n; i++) {
		const x = -0.97 + 1.94 * ((i + 0.15 + 0.7 * rand()) / n);
		pts.push([x, truth(x) + noise * gaussian(rand)]);
	}
	pts.sort((a, b) => a[0] - b[0]);
	return { x: Float64Array.from(pts, (p) => p[0]), y: Float64Array.from(pts, (p) => p[1]) };
}
