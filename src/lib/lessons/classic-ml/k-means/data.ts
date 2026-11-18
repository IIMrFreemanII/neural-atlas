import { gaussian, mulberry32 } from '../../../ml/rng.ts';

export const N = 12000;

interface Blob {
	x: number;
	y: number;
	sd: number;
	/** Relative share of the points. */
	w: number;
}

const THREE: Blob[] = [
	{ x: -0.8, y: -0.6, sd: 0.16, w: 1 },
	{ x: 0.85, y: -0.5, sd: 0.16, w: 1 },
	{ x: 0, y: 0.85, sd: 0.16, w: 1 }
];

const FIVE: Blob[] = [
	{ x: -0.85, y: 0.7, sd: 0.2, w: 1.1 },
	{ x: 0.0, y: 0.75, sd: 0.1, w: 0.6 },
	{ x: 0.85, y: 0.55, sd: 0.15, w: 0.9 },
	{ x: -0.6, y: -0.7, sd: 0.1, w: 0.7 },
	{ x: 0.55, y: -0.6, sd: 0.26, w: 1.7 }
];

function blobs(spec: Blob[], rand: () => number): Float64Array {
	const out = new Float64Array(N * 2);
	const total = spec.reduce((s, b) => s + b.w, 0);
	let i = 0;
	spec.forEach((b, bi) => {
		const n = bi === spec.length - 1 ? N - i : Math.round((b.w / total) * N);
		for (let k = 0; k < n && i < N; k++, i++) {
			out[i * 2] = b.x + b.sd * gaussian(rand);
			out[i * 2 + 1] = b.y + b.sd * gaussian(rand);
		}
	});
	return out;
}

function moons(rand: () => number): Float64Array {
	const out = new Float64Array(N * 2);
	const s = 0.8;
	for (let i = 0; i < N; i++) {
		const t = rand() * Math.PI;
		let x: number;
		let y: number;
		if (i % 2 === 0) {
			x = Math.cos(t);
			y = Math.sin(t);
		} else {
			x = 1 - Math.cos(t);
			y = 0.45 - Math.sin(t);
		}
		out[i * 2] = (x - 0.5 + 0.075 * gaussian(rand)) * s * 1.05;
		out[i * 2 + 1] = (y - 0.22 + 0.075 * gaussian(rand)) * s * 1.05;
	}
	return out;
}

function noise(rand: () => number): Float64Array {
	const out = new Float64Array(N * 2);
	for (let i = 0; i < N * 2; i++) out[i] = (rand() * 2 - 1) * 1.3;
	return out;
}

/** xy pairs (N * 2 floats) inside roughly [-1.3, 1.3]². Deterministic.
 *  0 = three round blobs of equal size; 1 = five blobs with different sizes and spreads;
 *  2 = two long curved bands (interleaved moons), where k-means does badly; 3 = uniform noise with no structure. */
export function makeCloud(id: number, seed: number): Float64Array {
	const rand = mulberry32(seed * 7919 + id * 104729 + 17);
	let pts: Float64Array;
	if (id === 1) pts = blobs(FIVE, rand);
	else if (id === 2) pts = moons(rand);
	else if (id === 3) pts = noise(rand);
	else pts = blobs(THREE, rand);
	// Shuffle so that point order carries no cluster information (the GPU subsample and drift rely on it).
	for (let i = N - 1; i > 0; i--) {
		const j = Math.floor(rand() * (i + 1));
		const x = pts[i * 2];
		const y = pts[i * 2 + 1];
		pts[i * 2] = pts[j * 2];
		pts[i * 2 + 1] = pts[j * 2 + 1];
		pts[j * 2] = x;
		pts[j * 2 + 1] = y;
	}
	for (let i = 0; i < N * 2; i++) pts[i] = Math.max(-1.35, Math.min(1.35, pts[i]));
	return pts;
}
