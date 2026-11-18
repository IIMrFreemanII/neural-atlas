import { strokePath, strokeSegment } from '../../../ml/raster.ts';

/** The image is N x N pixels. */
export const N = 24;

/** Row-major 3 x 3 kernels with integer weights. */
export const KERNELS: { name: string; k: number[] }[] = [
	{ name: 'Identity', k: [0, 0, 0, 0, 1, 0, 0, 0, 0] },
	{ name: 'Blur', k: [1, 1, 1, 1, 1, 1, 1, 1, 1] },
	{ name: 'Sharpen', k: [0, -1, 0, -1, 5, -1, 0, -1, 0] },
	{ name: 'Vertical edges', k: [-1, 0, 1, -2, 0, 2, -1, 0, 1] },
	{ name: 'Horizontal edges', k: [-1, -2, -1, 0, 0, 0, 1, 2, 1] },
	{ name: 'Outline', k: [-1, -1, -1, -1, 8, -1, -1, -1, -1] }
];

const clamp01 = (v: number) => Math.min(1, Math.max(0, v));

function shapes(): Float32Array {
	const d = new Float32Array(N * N);
	// filled rectangle, 9 x 7 pixels, upper left
	for (let r = 2; r < 9; r++) for (let c = 2; c < 11; c++) d[r * N + c] = 1;
	// filled disc with a soft one-pixel edge, lower right
	const cx = 17.5;
	const cy = 17.5;
	const rad = 4.5;
	for (let r = 0; r < N; r++) {
		for (let c = 0; c < N; c++) {
			const dist = Math.hypot(c + 0.5 - cx, r + 0.5 - cy);
			d[r * N + c] = Math.max(d[r * N + c], clamp01(rad + 0.5 - dist));
		}
	}
	// thin diagonal stroke between them
	strokeSegment(d, N, 11.5, 9.5, 14.2, 12.4, 0.5);
	return d;
}

function digit(): Float32Array {
	const d = new Float32Array(N * N);
	strokePath(
		d,
		N,
		[
			[16.5, 4],
			[8.5, 4],
			[8, 11]
		],
		1
	);
	strokePath(
		d,
		N,
		[
			[8, 11],
			[11, 9.8],
			[14.5, 10.5],
			[16.5, 13.5],
			[16.5, 17],
			[14, 20],
			[10.5, 20.5],
			[7, 19]
		],
		1
	);
	return d;
}

function gradient(): Float32Array {
	const d = new Float32Array(N * N);
	for (let r = 0; r < N; r++) {
		for (let c = 0; c < N; c++) {
			const v = 0.05 + 0.9 * (c / (N - 1));
			d[r * N + c] = c >= 10 && c < 14 ? v * 0.2 : v;
		}
	}
	return d;
}

export const IMAGES: { name: string; make(): Float32Array }[] = [
	{ name: 'Shapes', make: shapes },
	{ name: 'Digit', make: digit },
	{ name: 'Gradient', make: gradient },
	{ name: 'Blank', make: () => new Float32Array(N * N) }
];
