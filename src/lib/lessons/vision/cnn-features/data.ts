import { drawShape, SHAPE_KINDS, shapePaths, strokePath } from '../../../ml/raster.ts';

export const CLASS_NAMES = ['circle', 'square', 'triangle', 'cross'];

const N = 16;

/** Rotation that makes each clean sample look like the usual circle, square (flat sides), triangle (point up), X. */
const SAMPLE_ROT = [0, 0, 0, Math.PI / 4];

/** Four clean drawings in SHAPE_KINDS order. */
export const SAMPLES: Float32Array[] = SHAPE_KINDS.map((kind, i) => drawShape(kind, N, 8, 8, 5.5, SAMPLE_ROT[i], 0.7));

const between = (rand: () => number, a: number, b: number) => a + (b - a) * rand();

/**
 * One random hand-drawn-looking shape: random kind, position, size, rotation and stroke width, a wobbly outline,
 * and a little pixel noise. The same generator trains the network, tests it and feeds the lesson.
 */
export function randomShape(rand: () => number): { image: Float32Array; label: number } {
	const label = Math.min(SHAPE_KINDS.length - 1, Math.floor(rand() * SHAPE_KINDS.length));
	const kind = SHAPE_KINDS[label];
	const cx = 8 + between(rand, -1.5, 1.5);
	const cy = 8 + between(rand, -1.5, 1.5);
	const r = between(rand, 4, 6.5);
	const rot = rand() * Math.PI * 2;
	const halfWidth = between(rand, 0.5, 0.95);
	const paths = shapePaths(kind, cx, cy, r, rot);

	if (kind === 'circle') {
		// Smooth radius modulation, at most 8%.
		const p2 = rand() * Math.PI * 2;
		const p3 = rand() * Math.PI * 2;
		const a2 = between(rand, 0, 0.05);
		const a3 = between(rand, 0, 0.03);
		for (const path of paths) {
			for (const pt of path) {
				const ang = Math.atan2(pt[1] - cy, pt[0] - cx);
				const f = 1 + a2 * Math.cos(2 * ang + p2) + a3 * Math.cos(3 * ang + p3);
				pt[0] = cx + (pt[0] - cx) * f;
				pt[1] = cy + (pt[1] - cy) * f;
			}
		}
	} else {
		if (kind === 'cross') {
			// The two strokes may differ in length by up to 25%.
			const s0 = between(rand, 0.89, 1.11);
			const s1 = between(rand, 0.89, 1.11);
			paths.forEach((path, i) => {
				const s = i === 0 ? s0 : s1;
				for (const pt of path) {
					pt[0] = cx + (pt[0] - cx) * s;
					pt[1] = cy + (pt[1] - cy) * s;
				}
			});
		}
		// Every vertex moves by up to half a pixel each way. The closing vertex of a ring moves with the first.
		for (const path of paths) {
			const closed = path.length > 2;
			const last = path.length - 1;
			for (let i = 0; i < (closed ? last : path.length); i++) {
				path[i][0] += between(rand, -0.5, 0.5);
				path[i][1] += between(rand, -0.5, 0.5);
			}
			if (closed) path[last] = [path[0][0], path[0][1]];
		}
	}

	const image = new Float32Array(N * N);
	for (const path of paths) strokePath(image, N, path, halfWidth);
	for (let i = 0; i < image.length; i++) image[i] = Math.min(1, Math.max(0, image[i] + between(rand, -0.05, 0.05)));
	return { image, label };
}
