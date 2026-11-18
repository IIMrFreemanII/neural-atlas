import { encode } from '../../../ml/autoencoder.ts';
import type { MLP } from '../../../ml/mlp.ts';
import { SHAPE_KINDS, shapePaths, strokePath } from '../../../ml/raster.ts';
import { mulberry32 } from '../../../ml/rng.ts';

const N = 16;
/** The upright pose of each shape, as in the CNN lesson's clean samples: flat-sided square, point-up triangle, X. */
const UPRIGHT = [0, 0, 0, Math.PI / 4];

const between = (rand: () => number, a: number, b: number) => a + (b - a) * rand();

/**
 * One random hand-drawn-looking shape for this lesson: centered, upright and of similar size, with a wobbly
 * outline, varied stroke width and a little pixel noise. Unlike the CNN lesson's drawings, position, size and
 * rotation barely vary. With only two numbers to spend, an autoencoder spends them on whatever varies most,
 * and here that is which shape it is, so the shapes form clusters on the map. The same generator trains the
 * network (through scripts/export-shapes.ts), tests it and fills the lesson's map.
 */
export function centeredShape(rand: () => number): { image: Float32Array; label: number } {
	const label = Math.min(SHAPE_KINDS.length - 1, Math.floor(rand() * SHAPE_KINDS.length));
	const kind = SHAPE_KINDS[label];
	const cx = 8 + between(rand, -0.6, 0.6);
	const cy = 8 + between(rand, -0.6, 0.6);
	const r = between(rand, 4.8, 6.2);
	const rot = UPRIGHT[label] + between(rand, -0.2, 0.2);
	const halfWidth = between(rand, 0.55, 0.9);
	const paths = shapePaths(kind, cx, cy, r, rot);
	if (kind === 'circle') {
		// Smooth radius modulation, at most 8%.
		const p2 = rand() * Math.PI * 2;
		const p3 = rand() * Math.PI * 2;
		const a2 = between(rand, 0, 0.05);
		const a3 = between(rand, 0, 0.03);
		for (const pt of paths[0]) {
			const ang = Math.atan2(pt[1] - cy, pt[0] - cx);
			const f = 1 + a2 * Math.cos(2 * ang + p2) + a3 * Math.cos(3 * ang + p3);
			pt[0] = cx + (pt[0] - cx) * f;
			pt[1] = cy + (pt[1] - cy) * f;
		}
	} else {
		// Every vertex moves by up to 0.4 pixels each way. The closing vertex of a ring moves with the first.
		for (const path of paths) {
			const closed = path.length > 2;
			const last = path.length - 1;
			for (let i = 0; i < (closed ? last : path.length); i++) {
				path[i][0] += between(rand, -0.4, 0.4);
				path[i][1] += between(rand, -0.4, 0.4);
			}
			if (closed) path[last] = [path[0][0], path[0][1]];
		}
	}
	const image = new Float32Array(N * N);
	for (const path of paths) strokePath(image, N, path, halfWidth);
	for (let i = 0; i < image.length; i++) image[i] = Math.min(1, Math.max(0, image[i] + between(rand, -0.04, 0.04)));
	return { image, label };
}

export interface MapPoint {
	x: number;
	y: number;
	label: number;
}

/** How many encoded drawings the lesson's map shows, and the seed that generates them. */
export const MAP_COUNT = 800;
export const MAP_SEED = 31;

/** Encode `count` drawings from centeredShape(mulberry32(seed)). Deterministic. */
export function mapPoints(net: MLP, count = MAP_COUNT, seed = MAP_SEED): MapPoint[] {
	const rand = mulberry32(seed);
	const points: MapPoint[] = [];
	for (let i = 0; i < count; i++) {
		const s = centeredShape(rand);
		const [x, y] = encode(net, s.image);
		points.push({ x, y, label: s.label });
	}
	return points;
}

/** The most common label among the k map points nearest to (x, y) (Euclidean). Ties go to the label of the nearest point among the tied labels. */
export function nearestLabel(points: readonly MapPoint[], x: number, y: number, k = 10): number {
	const near = points
		.map((p) => ({ d: (p.x - x) ** 2 + (p.y - y) ** 2, label: p.label }))
		.sort((a, b) => a.d - b.d)
		.slice(0, k);
	const votes = new Map<number, number>();
	for (const p of near) votes.set(p.label, (votes.get(p.label) ?? 0) + 1);
	const top = Math.max(...votes.values());
	// near is sorted by distance, so the first point whose label has the top count is the nearest among the tied.
	return near.find((p) => votes.get(p.label) === top)!.label;
}
