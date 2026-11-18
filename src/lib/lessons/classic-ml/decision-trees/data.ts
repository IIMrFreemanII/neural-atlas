import { makeDataset, type Dataset } from '../../../ml/datasets.ts';
import { accuracy, buildTree, treeProb } from '../../../ml/tree.ts';

/** Points in one training sample. */
export const N_TRAIN = 100;
/** Points in the fixed test set. */
export const N_TEST = 350;
/** Gaussian jitter on the moons, in makeDataset units (before the stretch below). */
export const NOISE = 0.2;
/** The depth slider's maximum. */
export const MAX_DEPTH = 10;
/** Seed of the lesson's forest. */
export const FOREST_SEED = 3;

const BASE_SEED = 205;
const TEST_SEED = 9001;

/** Centre of the noise-free moons (makeDataset id 1 spans x -0.975..0.975, y -0.4875..0.4875). */
const CX = 0;
const CY = 0;
/** Stretch that makes the noise-free moons span about -0.9..0.9 on both axes. */
const SX = 0.9 / 0.975;
const SY = 0.9 / 0.4875;

/** The query line the lesson starts from: cut axis 0 near the left edge of the data. */
export const START_CUT = { axis: 0 as const, threshold: -0.8 };

/** Plane shown by the scene, [x0, y0, x1, y1]; every training and test point lies strictly inside it. */
export const BOUNDS: [number, number, number, number] = [-1.7, -1.7, 1.7, 1.7];

/** moons stretched to fill a square. */
function fill(d: Dataset): Dataset {
	for (let k = 0; k < d.n; k++) {
		d.x[2 * k] = SX * (d.x[2 * k] - CX);
		d.x[2 * k + 1] = SY * (d.x[2 * k + 1] - CY);
	}
	return d;
}

/** The training sample. sample 0 is the lesson's main data; 1, 2, ... are the alternatives "New sample" cycles through. Deterministic. */
export function trainingData(sample: number): Dataset {
	return fill(makeDataset(1, N_TRAIN, NOISE, sample === 0 ? BASE_SEED : BASE_SEED * 100 + sample));
}

/** One fixed test set from the same distribution, never used for training. */
export const TEST: Dataset = fill(makeDataset(1, N_TEST, NOISE, TEST_SEED));

export interface DepthPoint {
	depth: number;
	train: number;
	test: number;
}

/** Training and test accuracy (0..1) of buildTree(trainingData(sample), d) for d = 0..MAX_DEPTH. */
export function depthCurve(sample: number): DepthPoint[] {
	const train = trainingData(sample);
	const out: DepthPoint[] = [];
	for (let depth = 0; depth <= MAX_DEPTH; depth++) {
		const tree = buildTree(train, depth);
		const predict = (a: number, b: number) => treeProb(tree, a, b);
		out.push({ depth, train: accuracy(train, predict), test: accuracy(TEST, predict) });
	}
	return out;
}

/** The depth (0..MAX_DEPTH) with the highest test accuracy for sample 0; on a tie the smallest depth. */
export function bestTestDepth(): number {
	const curve = depthCurve(0);
	let best = curve[0];
	for (const p of curve) if (p.test > best.test) best = p;
	return best.depth;
}
