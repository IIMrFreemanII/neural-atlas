import type { Dataset } from './datasets.ts';
import { mulberry32 } from './rng.ts';

export interface TreeNode {
	depth: number;
	n: number;
	prob: number;
	axis?: 0 | 1;
	threshold?: number;
	left?: TreeNode;
	right?: TreeNode;
}

const gini = (ones: number, n: number) => {
	if (n === 0) return 0;
	const p = ones / n;
	return 2 * p * (1 - p);
};

export interface BuildOptions {
	/** When set, each node tries one axis picked with rand() < 0.5 ? 0 : 1, and falls back to the other axis only when the picked axis has no cut that lowers the impurity (like scikit-learn's max_features=1). */
	rand?: () => number;
}

/** CART with Gini impurity. Candidate thresholds are midpoints between consecutive distinct sorted values on each axis.
 *  A node becomes a leaf when it is pure, when depth === maxDepth, or when no split lowers the weighted impurity by more than 1e-9. */
export function buildTree(data: Dataset, maxDepth: number, opts: BuildOptions = {}): TreeNode {
	const all = Array.from({ length: data.n }, (_, i) => i);
	return grow(data, all, 0, Math.round(maxDepth), opts.rand);
}

const allIdx = (data: Dataset) => Array.from({ length: data.n }, (_, i) => i);

/** Weighted Gini impurity of cutting `data` (only the points in idx, default all) at `threshold` on `axis`: points with value < threshold go left. 0 = both sides pure. An empty side counts with weight 0. */
export function splitImpurity(data: Dataset, axis: 0 | 1, threshold: number, idx: number[] = allIdx(data)): number {
	let nl = 0;
	let leftOnes = 0;
	let ones = 0;
	for (const i of idx) {
		const one = data.y[i] > 0.5 ? 1 : 0;
		ones += one;
		if (data.x[2 * i + axis] < threshold) {
			nl++;
			leftOnes += one;
		}
	}
	const n = idx.length;
	if (n === 0) return 0;
	return (nl * gini(leftOnes, nl) + (n - nl) * gini(ones - leftOnes, n - nl)) / n;
}

export interface CurvePoint {
	threshold: number;
	impurity: number;
}

/** splitImpurity at every candidate threshold on one axis (midpoints between consecutive distinct sorted values, ascending), the same candidates buildTree considers. */
export function splitCurve(data: Dataset, axis: 0 | 1, idx: number[] = allIdx(data)): CurvePoint[] {
	const n = idx.length;
	const sorted = idx.slice().sort((a, b) => data.x[2 * a + axis] - data.x[2 * b + axis]);
	let ones = 0;
	for (const i of sorted) if (data.y[i] > 0.5) ones++;
	const out: CurvePoint[] = [];
	let leftOnes = 0;
	for (let s = 0; s < n - 1; s++) {
		if (data.y[sorted[s]] > 0.5) leftOnes++;
		const v = data.x[2 * sorted[s] + axis];
		const w = data.x[2 * sorted[s + 1] + axis];
		if (w <= v) continue;
		const nl = s + 1;
		const impurity = (nl * gini(leftOnes, nl) + (n - nl) * gini(ones - leftOnes, n - nl)) / n;
		out.push({ threshold: (v + w) / 2, impurity });
	}
	return out;
}

export interface Split {
	axis: 0 | 1;
	threshold: number;
	impurity: number;
}

/** Lowest-impurity cut over the given axes (first wins ties) that beats the node's own impurity, or null. */
function bestOver(data: Dataset, idx: number[], axes: (0 | 1)[]): Split | null {
	const n = idx.length;
	let ones = 0;
	for (const i of idx) if (data.y[i] > 0.5) ones++;
	if (ones === 0 || ones === n || n < 2) return null;
	const parent = gini(ones, n);
	let best: Split | null = null;
	for (const axis of axes) {
		for (const c of splitCurve(data, axis, idx)) {
			if (best === null || c.impurity < best.impurity) best = { axis, threshold: c.threshold, impurity: c.impurity };
		}
	}
	if (best === null || !(parent - best.impurity > 1e-9)) return null;
	return best;
}

/** The cut buildTree would choose at a node holding idx (default all): lowest weighted impurity over both axes, same tie-breaking as buildTree (axis 0 first, then lowest threshold). null when no cut lowers the impurity by more than 1e-9 or the node is pure. */
export function bestSplit(data: Dataset, idx: number[] = allIdx(data)): Split | null {
	return bestOver(data, idx, [0, 1]);
}

function grow(data: Dataset, idx: number[], depth: number, maxDepth: number, rand?: () => number): TreeNode {
	const n = idx.length;
	let ones = 0;
	for (const i of idx) if (data.y[i] > 0.5) ones++;
	const node: TreeNode = { depth, n, prob: n === 0 ? 0.5 : ones / n };
	if (ones === 0 || ones === n || depth >= maxDepth || n < 2) return node;

	let split: Split | null;
	if (rand) {
		const first: 0 | 1 = rand() < 0.5 ? 0 : 1;
		split = bestOver(data, idx, [first]) ?? bestOver(data, idx, [first === 0 ? 1 : 0]);
	} else {
		split = bestOver(data, idx, [0, 1]);
	}
	if (!split) return node;

	const left: number[] = [];
	const right: number[] = [];
	for (const i of idx) (data.x[2 * i + split.axis] < split.threshold ? left : right).push(i);
	if (left.length === 0 || right.length === 0) return node;
	node.axis = split.axis;
	node.threshold = split.threshold;
	node.left = grow(data, left, depth + 1, maxDepth, rand);
	node.right = grow(data, right, depth + 1, maxDepth, rand);
	return node;
}

/** n points drawn with replacement from data (n = data.n). */
export function bootstrap(data: Dataset, rand: () => number): Dataset {
	const x = new Float64Array(data.n * 2);
	const y = new Float64Array(data.n);
	for (let k = 0; k < data.n; k++) {
		const i = Math.min(data.n - 1, Math.floor(rand() * data.n));
		x[2 * k] = data.x[2 * i];
		x[2 * k + 1] = data.x[2 * i + 1];
		y[k] = data.y[i];
	}
	return { n: data.n, x, y };
}

/** nTrees trees for a random forest. Tree i: rand_i = mulberry32(seed * 1009 + i); sample = bootstrap(data, rand_i); tree = buildTree(sample, maxDepth, { rand: rand_i }). Tree i is the same whatever nTrees is, so growing the forest from 10 to 11 trees keeps the first 10. */
export function buildForest(data: Dataset, nTrees: number, maxDepth: number, seed: number): TreeNode[] {
	const trees: TreeNode[] = [];
	for (let i = 0; i < nTrees; i++) {
		const rand = mulberry32(seed * 1009 + i);
		trees.push(buildTree(bootstrap(data, rand), maxDepth, { rand }));
	}
	return trees;
}

/** Mean of treeProb over the trees (0.5 for an empty list). */
export function forestProb(trees: TreeNode[], x0: number, x1: number): number {
	if (trees.length === 0) return 0.5;
	let s = 0;
	for (const t of trees) s += treeProb(t, x0, x1);
	return s / trees.length;
}

export function treeProb(root: TreeNode, x0: number, x1: number): number {
	let node = root;
	while (node.left && node.right && node.axis !== undefined && node.threshold !== undefined) {
		const v = node.axis === 0 ? x0 : x1;
		node = v < node.threshold ? node.left : node.right;
	}
	return node.prob;
}

export function countLeaves(root: TreeNode): number {
	if (!root.left || !root.right) return 1;
	return countLeaves(root.left) + countLeaves(root.right);
}

export interface SplitSegment {
	axis: 0 | 1;
	threshold: number;
	lo: number;
	hi: number;
	depth: number;
}

/** One segment per internal node, clipped to that node's region inside bounds = [x0, y0, x1, y1]. */
export function treeSplits(root: TreeNode, bounds: [number, number, number, number]): SplitSegment[] {
	const out: SplitSegment[] = [];
	const walk = (node: TreeNode, b: [number, number, number, number]) => {
		if (!node.left || !node.right || node.axis === undefined || node.threshold === undefined) return;
		const t = node.threshold;
		if (node.axis === 0) {
			out.push({ axis: 0, threshold: t, lo: b[1], hi: b[3], depth: node.depth });
			walk(node.left, [b[0], b[1], t, b[3]]);
			walk(node.right, [t, b[1], b[2], b[3]]);
		} else {
			out.push({ axis: 1, threshold: t, lo: b[0], hi: b[2], depth: node.depth });
			walk(node.left, [b[0], b[1], b[2], t]);
			walk(node.right, [b[0], t, b[2], b[3]]);
		}
	};
	walk(root, bounds);
	return out;
}

/** The nodes visited from the root to the leaf that (x0, x1) falls in, root first. */
export function treePath(root: TreeNode, x0: number, x1: number): TreeNode[] {
	const path = [root];
	let node = root;
	while (node.left && node.right && node.axis !== undefined && node.threshold !== undefined) {
		const v = node.axis === 0 ? x0 : x1;
		node = v < node.threshold ? node.left : node.right;
		path.push(node);
	}
	return path;
}

export interface LeafBox {
	leaf: TreeNode;
	/** index of the leaf in left-to-right order */
	index: number;
	box: [x0: number, y0: number, x1: number, y1: number];
}

/** Every leaf with the rectangle of the plane it covers inside bounds = [x0, y0, x1, y1], leaves in left-to-right order (left child = value < threshold). */
export function treeBoxes(root: TreeNode, bounds: [number, number, number, number]): LeafBox[] {
	const out: LeafBox[] = [];
	const walk = (node: TreeNode, b: [number, number, number, number]) => {
		if (!node.left || !node.right || node.axis === undefined || node.threshold === undefined) {
			out.push({ leaf: node, index: out.length, box: b });
			return;
		}
		const t = node.threshold;
		if (node.axis === 0) {
			walk(node.left, [b[0], b[1], t, b[3]]);
			walk(node.right, [t, b[1], b[2], b[3]]);
		} else {
			walk(node.left, [b[0], b[1], b[2], t]);
			walk(node.right, [b[0], t, b[2], b[3]]);
		}
	};
	walk(root, bounds);
	return out;
}

export interface LaidOutNode {
	node: TreeNode;
	/** 0..1, leaves evenly spaced in left-to-right order (one leaf -> 0.5; k leaves -> (i + 0.5) / k), an internal node at the mean x of its two children */
	x: number;
	/** node.depth */
	depth: number;
	/** index of the parent in the returned array, -1 for the root */
	parent: number;
}

/** Positions for drawing the tree as a diagram, in pre-order (root first, then left subtree, then right). */
export function layoutTree(root: TreeNode): LaidOutNode[] {
	const k = countLeaves(root);
	const out: LaidOutNode[] = [];
	let leaf = 0;
	const walk = (node: TreeNode, parent: number): number => {
		const at = out.length;
		out.push({ node, x: 0, depth: node.depth, parent });
		if (!node.left || !node.right) {
			out[at].x = (leaf++ + 0.5) / k;
		} else {
			const l = walk(node.left, at);
			const r = walk(node.right, at);
			out[at].x = (l + r) / 2;
		}
		return out[at].x;
	};
	walk(root, -1);
	return out;
}

/** Share of the points of data that predict(x0, x1) >= 0.5 classifies as their label, 0..1. */
export function accuracy(data: Dataset, predict: (x0: number, x1: number) => number): number {
	if (data.n === 0) return 0;
	let ok = 0;
	for (let k = 0; k < data.n; k++) {
		if ((predict(data.x[2 * k], data.x[2 * k + 1]) >= 0.5 ? 1 : 0) === data.y[k]) ok++;
	}
	return ok / data.n;
}
