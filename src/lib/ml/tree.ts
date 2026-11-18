import type { Dataset } from './datasets.ts';

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

/** CART with Gini impurity. Candidate thresholds are midpoints between consecutive distinct sorted values on each axis.
 *  A node becomes a leaf when it is pure, when depth === maxDepth, or when no split lowers the weighted impurity by more than 1e-9. */
export function buildTree(data: Dataset, maxDepth: number): TreeNode {
	const all = Array.from({ length: data.n }, (_, i) => i);
	return grow(data, all, 0, Math.round(maxDepth));
}

function grow(data: Dataset, idx: number[], depth: number, maxDepth: number): TreeNode {
	const n = idx.length;
	let ones = 0;
	for (const i of idx) if (data.y[i] > 0.5) ones++;
	const node: TreeNode = { depth, n, prob: n === 0 ? 0.5 : ones / n };
	if (ones === 0 || ones === n || depth >= maxDepth || n < 2) return node;

	const parent = gini(ones, n);
	let best = Infinity;
	let bestAxis: 0 | 1 = 0;
	let bestThr = 0;
	for (const axis of [0, 1] as const) {
		const sorted = idx.slice().sort((a, b) => data.x[2 * a + axis] - data.x[2 * b + axis]);
		let leftOnes = 0;
		for (let s = 0; s < n - 1; s++) {
			if (data.y[sorted[s]] > 0.5) leftOnes++;
			const v = data.x[2 * sorted[s] + axis];
			const w = data.x[2 * sorted[s + 1] + axis];
			if (w <= v) continue;
			const nl = s + 1;
			const score = (nl * gini(leftOnes, nl) + (n - nl) * gini(ones - leftOnes, n - nl)) / n;
			if (score < best) {
				best = score;
				bestAxis = axis;
				bestThr = (v + w) / 2;
			}
		}
	}
	if (!(parent - best > 1e-9)) return node;

	const left: number[] = [];
	const right: number[] = [];
	for (const i of idx) (data.x[2 * i + bestAxis] < bestThr ? left : right).push(i);
	if (left.length === 0 || right.length === 0) return node;
	node.axis = bestAxis;
	node.threshold = bestThr;
	node.left = grow(data, left, depth + 1, maxDepth);
	node.right = grow(data, right, depth + 1, maxDepth);
	return node;
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
