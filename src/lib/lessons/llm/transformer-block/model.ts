import { layerNorm, positional, randomBlock, runBlock, type BlockTrace, type BlockWeights } from '../../../ml/block.ts';
import { gaussian, mulberry32 } from '../../../ml/rng.ts';
import { dot, softmax, type Mat, type Vec } from '../../../ml/vec.ts';

export const TOKENS = ['the', 'cat', 'sat', 'on', 'the'];
export const D = 8;
export const MAX_DEPTH = 12;

const WORD_SEED = 21;
const BLOCK_SEED = 100;

function buildInput(): Mat {
	const words = new Map<string, Vec>();
	const rand = mulberry32(WORD_SEED);
	for (const t of TOKENS) {
		if (!words.has(t)) words.set(t, Array.from({ length: D }, () => 0.8 * gaussian(rand)));
	}
	const pos = positional(TOKENS.length, D);
	return TOKENS.map((t, i) => words.get(t)!.map((v, k) => v + 0.5 * pos[i][k]));
}

export const INPUT: Mat = buildInput();

export const BLOCKS: BlockWeights[] = Array.from({ length: MAX_DEPTH }, (_, i) => randomBlock(D, BLOCK_SEED + i));

function buildTraces(): BlockTrace[] {
	const traces: BlockTrace[] = [];
	let x = INPUT;
	for (const w of BLOCKS) {
		const t = runBlock(x, w, true);
		traces.push(t);
		x = t.out;
	}
	return traces;
}

export const TRACES: BlockTrace[] = buildTraces();

export const CANDIDATES = ['mat', 'floor', 'sofa', 'bed', 'roof', 'table', 'moon', 'dog'];
export const TARGETS = [0.46, 0.2, 0.12, 0.08, 0.06, 0.04, 0.03, 0.01];

const LAST = TOKENS.length - 1;

/** Next-token probabilities after `depth` blocks, from the hand-set rank-one output matrix. */
export function nextTokenProbs(depth: number): number[] {
	const d = Math.min(MAX_DEPTH, Math.max(1, Math.round(depth)));
	const h12 = layerNorm(TRACES[MAX_DEPTH - 1].out[LAST]);
	const h = layerNorm(TRACES[d - 1].out[LAST]);
	const k = Math.max(0, dot(h12, h) / dot(h12, h12));
	return softmax(TARGETS.map((p) => Math.log(p) * k));
}
