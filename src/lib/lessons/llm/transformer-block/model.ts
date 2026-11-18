import { readout, trace, type LayerTrace } from '../../../ml/gpt.ts';
import { softmax } from '../../../ml/vec.ts';
import { encode } from '../../../ml/wordtok.ts';
import type { Lm } from '../shared/lm.ts';

/** Chosen because the right next word depends on the first one: 'she ... with her'. */
export const TEXT = 'she loved to play with';

export interface BlockModel {
	/** Token texts, 5 of them. */
	tokens: string[];
	/** Numbers per vector (64). */
	d: number;
	/** Number of blocks (4). */
	depth: number;
	/** Heads per block (4). */
	heads: number;
	/** One trace per block, straight from trace(). */
	layers: LayerTrace[];
	/** attention[l][i][j]: how much token i gathers from token j in block l, averaged over that block's heads. Rows sum to 1. */
	attention: number[][][];
	/** The 8 next words the full model finds most likely, best first (never '<unk>' or '<end>'). */
	candidates: string[];
	/** probs[k - 1][c]: probability of candidates[c] when the stream is read after k blocks. Real probabilities over the whole vocabulary, so a row sums to less than 1. */
	probs: number[][];
	/** rest[k - 1]: 1 minus the sum of probs[k - 1]: the share of all other words. */
	rest: number[];
	/** Largest absolute value in any stream vector (x, mid, out of any block), for colour scales. */
	maxStream: number;
	/** Largest absolute value in any hidden vector. */
	maxHidden: number;
}

const CANDIDATES = 8;

/** Next-word probabilities after reading the stream vector, with '<unk>' ruled out. */
function distribution(lm: Lm, vector: ArrayLike<number>): number[] {
	const logits = Array.from(readout(lm.model, vector));
	logits[0] = -Infinity;
	return softmax(logits);
}

export function buildBlockModel(lm: Lm): BlockModel {
	const ids = encode(lm.vocab, TEXT);
	const tr = trace(lm.model, ids);
	const { d, layers: depth, heads } = lm.model.config;
	const n = ids.length;
	const last = n - 1;

	const attention = tr.layers.map((l) =>
		Array.from({ length: n }, (_, i) =>
			Array.from({ length: n }, (_, j) => l.heads.reduce((s, h) => s + h[i][j], 0) / heads)
		)
	);

	// Candidates come from the full model's own prediction (the last block's output).
	const full = distribution(lm, tr.layers[depth - 1].out[last]);
	const order = Array.from(full.keys())
		.filter((id) => id !== 0 && id !== 1)
		.sort((a, b) => full[b] - full[a])
		.slice(0, CANDIDATES);
	const candidates = order.map((id) => lm.vocab.words[id]);

	const probs: number[][] = [];
	const rest: number[] = [];
	for (let k = 0; k < depth; k++) {
		const p = distribution(lm, tr.layers[k].out[last]);
		const row = order.map((id) => p[id]);
		probs.push(row);
		rest.push(Math.max(0, 1 - row.reduce((a, b) => a + b, 0)));
	}

	let maxStream = 0;
	let maxHidden = 0;
	for (const l of tr.layers) {
		for (const m of [l.x, l.mid, l.out]) for (const row of m) for (const v of row) maxStream = Math.max(maxStream, Math.abs(v));
		for (const row of l.hidden) for (const v of row) maxHidden = Math.max(maxHidden, Math.abs(v));
	}

	return {
		tokens: ids.map((id) => lm.vocab.words[id]),
		d,
		depth,
		heads,
		layers: tr.layers,
		attention,
		candidates,
		probs,
		rest,
		maxStream,
		maxHidden
	};
}
