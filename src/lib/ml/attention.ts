import { dot, softmax, type Mat, type Vec } from './vec.ts';

export interface AttentionResult {
	/** scores[i][j] = scale * (query i . key j). Masked entries are -Infinity. */
	scores: Mat;
	/** weights[i] = softmax(scores[i]). Every row sums to 1. Masked entries are 0. */
	weights: Mat;
	/** out[i] = sum over j of weights[i][j] * values[j]. */
	out: Mat;
}

export interface AttentionOptions {
	/** Token i may only attend to tokens j <= i (decoder-style mask). Default false. */
	causal?: boolean;
	/** Multiplies every score before the softmax. Default 1 / sqrt(key length). */
	scale?: number;
}

/**
 * Scaled dot-product attention over one sequence.
 * queries, keys and values have one row per token. Queries and keys share a length; values may differ.
 */
export function attention(queries: Mat, keys: Mat, values: Mat, opts: AttentionOptions = {}): AttentionResult {
	const n = queries.length;
	const dk = n > 0 ? queries[0].length : 1;
	const scale = opts.scale ?? 1 / Math.sqrt(dk);
	const dv = n > 0 ? values[0].length : 0;
	const scores: Mat = [];
	const weights: Mat = [];
	const out: Mat = [];
	for (let i = 0; i < n; i++) {
		const row: Vec = new Array<number>(n);
		for (let j = 0; j < n; j++) {
			row[j] = opts.causal && j > i ? -Infinity : scale * dot(queries[i], keys[j]);
		}
		const w = softmax(row);
		const o: Vec = new Array<number>(dv).fill(0);
		for (let j = 0; j < n; j++) {
			if (w[j] === 0) continue;
			for (let d = 0; d < dv; d++) o[d] += w[j] * values[j][d];
		}
		scores.push(row);
		weights.push(w);
		out.push(o);
	}
	return { scores, weights, out };
}
