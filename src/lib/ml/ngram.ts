/** Lowercases, then splits into words (letters and apostrophes) and the punctuation marks . , ? ! as separate tokens. */
export function tokenizeWords(text: string): string[] {
	return text.toLowerCase().match(/[a-z']+|[.,?!]/g) ?? [];
}

export interface NgramModel {
	/** Distinct tokens, most frequent first (ties: first appearance). */
	vocab: string[];
	ids: Map<string, number>;
	/** Count of each token. */
	uni: Float64Array;
	/** Previous token -> next token -> count. */
	bi: Map<number, Map<number, number>>;
	/** (prev2 * vocab.length + prev1) -> next token -> count. */
	tri: Map<number, Map<number, number>>;
	total: number;
}

function bump(m: Map<number, Map<number, number>>, key: number, next: number): void {
	let row = m.get(key);
	if (!row) {
		row = new Map();
		m.set(key, row);
	}
	row.set(next, (row.get(next) ?? 0) + 1);
}

export function buildNgram(tokens: string[]): NgramModel {
	const counts = new Map<string, number>();
	const first = new Map<string, number>();
	tokens.forEach((t, i) => {
		counts.set(t, (counts.get(t) ?? 0) + 1);
		if (!first.has(t)) first.set(t, i);
	});
	const vocab = [...counts.keys()].sort((a, b) => counts.get(b)! - counts.get(a)! || first.get(a)! - first.get(b)!);
	const ids = new Map(vocab.map((t, i) => [t, i]));
	const V = vocab.length;
	const uni = new Float64Array(V);
	const bi = new Map<number, Map<number, number>>();
	const tri = new Map<number, Map<number, number>>();
	const seq = tokens.map((t) => ids.get(t)!);
	for (let i = 0; i < seq.length; i++) {
		uni[seq[i]]++;
		if (i >= 1) bump(bi, seq[i - 1], seq[i]);
		if (i >= 2) bump(tri, seq[i - 2] * V + seq[i - 1], seq[i]);
	}
	return { vocab, ids, uni, bi, tri, total: seq.length };
}

const W_TRI = 0.7;
const W_BI = 0.22;
const W_UNI = 0.07;
const W_UNIFORM = 0.01;

/**
 * Log-probability of every vocabulary entry as the next token, given the context (token ids; the last two are used).
 * p = 0.70 trigram + 0.22 bigram + 0.07 unigram + 0.01 uniform. When the trigram (or bigram) context was never
 * seen, its share is given to the remaining parts in proportion to their weights.
 */
export function nextLogits(model: NgramModel, context: readonly number[]): number[] {
	const V = model.vocab.length;
	const n = context.length;
	const triRow = n >= 2 ? model.tri.get(context[n - 2] * V + context[n - 1]) : undefined;
	const biRow = n >= 1 ? model.bi.get(context[n - 1]) : undefined;
	const wTri = triRow ? W_TRI : 0;
	const wBi = biRow ? W_BI : 0;
	const scale = 1 / (wTri + wBi + W_UNI + W_UNIFORM);
	let triTotal = 0;
	let biTotal = 0;
	if (triRow) for (const c of triRow.values()) triTotal += c;
	if (biRow) for (const c of biRow.values()) biTotal += c;
	const out = new Array<number>(V);
	for (let i = 0; i < V; i++) {
		let p = W_UNI * (model.uni[i] / model.total) + W_UNIFORM / V;
		if (triRow) p += W_TRI * ((triRow.get(i) ?? 0) / triTotal);
		if (biRow) p += W_BI * ((biRow.get(i) ?? 0) / biTotal);
		out[i] = Math.log(p * scale);
	}
	return out;
}
