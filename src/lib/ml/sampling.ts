import { softmax } from './vec.ts';

export interface SamplingOptions {
	/** 0 (or anything below 0.01) means greedy: the most likely token gets probability 1. */
	temperature: number;
	/** Keep only the k most likely tokens. 0 = off. */
	topK: number;
	/** Keep the smallest set of most likely tokens whose probabilities add up to at least p. 1 = off. */
	topP: number;
}

/** Indices sorted by value, largest first. Ties: lower index first. */
function rank(values: ArrayLike<number>): number[] {
	const idx = Array.from({ length: values.length }, (_, i) => i);
	idx.sort((a, b) => values[b] - values[a] || a - b);
	return idx;
}

/** Keep flags for the k largest entries (ties: lower index wins). k <= 0 or k >= length keeps all. */
export function topKMask(probs: ArrayLike<number>, k: number): boolean[] {
	const n = probs.length;
	if (k <= 0 || k >= n) return new Array<boolean>(n).fill(true);
	const keep = new Array<boolean>(n).fill(false);
	const order = rank(probs);
	for (let i = 0; i < k; i++) keep[order[i]] = true;
	return keep;
}

/** Sort descending (ties: lower index first), keep the shortest prefix whose sum is >= p, always at least one. p >= 1 keeps every entry with probability > 0. */
export function topPMask(probs: ArrayLike<number>, p: number): boolean[] {
	const n = probs.length;
	const keep = new Array<boolean>(n).fill(false);
	if (n === 0) return keep;
	const order = rank(probs);
	if (p >= 1) {
		for (let i = 0; i < n; i++) keep[i] = probs[i] > 0;
		if (!keep.some(Boolean)) keep[order[0]] = true;
		return keep;
	}
	let sum = 0;
	for (let i = 0; i < n; i++) {
		keep[order[i]] = true;
		sum += probs[order[i]];
		if (sum >= p - 1e-12) break;
	}
	return keep;
}

function renormalize(v: number[]): number[] {
	let s = 0;
	for (const x of v) s += x;
	return s > 0 ? v.map((x) => x / s) : v;
}

/** softmax(logits / temperature), then top-k, then top-p on what is left (renormalized before the top-p test), then renormalize. Removed entries are 0. */
export function distribution(logits: ArrayLike<number>, o: SamplingOptions): number[] {
	const n = logits.length;
	if (n === 0) return [];
	if (o.temperature < 0.01) {
		let best = 0;
		for (let i = 1; i < n; i++) if (logits[i] > logits[best]) best = i;
		const out = new Array<number>(n).fill(0);
		out[best] = 1;
		return out;
	}
	let p = softmax(logits, o.temperature);
	if (o.topK > 0 && o.topK < n) {
		const keep = topKMask(p, o.topK);
		p = renormalize(p.map((x, i) => (keep[i] ? x : 0)));
	}
	if (o.topP < 1) {
		const keep = topPMask(p, o.topP);
		p = renormalize(p.map((x, i) => (keep[i] ? x : 0)));
	}
	return p;
}

/** Inverse CDF in index order: the first index whose running sum exceeds u (u in [0, 1)). Falls back to the last entry with probability > 0. */
export function sampleIndex(probs: ArrayLike<number>, u: number): number {
	let sum = 0;
	let last = -1;
	for (let i = 0; i < probs.length; i++) {
		if (probs[i] <= 0) continue;
		last = i;
		sum += probs[i];
		if (sum > u) return i;
	}
	return Math.max(0, last);
}

/** Shannon entropy in bits. */
export function entropy(probs: ArrayLike<number>): number {
	let h = 0;
	for (let i = 0; i < probs.length; i++) if (probs[i] > 0) h -= probs[i] * Math.log2(probs[i]);
	return h;
}
