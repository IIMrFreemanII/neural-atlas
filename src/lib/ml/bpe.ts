/**
 * Byte pair encoding on code points, a toy version of the tokenizer idea used by GPT-2 style models.
 * Pure TypeScript: deterministic, no DOM, no GPU.
 */

export interface BpeModel {
	/** Learned merges in order. The token made by merges[i] is left + right. */
	merges: [string, string][];
	/** Token strings by id: first every distinct character of the training text in code point order, then one token per merge, in order. */
	vocab: string[];
	/** Token string -> id. */
	ids: Map<string, number>;
}

/** Split into words, each keeping one leading space (GPT-2 style): 'the cat' -> ['the', ' cat']. A space not followed by a word is its own word. Joining the result gives the input back, for any input. */
export function splitWords(text: string): string[] {
	const out: string[] = [];
	let i = 0;
	const n = text.length;
	const isSpace = (c: string) => c === ' ';
	while (i < n) {
		let j = i;
		if (isSpace(text[j]) && j + 1 < n && !isSpace(text[j + 1])) j++;
		else if (isSpace(text[j])) {
			out.push(' ');
			i++;
			continue;
		}
		while (j < n && !isSpace(text[j])) j++;
		out.push(text.slice(i, j));
		i = j;
	}
	return out;
}

/** Adjacent pair counts over all words (pairs never cross a word boundary). Sorted by count, highest first; ties keep the order of first appearance in the text. */
export function countPairs(words: string[][]): { pair: [string, string]; count: number }[] {
	const map = new Map<string, { pair: [string, string]; count: number }>();
	for (const w of words) {
		for (let i = 0; i + 1 < w.length; i++) {
			const key = w[i] + '\u0000' + w[i + 1];
			const e = map.get(key);
			if (e) e.count++;
			else map.set(key, { pair: [w[i], w[i + 1]], count: 1 });
		}
	}
	// Array.prototype.sort is stable, and Map keeps insertion (first appearance) order.
	return [...map.values()].sort((a, b) => b.count - a.count);
}

/** Replace every non-overlapping occurrence of the pair, left to right, in every word. Returns new arrays. */
export function applyMerge(words: string[][], pair: [string, string]): string[][] {
	const joined = pair[0] + pair[1];
	return words.map((w) => {
		const out: string[] = [];
		for (let i = 0; i < w.length; i++) {
			if (i + 1 < w.length && w[i] === pair[0] && w[i + 1] === pair[1]) {
				out.push(joined);
				i++;
			} else out.push(w[i]);
		}
		return out;
	});
}

/** Words of the text as character lists with the first `limit` merges of `merges` applied in order. */
export function segment(text: string, merges: [string, string][], limit: number = merges.length): string[][] {
	let words = splitWords(text).map((w) => [...w]);
	const n = Math.max(0, Math.min(merges.length, Math.floor(limit)));
	for (let i = 0; i < n; i++) words = applyMerge(words, merges[i]);
	return words;
}

/** Start from characters; repeatedly merge the most frequent pair; stop after maxMerges or when the best pair occurs fewer than 2 times. */
export function trainBpe(text: string, maxMerges: number): BpeModel {
	let words = splitWords(text).map((w) => [...w]);
	const merges: [string, string][] = [];
	const chars = [...new Set([...text])].sort((a, b) => a.codePointAt(0)! - b.codePointAt(0)!);
	const vocab = [...chars];
	while (merges.length < maxMerges) {
		const counts = countPairs(words);
		if (counts.length === 0 || counts[0].count < 2) break;
		const pair = counts[0].pair;
		merges.push(pair);
		vocab.push(pair[0] + pair[1]);
		words = applyMerge(words, pair);
	}
	const ids = new Map<string, number>();
	vocab.forEach((t, i) => {
		// A merged string can equal an earlier token only through a different split; keep the first id.
		if (!ids.has(t)) ids.set(t, i);
	});
	return { merges, vocab, ids };
}

/** Tokens of `text` using the first `limit` merges (default all). A character the model never saw stays a one-character token with id -1. */
export function encode(text: string, model: BpeModel, limit?: number): { tokens: string[]; ids: number[] } {
	const tokens = segment(text, model.merges, limit ?? model.merges.length).flat();
	return { tokens, ids: tokens.map((t) => model.ids.get(t) ?? -1) };
}
