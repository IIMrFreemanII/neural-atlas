/** Token id of any word outside the vocabulary. */
export const UNK = 0;
/** Token id of the end-of-story marker. */
export const END = 1;
export const UNK_TEXT = '<unk>';
export const END_TEXT = '<end>';
/** Marker that separates stories in the TinyStories file. */
export const STORY_SEPARATOR = '<|endoftext|>';

const TOKEN = /[a-z]+(?:'[a-z]+)?|[.,!?"]/g;

/** Lowercases; turns curly quotes into straight ones; returns words and the marks . , ! ? " as separate tokens. Everything else is dropped. */
export function tokenizeWords(text: string): string[] {
	const normal = text.toLowerCase().replace(/[‘’]/g, "'").replace(/[“”]/g, '"');
	return normal.match(TOKEN) ?? [];
}

export interface Vocab {
	words: string[];
	ids: Map<string, number>;
}

/** words[0] = '<unk>', words[1] = '<end>', then the (size - 2) most frequent tokens, most frequent first; ties broken by first appearance. */
export function buildVocab(stories: string[][], size: number): Vocab {
	const counts = new Map<string, number>();
	for (const story of stories) {
		for (const w of story) counts.set(w, (counts.get(w) ?? 0) + 1);
	}
	// Map keeps insertion order and Array.sort is stable, so ties stay in order of first appearance.
	const ranked = [...counts.entries()].sort((a, b) => b[1] - a[1]);
	const words = [UNK_TEXT, END_TEXT];
	for (const [w] of ranked) {
		if (words.length >= size) break;
		if (w === UNK_TEXT || w === END_TEXT) continue;
		words.push(w);
	}
	return vocabFromWords(words);
}

export function vocabFromWords(words: string[]): Vocab {
	const ids = new Map<string, number>();
	words.forEach((w, i) => ids.set(w, i));
	return { words, ids };
}

/** Token ids of a text. Unknown words become UNK. */
export function encode(vocab: Vocab, text: string): number[] {
	return tokenizeWords(text).map((w) => vocab.ids.get(w) ?? UNK);
}

/** Readable text for token ids: tokens joined by spaces, but no space before . , ! ? and END skipped. */
export function decode(vocab: Vocab, ids: ArrayLike<number>): string {
	let out = '';
	for (let i = 0; i < ids.length; i++) {
		if (ids[i] === END) continue;
		const w = vocab.words[ids[i]] ?? UNK_TEXT;
		const attach = w === '.' || w === ',' || w === '!' || w === '?';
		out += out === '' || attach ? w : ' ' + w;
	}
	return out;
}
