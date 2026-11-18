import { describe, expect, it } from 'vitest';
import { buildVocab, decode, encode, END, tokenizeWords, UNK, vocabFromWords } from './wordtok.ts';

describe('tokenizeWords', () => {
	it('lowercases, keeps contractions, straightens quotes, drops digits and dashes', () => {
		const text = 'The Cat didn’t see 3 dogs — “Why?”, she said.';
		expect(tokenizeWords(text)).toEqual(['the', 'cat', "didn't", 'see', 'dogs', '"', 'why', '?', '"', ',', 'she', 'said', '.']);
	});
	it('returns an empty list for text without words', () => {
		expect(tokenizeWords('123 - 456')).toEqual([]);
	});
});

describe('buildVocab', () => {
	const stories = [
		['b', 'a', 'a', 'c'],
		['c', 'b', 'd', 'a']
	];
	it('puts specials first, then tokens by frequency, ties by first appearance', () => {
		// counts: a 3, b 2, c 2, d 1
		expect(buildVocab(stories, 10).words).toEqual(['<unk>', '<end>', 'a', 'b', 'c', 'd']);
	});
	it('cuts to the requested size', () => {
		expect(buildVocab(stories, 4).words).toEqual(['<unk>', '<end>', 'a', 'b']);
	});
});

describe('encode and decode', () => {
	const vocab = vocabFromWords(['<unk>', '<end>', 'the', 'cat', 'sat', '.', ',']);
	it('maps unknown words to UNK', () => {
		expect(encode(vocab, 'The cat flew.')).toEqual([2, 3, UNK, 5]);
	});
	it('joins with spaces, no space before marks, skips END', () => {
		expect(decode(vocab, [2, 3, 6, 4, 5, END])).toBe('the cat, sat.');
	});
	it('writes <unk> for unknown ids', () => {
		expect(decode(vocab, [UNK, 3])).toBe('<unk> cat');
	});
});
