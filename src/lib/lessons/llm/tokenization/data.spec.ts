import { describe, expect, it } from 'vitest';
import { encode } from '../../../ml/bpe.ts';
import { CORPUS, DEFAULT_TEXT, MAX_MERGES, MODEL, TRAIN_MAX, TRAIN_TEXT, WORDS } from './data.ts';

describe('tokenization data', () => {
	it('has a corpus of the planned size', () => {
		expect(CORPUS.length).toBeGreaterThan(3000);
		expect(CORPUS.length).toBeLessThanOrEqual(6000);
	});
	it('has a training text of 60 to 80 characters', () => {
		expect(TRAIN_TEXT.length).toBeGreaterThanOrEqual(60);
		expect(TRAIN_TEXT.length).toBeLessThanOrEqual(80);
	});
	it('learns MAX_MERGES merges', () => {
		expect(MODEL.merges.length).toBe(MAX_MERGES);
	});
	it('splits the default text into meaningful pieces', () => {
		const e = encode(DEFAULT_TEXT, MODEL);
		expect(e.ids.every((i) => i >= 0)).toBe(true);
		expect(e.tokens.length).toBeLessThanOrEqual(10);
		expect(encode('Tokenizers', MODEL).tokens.length).toBeLessThanOrEqual(4);
		const un = encode(' unbelievable', MODEL).tokens;
		expect(un.length).toBe(3);
		expect(un.every((t) => [...t].length >= 2)).toBe(true);
	});
	it('keeps the example words out of the corpus', () => {
		expect(WORDS.has('tokenizers')).toBe(false);
		expect(WORDS.has('unbelievable')).toBe(false);
		expect(WORDS.has('split')).toBe(true);
		expect(WORDS.has('words')).toBe(true);
	});
	it('has a usable trainer text', () => {
		expect(TRAIN_MAX).toBeGreaterThanOrEqual(10);
	});
	it('keeps shrinking the token count over most of the slider', () => {
		const text = 'The reader believed the unreadable words and organized the tokens of a comfortable town';
		const n = (l: number) => encode(text, MODEL, l).tokens.length;
		expect(n(0)).toBeGreaterThan(n(100));
		expect(n(100)).toBeGreaterThan(n(200));
		expect(n(200)).toBeGreaterThan(n(MAX_MERGES));
	});
});
