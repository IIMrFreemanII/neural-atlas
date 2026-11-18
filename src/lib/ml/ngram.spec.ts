import { describe, expect, it } from 'vitest';
import { buildNgram, nextLogits, tokenizeWords } from './ngram.ts';

const sumExp = (v: number[]) => v.reduce((a, b) => a + Math.exp(b), 0);
const argmax = (v: number[]) => v.indexOf(Math.max(...v));

describe('tokenizeWords', () => {
	it('lowercases and splits punctuation', () => {
		expect(tokenizeWords("The cat's here, isn't it? Yes! Ok.")).toEqual([
			'the', "cat's", 'here', ',', "isn't", 'it', '?', 'yes', '!', 'ok', '.'
		]);
		expect(tokenizeWords('')).toEqual([]);
	});
});

describe('buildNgram', () => {
	const m = buildNgram(tokenizeWords('a b a b a c . a b a'));
	it('counts tokens, most frequent first', () => {
		expect(m.vocab).toEqual(['a', 'b', 'c', '.']);
		expect(Array.from(m.uni)).toEqual([5, 3, 1, 1]);
		expect(m.total).toBe(10);
		expect(m.ids.get('c')).toBe(2);
	});
	it('counts bigrams and trigrams', () => {
		expect(m.bi.get(0)!.get(1)).toBe(3); // a -> b
		expect(m.bi.get(0)!.get(2)).toBe(1); // a -> c
		expect(m.tri.get(0 * 4 + 1)!.get(0)).toBe(3); // a b -> a (three times)
		expect(m.tri.get(0 * 4 + 1)!.get(3)).toBeUndefined();
	});
	it('breaks frequency ties by first appearance', () => {
		expect(buildNgram(['x', 'y', 'z']).vocab).toEqual(['x', 'y', 'z']);
	});
});

describe('nextLogits', () => {
	const m = buildNgram(tokenizeWords('the cat sat on the mat . the cat sat on the mat . the dog sat on the rug .'));
	const id = (t: string) => m.ids.get(t)!;
	it('sums to 1 with seen contexts, unseen contexts and an empty context', () => {
		expect(sumExp(nextLogits(m, [id('on'), id('the')]))).toBeCloseTo(1, 10);
		expect(sumExp(nextLogits(m, [id('rug'), id('the')]))).toBeCloseTo(1, 10);
		expect(sumExp(nextLogits(m, [id('mat'), id('mat')]))).toBeCloseTo(1, 10);
		expect(sumExp(nextLogits(m, [id('.')]))).toBeCloseTo(1, 10);
		expect(sumExp(nextLogits(m, []))).toBeCloseTo(1, 10);
	});
	it('makes the observed continuation most likely', () => {
		expect(m.vocab[argmax(nextLogits(m, [id('on'), id('the')]))]).toBe('mat');
		expect(m.vocab[argmax(nextLogits(m, [id('cat'), id('sat')]))]).toBe('on');
	});
	it('gives every token a finite log-probability', () => {
		expect(nextLogits(m, [id('on'), id('the')]).every(Number.isFinite)).toBe(true);
	});
});
