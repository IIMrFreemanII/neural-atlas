import { describe, expect, it } from 'vitest';
import { applyMerge, countPairs, encode, segment, splitWords, trainBpe } from './bpe.ts';

describe('splitWords', () => {
	it('keeps one leading space per word', () => {
		expect(splitWords('the cat sat')).toEqual(['the', ' cat', ' sat']);
	});
	it('round trips awkward input', () => {
		for (const t of ['', ' ', '  ', 'a  b', ' a', 'a ', '  a  b  ', 'Hi, there! (ok)', 'x\ny z', 'héllo wörld 😀 ok']) {
			expect(splitWords(t).join('')).toBe(t);
		}
	});
	it('makes a lone space its own word', () => {
		expect(splitWords('a  b')).toEqual(['a', ' ', ' b']);
		expect(splitWords('a ')).toEqual(['a', ' ']);
		expect(splitWords(' a')).toEqual([' a']);
	});
});

describe('countPairs', () => {
	it('sorts by count and keeps first appearance on ties', () => {
		const words = [['a', 'b'], ['c', 'd'], ['a', 'b'], ['x', 'y', 'z']];
		const c = countPairs(words);
		expect(c[0]).toEqual({ pair: ['a', 'b'], count: 2 });
		expect(c.slice(1).map((e) => e.pair.join(''))).toEqual(['cd', 'xy', 'yz']);
	});
	it('does not cross word boundaries', () => {
		expect(countPairs([['a'], ['b']])).toEqual([]);
	});
});

describe('applyMerge', () => {
	it('merges overlapping runs left to right', () => {
		expect(applyMerge([['a', 'a', 'a']], ['a', 'a'])).toEqual([['aa', 'a']]);
		expect(applyMerge([['a', 'a', 'a', 'a']], ['a', 'a'])).toEqual([['aa', 'aa']]);
	});
	it('returns new arrays', () => {
		const w = [['a', 'b']];
		const out = applyMerge(w, ['a', 'b']);
		expect(out).not.toBe(w);
		expect(w).toEqual([['a', 'b']]);
	});
});

describe('trainBpe', () => {
	const text = 'the cat the cat the cat';
	const m = trainBpe(text, 10);
	it('learns the obvious first merges', () => {
		expect(m.merges[0]).toEqual(['t', 'h']);
		expect(m.merges.length).toBeGreaterThan(2);
	});
	it('builds vocab from sorted characters then merges', () => {
		const chars = [...new Set(text)].sort();
		expect(m.vocab.slice(0, chars.length)).toEqual(chars);
		expect(m.vocab[chars.length]).toBe('th');
		expect(m.ids.get('th')).toBe(chars.length);
	});
	it('stops when no pair repeats', () => {
		expect(trainBpe('abc', 5).merges).toEqual([]);
	});
	it('respects maxMerges', () => {
		expect(trainBpe(text, 2).merges.length).toBe(2);
	});
});

describe('encode', () => {
	const model = trainBpe('the cat sat on the mat the fat cat ate the hat', 20);
	const texts = ['the cat', 'Hello, World!  ', '', ' the  hat ', 'cats 😀'];
	it('reproduces the text for any limit', () => {
		for (const t of texts) for (const lim of [0, 1, 5, 20, undefined]) {
			expect(encode(t, model, lim).tokens.join('')).toBe(t);
		}
	});
	it('gives id -1 to unseen characters', () => {
		const e = encode('qz', model);
		expect(e.ids).toEqual([-1, -1]);
	});
	it('limit 0 gives single characters', () => {
		expect(encode('the cat', model, 0).tokens).toEqual([...'the cat']);
	});
	it('token count never increases with limit', () => {
		let prev = Infinity;
		for (let l = 0; l <= model.merges.length; l++) {
			const n = encode('the cat sat on that hat', model, l).tokens.length;
			expect(n).toBeLessThanOrEqual(prev);
			prev = n;
		}
	});
	it('segment agrees with encode', () => {
		expect(segment('the cat', model.merges, 3).flat()).toEqual(encode('the cat', model, 3).tokens);
	});
});
