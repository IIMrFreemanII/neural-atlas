import { describe, expect, it } from 'vitest';
import { mulberry32 } from '../../../ml/rng.ts';
import { distribution, sampleIndex } from '../../../ml/sampling.ts';
import { softmax } from '../../../ml/vec.ts';
import { END, UNK } from '../../../ml/wordtok.ts';
import { lmFromBuffer, type Lm } from '../shared/lm.ts';
import { PROMPTS, nextScores, promptIds, tokenText } from './data.ts';

// The project has no Node type definitions, so node:fs is loaded through a variable and typed by hand.
const fsName = 'node:fs';
const fs = (await import(/* @vite-ignore */ fsName)) as { readFileSync(url: URL): Uint8Array };
const bytes = fs.readFileSync(new URL('../shared/tinystories.bin', import.meta.url));
const lm: Lm = lmFromBuffer(bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength) as ArrayBuffer);

function generate(prompt: number, n: number, temperature: number, seed: number): number[] {
	const rand = mulberry32(seed);
	const ids = promptIds(lm, prompt);
	const out: number[] = [];
	for (let i = 0; i < n; i++) {
		const d = distribution(nextScores(lm, ids), { temperature, topK: 0, topP: 1 });
		const id = sampleIndex(d, rand());
		ids.push(id);
		out.push(id);
	}
	return out;
}

describe('sampling data', () => {
	it('has three prompts whose words are all in the vocabulary', () => {
		expect(PROMPTS.length).toBe(3);
		for (let i = 0; i < 3; i++) {
			const ids = promptIds(lm, i);
			expect(ids.length).toBeGreaterThan(0);
			expect(ids).not.toContain(UNK);
		}
	});
	it('scores every token, with -Infinity only at <unk>', () => {
		for (let i = 0; i < 3; i++) {
			const s = nextScores(lm, promptIds(lm, i));
			expect(s.length).toBe(lm.model.config.vocab);
			s.forEach((x, j) => {
				if (j === UNK) expect(x).toBe(-Infinity);
				else expect(Number.isFinite(x)).toBe(true);
			});
		}
	});
	it('gives probabilities that sum to 1 and none to <unk>', () => {
		const p = softmax(nextScores(lm, promptIds(lm, 0)));
		expect(p.reduce((a, b) => a + b, 0)).toBeCloseTo(1, 6);
		expect(p[UNK]).toBe(0);
	});
	it('keeps top-k and top-p working with the -Infinity entry', () => {
		const s = nextScores(lm, promptIds(lm, 1));
		for (const o of [{ topK: 5, topP: 1 }, { topK: 0, topP: 0.8 }]) {
			const d = distribution(s, { temperature: 1, ...o });
			expect(d.reduce((a, b) => a + b, 0)).toBeCloseTo(1, 6);
			expect(d[UNK]).toBe(0);
		}
	});
	it('never generates <unk> at temperature 1', () => {
		for (let p = 0; p < 3; p++) expect(generate(p, 60, 1, 7 + p)).not.toContain(UNK);
	});
	it('is repeatable at temperature 0', () => {
		expect(generate(0, 30, 0, 1)).toEqual(generate(0, 30, 0, 2));
	});
	it('writes the end token as end of story', () => {
		expect(tokenText(lm, END)).toBe('end of story');
		expect(tokenText(lm, 2)).toBe(lm.vocab.words[2]);
	});
});

// The top-p step describes what this particular trained model does. If a retrained model fails these, rewrite
// that step's narration (lesson.ts) to match the new model, then update the tests.
describe('what the narration says about the trained weights', () => {
	const keptAt = (prompt: number) =>
		distribution(nextScores(lm, promptIds(lm, prompt)), { temperature: 1.3, topK: 0, topP: 0.8 }).filter((p) => p > 0).length;
	it('is nearly sure that "to" follows "one day, lily went"', () => {
		const p = softmax(nextScores(lm, promptIds(lm, 1)));
		const best = p.indexOf(Math.max(...p));
		expect(tokenText(lm, best)).toBe('to');
		expect(p[best]).toBeGreaterThan(0.5);
	});
	it('keeps a handful of tokens for that start under top-p and many for the other two', () => {
		expect(keptAt(1)).toBeLessThanOrEqual(10);
		expect(keptAt(0)).toBeGreaterThanOrEqual(50);
		expect(keptAt(2)).toBeGreaterThanOrEqual(50);
	});
});
