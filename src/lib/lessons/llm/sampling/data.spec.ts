import { describe, expect, it } from 'vitest';
import { nextLogits } from '../../../ml/ngram.ts';
import { mulberry32 } from '../../../ml/rng.ts';
import { distribution, sampleIndex } from '../../../ml/sampling.ts';
import { softmax } from '../../../ml/vec.ts';
import { CORPUS, MODEL, PROMPTS, promptIds } from './data.ts';

const probsFor = (prompt: number) => softmax(nextLogits(MODEL, promptIds(prompt)));

function generate(prompt: number, n: number, temperature: number, seed: number): number[] {
	const rand = mulberry32(seed);
	const ids = promptIds(prompt);
	const out: number[] = [];
	for (let i = 0; i < n; i++) {
		const d = distribution(nextLogits(MODEL, ids), { temperature, topK: 0, topP: 1 });
		const id = sampleIndex(d, rand());
		ids.push(id);
		out.push(id);
	}
	return out;
}

describe('sampling data', () => {
	it('has a corpus of the intended size and a small vocabulary', () => {
		const words = CORPUS.split(/\s+/).filter(Boolean).length;
		expect(words).toBeGreaterThan(540);
		expect(words).toBeLessThan(800);
		expect(MODEL.vocab.length).toBeGreaterThanOrEqual(90);
		expect(MODEL.vocab.length).toBeLessThanOrEqual(220);
		expect(PROMPTS.length).toBe(3);
		for (let i = 0; i < 3; i++) expect(promptIds(i).length).toBeGreaterThan(0);
	});
	it('prompt 0: mat is the favorite but not certain', () => {
		const p = probsFor(0);
		const top = p.indexOf(Math.max(...p));
		expect(MODEL.vocab[top]).toBe('mat');
		expect(p[top]).toBeGreaterThan(0.25);
		expect(p[top]).toBeLessThan(0.55);
		expect(p.filter((x) => x >= 0.03).length).toBeGreaterThanOrEqual(6);
	});
	it('prompt 1 has many plausible continuations', () => {
		expect(probsFor(1).filter((x) => x >= 0.03).length).toBeGreaterThanOrEqual(6);
	});
	it('prompt 2 has one clear favorite', () => {
		expect(Math.max(...probsFor(2))).toBeGreaterThan(0.5);
	});
	it('greedy generation loops', () => {
		const g = generate(0, 40, 0, 1).map(String);
		const seen = new Set<string>();
		let loops = false;
		for (let i = 0; i + 6 <= g.length; i++) {
			const key = g.slice(i, i + 6).join(',');
			if (seen.has(key)) loops = true;
			seen.add(key);
		}
		expect(loops).toBe(true);
	});
	it('temperature 1 generation is varied', () => {
		expect(new Set(generate(0, 40, 1, 1)).size).toBeGreaterThanOrEqual(12);
	});
});
