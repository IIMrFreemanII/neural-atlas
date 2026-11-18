import { describe, expect, it } from 'vitest';
import { LM_CONFIG } from '../shared/lm-meta.ts';
import { lmFromBuffer } from '../shared/lm.ts';
import { UNK } from '../../../ml/wordtok.ts';
import { buildTrained, TRAINED_TEXTS } from './trained.ts';

// The project has no Node type definitions, so node:fs is loaded through a variable and typed by hand.
const fsName = 'node:fs';
const fs = (await import(/* @vite-ignore */ fsName)) as { readFileSync(url: URL): Uint8Array };

function readWeights(): ArrayBuffer {
	const bytes = fs.readFileSync(new URL('../shared/tinystories.bin', import.meta.url));
	return bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength) as ArrayBuffer;
}

const sentences = buildTrained(lmFromBuffer(readWeights()));

describe('trained attention data', () => {
	it('has one entry per text and no unknown words', () => {
		expect(sentences.length).toBe(TRAINED_TEXTS.length);
		sentences.forEach((s) => {
			expect(s.tokens.length).toBeGreaterThan(1);
			expect(s.tokens).not.toContain('<unk>');
			expect(s.tokens[0]).not.toBe(UNK);
		});
	});
	it('has every block and head, n x n', () => {
		for (const s of sentences) {
			const n = s.tokens.length;
			expect(s.weights.length).toBe(LM_CONFIG.layers);
			for (const block of s.weights) {
				expect(block.length).toBe(LM_CONFIG.heads);
				for (const m of block) {
					expect(m.length).toBe(n);
					for (const row of m) expect(row.length).toBe(n);
				}
			}
		}
	});
	it('has rows that sum to 1, zeros above the diagonal and finite numbers', () => {
		for (const s of sentences) {
			for (const block of s.weights) {
				for (const m of block) {
					m.forEach((row, i) => {
						expect(row.reduce((a, b) => a + b, 0)).toBeCloseTo(1, 5);
						row.forEach((w, j) => {
							expect(Number.isFinite(w)).toBe(true);
							expect(w).toBeGreaterThanOrEqual(0);
							if (j > i) expect(w).toBe(0);
						});
					});
				}
			}
		}
	});
});

// The last step names the jobs of three heads of this particular trained model. If a retrained model fails
// these, look at its heads again, rewrite that step's narration (lesson.ts), then update the tests.
describe('what the narration says about the trained weights', () => {
	const [, tom, girl] = sentences;
	const top = (row: number[]) => row.indexOf(Math.max(...row));
	it('block 2 head 3 looks back at who the sentence is about', () => {
		const w = tom.weights[1][2];
		expect(tom.tokens[5]).toBe('he');
		expect(top(w[5])).toBe(0); // he -> tom
		expect(top(w[1])).toBe(0); // lost -> tom
		expect(top(girl.weights[1][2][3])).toBe(2); // saw -> girl
	});
	it('block 2 head 1 looks at the latest verb', () => {
		const w = tom.weights[1][0];
		expect(w[2][1]).toBeGreaterThan(0.8); // his -> lost
		expect(w[3][1]).toBeGreaterThan(0.8); // ball -> lost
		expect(girl.weights[1][0][5][3]).toBeGreaterThan(0.8); // big -> saw
	});
	it('block 1 head 1 mostly looks at the word just before', () => {
		let rows = 0;
		let previous = 0;
		for (const s of sentences) {
			const w = s.weights[0][0];
			for (let i = 1; i < w.length; i++) {
				rows++;
				if (top(w[i]) === i - 1) previous++;
			}
		}
		expect(previous / rows).toBeGreaterThan(0.6);
	});
});
