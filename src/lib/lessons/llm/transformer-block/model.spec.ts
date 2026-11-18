import { describe, expect, it } from 'vitest';
import { nextLogits } from '../../../ml/gpt.ts';
import { softmax } from '../../../ml/vec.ts';
import { encode } from '../../../ml/wordtok.ts';
import { lmFromBuffer } from '../shared/lm.ts';
import { buildBlockModel, TEXT } from './model.ts';

// The project has no Node type definitions, so node:fs is loaded through a variable and typed by hand.
const fsName = 'node:fs';
const fs = (await import(/* @vite-ignore */ fsName)) as { readFileSync(url: URL): Uint8Array };
const bytes = fs.readFileSync(new URL('../shared/tinystories.bin', import.meta.url));
const lm = lmFromBuffer(bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength) as ArrayBuffer);
const m = buildBlockModel(lm);

function maxAbs(mats: number[][][]): number {
	let best = 0;
	for (const mat of mats) for (const row of mat) for (const v of row) best = Math.max(best, Math.abs(v));
	return best;
}

describe('transformer block model', () => {
	it('has five known tokens and one trace per block', () => {
		expect(m.tokens).toHaveLength(5);
		expect(m.tokens).not.toContain('<unk>');
		expect(m.depth).toBe(4);
		expect(m.layers).toHaveLength(m.depth);
		expect(m.d).toBe(64);
		for (const l of m.layers) expect(l.x[0]).toHaveLength(64);
	});
	it('averages attention into rows that sum to one and look only backwards', () => {
		expect(m.attention).toHaveLength(m.depth);
		for (const layer of m.attention) {
			layer.forEach((row, i) => {
				expect(row.reduce((a, b) => a + b, 0)).toBeCloseTo(1, 6);
				row.forEach((v, j) => {
					if (j > i) expect(v).toBe(0);
				});
			});
		}
	});
	it('has eight distinct candidates that are real words', () => {
		expect(m.candidates).toHaveLength(8);
		expect(new Set(m.candidates).size).toBe(8);
		expect(m.candidates).not.toContain('<unk>');
		expect(m.candidates).not.toContain('<end>');
	});
	it('reads the last block exactly as the model predicts, best first', () => {
		const logits = Array.from(nextLogits(lm.model, encode(lm.vocab, TEXT)));
		logits[0] = -Infinity;
		const p = softmax(logits);
		const own = m.candidates.map((w) => p[lm.vocab.ids.get(w)!]);
		m.probs[3].forEach((v, c) => expect(v).toBeCloseTo(own[c], 6));
		for (let c = 1; c < 8; c++) expect(m.probs[3][c - 1]).toBeGreaterThanOrEqual(m.probs[3][c]);
	});
	it('gives probabilities that sum with the rest to one', () => {
		for (let k = 0; k < m.depth; k++) {
			expect(m.probs[k]).toHaveLength(8);
			for (const v of m.probs[k]) expect(Number.isFinite(v) && v >= 0 && v <= 1).toBe(true);
			expect(m.probs[k].reduce((a, b) => a + b, 0) + m.rest[k]).toBeCloseTo(1, 6);
		}
	});
	it('reports the true maxima for colour scales', () => {
		const stream = maxAbs(m.layers.flatMap((l) => [l.x, l.mid, l.out]));
		const hidden = maxAbs(m.layers.map((l) => l.hidden));
		expect(Number.isFinite(stream) && stream > 0).toBe(true);
		expect(m.maxStream).toBe(stream);
		expect(m.maxHidden).toBe(hidden);
	});
});

// The lesson's last step describes what this particular trained model does. If a retrained model fails these,
// rewrite that step's narration (lesson.ts) to match the new model, then update the tests.
describe('what the narration says about the trained weights', () => {
	const her = m.candidates.indexOf('her');
	const his = m.candidates.indexOf('his');
	it('predicts "her" with the full stack', () => {
		expect(m.candidates[0]).toBe('her');
		expect(m.probs[3][her]).toBeGreaterThan(0.5);
	});
	it('guesses "his" when read after one block', () => {
		expect(his).toBeGreaterThanOrEqual(0);
		expect(Math.max(...m.probs[0])).toBe(m.probs[0][his]);
		expect(m.probs[0][his]).toBeGreaterThan(3 * m.probs[0][her]);
	});
});
