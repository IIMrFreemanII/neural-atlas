import { describe, expect, it } from 'vitest';
import { FEATURES, HEADS, SENTENCES, runHead } from './model.ts';

const run = (s: number, h: number, causal = false, sharp = 1) => runHead(SENTENCES[s], HEADS[h], { causal, sharp });
const T = SENTENCES[0].tokens;
const DOG = T.indexOf('dog');
const BALL = T.indexOf('ball');
const IT = T.indexOf('it');
const LAST = T.length - 1;

describe('attention toy model', () => {
	it('has nine tokens and ten numbers per token', () => {
		for (const s of SENTENCES) {
			expect(s.tokens).toHaveLength(9);
			for (const row of s.x) expect(row).toHaveLength(10);
		}
		expect(FEATURES).toHaveLength(6);
	});

	it('every weight row sums to 1', () => {
		for (let s = 0; s < 2; s++)
			for (let h = 0; h < 3; h++)
				for (const causal of [false, true])
					for (const row of run(s, h, causal).weights)
						expect(row.reduce((a, b) => a + b, 0)).toBeCloseTo(1, 10);
	});

	it('head 0: wet looks at ball, tired at dog, it splits', () => {
		expect(run(0, 0).weights[LAST][BALL]).toBeGreaterThan(0.6);
		expect(run(1, 0).weights[LAST][DOG]).toBeGreaterThan(0.6);
		const w = run(0, 0).weights[IT];
		expect(w[DOG]).toBeGreaterThan(0.3);
		expect(w[DOG]).toBeLessThan(0.5);
		expect(w[BALL]).toBeGreaterThan(0.3);
		expect(w[BALL]).toBeLessThan(0.5);
	});

	it('head 1: every token looks at the previous one', () => {
		const w = run(0, 1).weights;
		for (let i = 1; i < w.length; i++) {
			const best = w[i].indexOf(Math.max(...w[i]));
			expect(best).toBe(i - 1);
			expect(w[i][i - 1]).toBeGreaterThan(0.5);
		}
	});

	it('head 2: repeats and self', () => {
		const w = run(0, 2).weights;
		expect(w[3][0]).toBeGreaterThanOrEqual(0.3);
		expect(w[3][3]).toBeGreaterThanOrEqual(0.3);
		expect(w[DOG][DOG]).toBeGreaterThan(0.5);
	});

	it('causal mask zeroes the upper triangle and keeps wet on ball', () => {
		for (let h = 0; h < 3; h++) {
			const w = run(0, h, true).weights;
			for (let i = 0; i < w.length; i++) for (let j = i + 1; j < w.length; j++) expect(w[i][j]).toBe(0);
		}
		expect(run(0, 0, true).weights[LAST][BALL]).toBeGreaterThan(0.6);
	});

	it('a larger sharpness raises the largest weight', () => {
		const a = Math.max(...run(0, 0, false, 1).weights[LAST]);
		const b = Math.max(...run(0, 0, false, 2).weights[LAST]);
		expect(b).toBeGreaterThan(a);
	});
});
