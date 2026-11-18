import { describe, expect, it } from 'vitest';
import { distribution, entropy, sampleIndex, topKMask, topPMask } from './sampling.ts';

const sum = (v: ArrayLike<number>) => Array.from(v).reduce((a, b) => a + b, 0);
const logits = [2.1, 0.3, -1, 1.4, 0.9, -0.2, 3.0, 1.4];

describe('distribution', () => {
	it('sums to 1 for many option sets', () => {
		for (const temperature of [0, 0.3, 1, 1.7, 2])
			for (const topK of [0, 1, 3, 20])
				for (const topP of [0.1, 0.5, 0.9, 1])
					expect(sum(distribution(logits, { temperature, topK, topP }))).toBeCloseTo(1, 10);
	});
	it('is one-hot at the argmax when greedy', () => {
		const d = distribution(logits, { temperature: 0, topK: 0, topP: 1 });
		expect(d[6]).toBe(1);
		expect(sum(d)).toBe(1);
		expect(d.filter((x) => x > 0).length).toBe(1);
	});
	it('keeps the ranking at every temperature', () => {
		const order = (d: number[]) => d.map((x, i) => [x, i]).sort((a, b) => b[0] - a[0] || a[1] - b[1]).map((e) => e[1]);
		const base = order(distribution(logits, { temperature: 1, topK: 0, topP: 1 }));
		for (const t of [0.2, 0.5, 1.5, 2]) expect(order(distribution(logits, { temperature: t, topK: 0, topP: 1 }))).toEqual(base);
	});
	it('raises entropy with temperature', () => {
		let prev = -1;
		for (const t of [0.2, 0.5, 1, 1.5, 2]) {
			const h = entropy(distribution(logits, { temperature: t, topK: 0, topP: 1 }));
			expect(h).toBeGreaterThan(prev);
			prev = h;
		}
	});
	it('top-k leaves exactly k nonzero entries', () => {
		for (const k of [1, 2, 5]) {
			const d = distribution(logits, { temperature: 1, topK: k, topP: 1 });
			expect(d.filter((x) => x > 0).length).toBe(k);
		}
	});
	it('applies top-p after top-k on the renormalized rest', () => {
		const d = distribution(logits, { temperature: 1, topK: 4, topP: 0.5 });
		expect(sum(d)).toBeCloseTo(1, 10);
		expect(d.filter((x) => x > 0).length).toBeLessThanOrEqual(4);
	});
});

describe('masks', () => {
	const probs = [0.5, 0.3, 0.15, 0.05];
	it('top-k keeps the largest, lower index wins ties', () => {
		expect(topKMask([0.2, 0.4, 0.4], 1)).toEqual([false, true, false]);
		expect(topKMask([0.3, 0.3, 0.4], 2)).toEqual([true, false, true]);
		expect(topKMask(probs, 0)).toEqual([true, true, true, true]);
		expect(topKMask(probs, 9)).toEqual([true, true, true, true]);
	});
	it('top-p keeps the smallest prefix', () => {
		expect(topPMask(probs, 0.8)).toEqual([true, true, false, false]);
		expect(topPMask(probs, 0.81)).toEqual([true, true, true, false]);
		expect(topPMask(probs, 0.01)).toEqual([true, false, false, false]);
		expect(topPMask([0.6, 0.4, 0], 1)).toEqual([true, true, false]);
	});
	it('top-p works on unsorted input', () => {
		expect(topPMask([0.05, 0.15, 0.3, 0.5], 0.8)).toEqual([false, false, true, true]);
	});
});

describe('sampleIndex', () => {
	const p = [0.2, 0, 0.5, 0.3];
	it('handles the boundaries', () => {
		expect(sampleIndex(p, 0)).toBe(0);
		expect(sampleIndex(p, 0.1999)).toBe(0);
		expect(sampleIndex(p, 0.2)).toBe(2);
		expect(sampleIndex(p, 0.6999)).toBe(2);
		expect(sampleIndex(p, 0.7)).toBe(3);
		expect(sampleIndex(p, 0.999999)).toBe(3);
	});
	it('never returns an entry with probability 0', () => {
		for (let i = 0; i < 1000; i++) expect(p[sampleIndex(p, i / 1000)]).toBeGreaterThan(0);
		expect(sampleIndex([0.5, 0.5, 0], 0.9999999999)).toBe(1);
		expect(sampleIndex([0.5, 0.4, 0], 0.99)).toBe(1);
	});
});

describe('entropy', () => {
	it('is in bits', () => {
		expect(entropy([0.5, 0.5])).toBeCloseTo(1, 12);
		expect(entropy([1, 0])).toBe(0);
		expect(entropy([0.25, 0.25, 0.25, 0.25])).toBeCloseTo(2, 12);
	});
});
