import { describe, expect, it } from 'vitest';
import { forward } from '../../../ml/cnn.ts';
import { mulberry32 } from '../../../ml/rng.ts';
import { randomShape, SAMPLES } from './data.ts';
import { WEIGHTS } from './weights.ts';

const argmax = (p: ArrayLike<number>) => Array.from({ length: p.length }, (_, i) => i).reduce((a, b) => (p[b] > p[a] ? b : a));

describe('trained weights', () => {
	it('classify generated drawings with at least 95% accuracy', () => {
		const rand = mulberry32(4242);
		let ok = 0;
		for (let i = 0; i < 1000; i++) {
			const s = randomShape(rand);
			if (argmax(forward(WEIGHTS, s.image).probs) === s.label) ok++;
		}
		expect(ok / 1000).toBeGreaterThanOrEqual(0.95);
	});
	it('recognize each clean lesson sample with probability above 0.8', () => {
		SAMPLES.forEach((img, i) => {
			expect(forward(WEIGHTS, img).probs[i]).toBeGreaterThan(0.8);
		});
	});
});

describe('randomShape', () => {
	it('is deterministic, covers all labels and stays in range', () => {
		const a = randomShape(mulberry32(5));
		const b = randomShape(mulberry32(5));
		expect(Array.from(a.image)).toEqual(Array.from(b.image));
		const rand = mulberry32(6);
		const seen = new Set<number>();
		for (let i = 0; i < 40; i++) {
			const s = randomShape(rand);
			seen.add(s.label);
			expect(s.image.length).toBe(256);
			for (const x of s.image) {
				expect(x).toBeGreaterThanOrEqual(0);
				expect(x).toBeLessThanOrEqual(1);
			}
		}
		expect(seen.size).toBe(4);
	});
});
