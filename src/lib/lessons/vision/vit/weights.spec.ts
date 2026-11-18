import { describe, expect, it } from 'vitest';
import { mulberry32 } from '../../../ml/rng.ts';
import { vitForward } from '../../../ml/vit.ts';
import { randomShape, SAMPLES } from '../cnn-features/data.ts';
import { REFERENCE_LOGITS, WEIGHTS } from './weights.ts';

describe('trained vision transformer', () => {
	it('gives the same logits as PyTorch for the clean samples', () => {
		expect(REFERENCE_LOGITS.length).toBe(SAMPLES.length);
		SAMPLES.forEach((img, i) => {
			const logits = vitForward(WEIGHTS, img).logits;
			REFERENCE_LOGITS[i].forEach((v, c) => expect(Math.abs(logits[c] - v)).toBeLessThan(1e-3));
		});
	});
	it('classifies at least 95% of 1000 held-out random drawings', () => {
		const rand = mulberry32(777);
		let ok = 0;
		for (let i = 0; i < 1000; i++) {
			const s = randomShape(rand);
			const p = vitForward(WEIGHTS, s.image).probs;
			if (p.indexOf(Math.max(...p)) === s.label) ok++;
		}
		expect(ok / 1000).toBeGreaterThanOrEqual(0.95);
	});
	it('recognises each clean sample as its own class with probability above 0.8', () => {
		SAMPLES.forEach((img, i) => {
			expect(vitForward(WEIGHTS, img).probs[i]).toBeGreaterThan(0.8);
		});
	});
});
