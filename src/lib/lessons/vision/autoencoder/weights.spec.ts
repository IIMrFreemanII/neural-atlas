import { describe, expect, it } from 'vitest';
import { decode, encode } from '../../../ml/autoencoder.ts';
import { forward as classify } from '../../../ml/cnn.ts';
import { mulberry32 } from '../../../ml/rng.ts';
import { SAMPLES } from '../cnn-features/data.ts';
import { WEIGHTS as CNN_WEIGHTS } from '../cnn-features/weights.ts';
import { centeredShape, mapPoints, nearestLabel } from './data.ts';
import { WEIGHTS } from './weights.ts';

const points = mapPoints(WEIGHTS);
const held = (() => {
	const rand = mapRand();
	return Array.from({ length: 1000 }, () => centeredShape(rand));
})();

function mapRand() {
	return mulberry32(777);
}

describe('trained autoencoder', () => {
	it('puts the four shapes in separate clusters (k-NN on the map, at least 85%)', () => {
		let ok = 0;
		for (const s of held) {
			const [x, y] = encode(WEIGHTS, s.image);
			if (nearestLabel(points, x, y) === s.label) ok++;
		}
		expect(ok / held.length).toBeGreaterThanOrEqual(0.85);
	});
	it('rebuilds drawings much better than the average image', () => {
		const mean = new Float64Array(256);
		for (const s of held) for (let i = 0; i < 256; i++) mean[i] += s.image[i] / held.length;
		let base = 0;
		let rec = 0;
		for (const s of held) {
			const out = decode(WEIGHTS, encode(WEIGHTS, s.image));
			for (let i = 0; i < 256; i++) {
				base += (s.image[i] - mean[i]) ** 2;
				rec += (s.image[i] - out[i]) ** 2;
			}
		}
		expect(rec / base).toBeLessThanOrEqual(0.6);
	});
	it('rebuilt clean samples are still recognised by the CNN lesson classifier', () => {
		SAMPLES.forEach((img, i) => {
			const out = decode(WEIGHTS, encode(WEIGHTS, img));
			expect(classify(CNN_WEIGHTS, out).probs[i]).toBeGreaterThan(0.6);
		});
	});
	it('keeps every code inside (-1, 1) and gives the clean samples four different labels', () => {
		for (const s of held) {
			const [x, y] = encode(WEIGHTS, s.image);
			expect(Math.abs(x)).toBeLessThan(1);
			expect(Math.abs(y)).toBeLessThan(1);
		}
		const labels = SAMPLES.map((img) => {
			const [x, y] = encode(WEIGHTS, img);
			return nearestLabel(points, x, y);
		});
		expect(labels).toEqual([0, 1, 2, 3]);
	});
	it('centeredShape is deterministic, covers every shape and stays in 0..1', () => {
		expect(Array.from(centeredShape(mulberry32(5)).image)).toEqual(Array.from(centeredShape(mulberry32(5)).image));
		const rand = mulberry32(6);
		const seen = new Set<number>();
		for (let i = 0; i < 40; i++) {
			const s = centeredShape(rand);
			seen.add(s.label);
			expect(s.image.length).toBe(256);
			expect(Math.min(...s.image)).toBeGreaterThanOrEqual(0);
			expect(Math.max(...s.image)).toBeLessThanOrEqual(1);
		}
		expect(seen.size).toBe(4);
	});
	it('mapPoints is deterministic', () => {
		expect(mapPoints(WEIGHTS, 50, 5)).toEqual(mapPoints(WEIGHTS, 50, 5));
	});
});
