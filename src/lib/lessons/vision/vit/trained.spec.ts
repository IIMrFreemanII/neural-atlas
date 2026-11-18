import { describe, expect, it } from 'vitest';
import { PATCHES, vitForward } from '../../../ml/vit.ts';
import { mulberry32 } from '../../../ml/rng.ts';
import { randomShape, SAMPLES } from '../cnn-features/data.ts';
import { WEIGHTS } from './weights.ts';

const argmax = (a: ArrayLike<number>) => {
	let best = 0;
	for (let i = 1; i < a.length; i++) if (a[i] > a[best]) best = i;
	return best;
};
const CROSS = 3;

/** An ink patch has 16 pixels that sum to more than 1.5. */
const inkPatches = (image: Float32Array): boolean[] =>
	Array.from({ length: PATCHES }, (_, p) => {
		const pr = Math.floor(p / 4);
		const pc = p % 4;
		let s = 0;
		for (let r = 0; r < 4; r++) for (let c = 0; c < 4; c++) s += image[(pr * 4 + r) * 16 + pc * 4 + c];
		return s > 1.5;
	});

describe('what the narration says about the trained weights', () => {
	it('is right on drawings with positions, and near guessing without them', () => {
		const rand = mulberry32(777);
		let withPos = 0;
		let without = 0;
		const N = 1000;
		for (let i = 0; i < N; i++) {
			const { image, label } = randomShape(rand);
			if (argmax(vitForward(WEIGHTS, image).probs) === label) withPos++;
			if (argmax(vitForward(WEIGHTS, image, { noPositions: true }).probs) === label) without++;
		}
		expect(withPos / N).toBeGreaterThanOrEqual(0.95);
		expect(without / N).toBeLessThan(0.6);
	});

	it('takes the clean cross for a cross only with positions', () => {
		const image = SAMPLES[CROSS];
		expect(argmax(vitForward(WEIGHTS, image).probs)).toBe(CROSS);
		expect(argmax(vitForward(WEIGHTS, image, { noPositions: true }).probs)).not.toBe(CROSS);
	});

	it('gives the same answer for any order of the patches without positions', () => {
		const rand = mulberry32(31);
		const image = SAMPLES[CROSS];
		const before = vitForward(WEIGHTS, image, { noPositions: true }).probs;
		for (let trial = 0; trial < 5; trial++) {
			const perm = Array.from({ length: PATCHES }, (_, i) => i);
			for (let i = perm.length - 1; i > 0; i--) {
				const j = Math.floor(rand() * (i + 1));
				[perm[i], perm[j]] = [perm[j], perm[i]];
			}
			const moved = new Float32Array(256);
			for (let s = 0; s < PATCHES; s++) {
				const from = perm[s];
				for (let r = 0; r < 4; r++) {
					for (let c = 0; c < 4; c++) {
						moved[(Math.floor(s / 4) * 4 + r) * 16 + (s % 4) * 4 + c] =
							image[(Math.floor(from / 4) * 4 + r) * 16 + (from % 4) * 4 + c];
					}
				}
			}
			const after = vitForward(WEIGHTS, moved, { noPositions: true }).probs;
			for (let k = 0; k < after.length; k++) expect(Math.abs(after[k] - before[k])).toBeLessThan(1e-5);
		}
	});

	it('has the class token of block 2, head 1 look mostly at ink, more than any other block and head', () => {
		const rand = mulberry32(555);
		const share = [
			[0, 0],
			[0, 0]
		];
		const N = 400;
		for (let i = 0; i < N; i++) {
			const { image } = randomShape(rand);
			const ink = inkPatches(image);
			const trace = vitForward(WEIGHTS, image);
			for (let l = 0; l < 2; l++) {
				for (let h = 0; h < 2; h++) {
					let s = 0;
					for (let p = 0; p < PATCHES; p++) if (ink[p]) s += trace.layers[l].attn[h][0][1 + p];
					share[l][h] += s / N;
				}
			}
		}
		expect(share[1][0]).toBeGreaterThan(0.7);
		for (const [l, h] of [[0, 0], [0, 1], [1, 1]]) expect(share[1][0]).toBeGreaterThan(share[l][h]);
	});
});
