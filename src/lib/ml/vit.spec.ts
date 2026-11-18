import { describe, expect, it } from 'vitest';
import { mulberry32 } from './rng.ts';
import {
	CLASSES,
	DIM,
	GRID,
	HEADS,
	IMAGE,
	initVit,
	LAYERS,
	PARAM_COUNT,
	PATCH,
	PATCHES,
	packVit,
	patchPixels,
	positionSimilarity,
	TOKENS,
	unpackVit,
	vitForward
} from './vit.ts';

function randomImage(seed: number): Float32Array {
	const r = mulberry32(seed);
	return Float32Array.from({ length: IMAGE * IMAGE }, () => r());
}

/** Moves whole patches: patch p of the result is patch perm[p] of the source. */
function permutePatches(image: Float32Array, perm: number[]): Float32Array {
	const out = new Float32Array(image.length);
	for (let p = 0; p < PATCHES; p++) {
		const src = perm[p];
		for (let r = 0; r < PATCH; r++) {
			for (let c = 0; c < PATCH; c++) {
				const from = (Math.floor(src / GRID) * PATCH + r) * IMAGE + (src % GRID) * PATCH + c;
				const to = (Math.floor(p / GRID) * PATCH + r) * IMAGE + (p % GRID) * PATCH + c;
				out[to] = image[from];
			}
		}
	}
	return out;
}

const PERM = [5, 12, 0, 9, 3, 15, 7, 1, 14, 2, 10, 6, 11, 4, 13, 8];

describe('vitForward', () => {
	const w = initVit(1);
	const t = vitForward(w, randomImage(2));

	it('gives finite probabilities that sum to 1', () => {
		expect(t.probs.length).toBe(CLASSES);
		expect(t.logits.length).toBe(CLASSES);
		for (const p of t.probs) expect(Number.isFinite(p)).toBe(true);
		expect(Array.from(t.probs).reduce((a, b) => a + b, 0)).toBeCloseTo(1, 5);
	});
	it('has the documented shapes', () => {
		expect(t.patchEmbed.length).toBe(PATCHES);
		expect(t.tokens.length).toBe(TOKENS);
		expect(t.cls.length).toBe(DIM);
		expect(t.layers.length).toBe(LAYERS);
		for (const l of t.layers) {
			for (const toks of [l.input, l.mid, l.output]) {
				expect(toks.length).toBe(TOKENS);
				for (const tok of toks) expect(tok.length).toBe(DIM);
			}
			expect(l.attn.length).toBe(HEADS);
			for (const head of l.attn) expect(head.length).toBe(TOKENS);
		}
		expect(t.layers[0].input).toBe(t.tokens);
		expect(t.layers[1].input).toBe(t.layers[0].output);
	});
	it('has attention rows that sum to 1', () => {
		for (const l of t.layers) {
			for (const head of l.attn) {
				for (const row of head) {
					expect(row.length).toBe(TOKENS);
					expect(row.reduce((a, b) => a + b, 0)).toBeCloseTo(1, 6);
				}
			}
		}
	});
	it('is deterministic', () => {
		expect(Array.from(vitForward(initVit(1), randomImage(2)).logits)).toEqual(Array.from(t.logits));
	});
});

describe('packing', () => {
	it('round-trips', () => {
		const w = initVit(3);
		const b64 = packVit(w);
		expect(atob(b64).length).toBe(PARAM_COUNT * 4);
		const back = unpackVit(b64);
		expect(packVit(back)).toBe(b64);
		const img = randomImage(4);
		expect(Array.from(vitForward(back, img).logits)).toEqual(Array.from(vitForward(w, img).logits));
	});
	it('throws on the wrong length', () => {
		expect(() => unpackVit(btoa('abcd'))).toThrow();
		expect(() => unpackVit('')).toThrow();
	});
});

describe('patchPixels', () => {
	const image = Float32Array.from({ length: 256 }, (_, i) => i);
	it('picks a patch in the middle', () => {
		// Patch 5 = row 1, column 1: rows 4..7, columns 4..7.
		const px = Array.from(patchPixels(image, 5));
		expect(px).toEqual([68, 69, 70, 71, 84, 85, 86, 87, 100, 101, 102, 103, 116, 117, 118, 119]);
	});
	it('picks the bottom-right corner', () => {
		const px = Array.from(patchPixels(image, 15));
		expect(px).toEqual([204, 205, 206, 207, 220, 221, 222, 223, 236, 237, 238, 239, 252, 253, 254, 255]);
	});
});

describe('positions', () => {
	const w = initVit(5);
	const img = randomImage(6);
	const moved = permutePatches(img, PERM);

	it('without positions, moving patches permutes the patch tokens and keeps the logits', () => {
		const a = vitForward(w, img, { noPositions: true });
		const b = vitForward(w, moved, { noPositions: true });
		for (let c = 0; c < CLASSES; c++) expect(Math.abs(a.logits[c] - b.logits[c])).toBeLessThan(1e-5);
		for (let l = 0; l < LAYERS; l++) {
			for (const key of ['input', 'mid', 'output'] as const) {
				expect(Math.max(...diff(a.layers[l][key][0], b.layers[l][key][0]))).toBeLessThan(1e-5);
				for (let p = 0; p < PATCHES; p++) {
					expect(Math.max(...diff(a.layers[l][key][1 + PERM[p]], b.layers[l][key][1 + p]))).toBeLessThan(1e-5);
				}
			}
		}
	});
	it('with positions, the same move changes the logits', () => {
		const big = initVit(5);
		for (let i = 0; i < big.pos.length; i++) big.pos[i] *= 50;
		const a = vitForward(big, img);
		const b = vitForward(big, moved);
		expect(Math.max(...diff(a.logits, b.logits))).toBeGreaterThan(1e-3);
	});
	it('positionSimilarity is a cosine', () => {
		expect(positionSimilarity(w, 3, 3)).toBeCloseTo(1, 5);
		expect(positionSimilarity(w, 2, 9)).toBeCloseTo(positionSimilarity(w, 9, 2), 10);
		expect(Math.abs(positionSimilarity(w, 2, 9))).toBeLessThanOrEqual(1);
	});
});

function diff(a: ArrayLike<number>, b: ArrayLike<number>): number[] {
	return Array.from(a, (v, i) => Math.abs(v - b[i]));
}
