import { describe, expect, it } from 'vitest';
import { conv2d, makeGrid, maxPool, outSize, patchAt, relu } from './conv.ts';

const IDENTITY = makeGrid(3, 3, [0, 0, 0, 0, 1, 0, 0, 0, 0]);

function ramp(w: number, h: number) {
	return makeGrid(
		w,
		h,
		Array.from({ length: w * h }, (_, i) => (i * 7) % 11)
	);
}

describe('outSize', () => {
	it('follows floor((n + 2 pad - k) / stride) + 1', () => {
		expect(outSize(24, 3, 1, 0)).toBe(22);
		expect(outSize(24, 3, 1, 1)).toBe(24);
		expect(outSize(24, 3, 2, 0)).toBe(11);
		expect(outSize(24, 3, 2, 1)).toBe(12);
	});
});

describe('conv2d', () => {
	it('identity kernel with pad 1 returns the image', () => {
		const img = ramp(6, 5);
		const out = conv2d(img, IDENTITY, { pad: 1 });
		expect(out.w).toBe(6);
		expect(out.h).toBe(5);
		expect(Array.from(out.data)).toEqual(Array.from(img.data));
	});

	it('matches a hand-computed 4 x 4 example', () => {
		const img = makeGrid(4, 4, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]);
		const k = makeGrid(3, 3, [1, 0, -1, 2, 0, -2, 1, 0, -1]);
		const out = conv2d(img, k);
		expect(out.w).toBe(2);
		expect(out.h).toBe(2);
		// every window is a linear ramp, so each cell is -2 * (1 + 2 + 1) = -8
		expect(Array.from(out.data)).toEqual([-8, -8, -8, -8]);
		const blur = conv2d(img, makeGrid(3, 3, Array(9).fill(1)));
		expect(Array.from(blur.data)).toEqual([54, 63, 90, 99]);
	});

	it('vertical-edge kernel is zero on flat areas and nonzero on the step', () => {
		const img = makeGrid(
			8,
			6,
			Array.from({ length: 48 }, (_, i) => (i % 8 >= 4 ? 1 : 0))
		);
		const k = makeGrid(3, 3, [-1, 0, 1, -2, 0, 2, -1, 0, 1]);
		const out = conv2d(img, k);
		expect(out.w).toBe(6);
		for (let r = 0; r < out.h; r++) {
			for (let c = 0; c < out.w; c++) {
				const v = out.data[r * out.w + c];
				// output column c looks at image columns c..c+2; the step sits between columns 3 and 4
				if (c === 2 || c === 3) expect(v).toBe(4);
				else expect(v).toBe(0);
			}
		}
	});
});

describe('patchAt', () => {
	it('returns the pixels under the kernel', () => {
		const img = ramp(5, 5);
		const p = patchAt(img, IDENTITY, 1, 2);
		for (let i = 0; i < 3; i++) {
			for (let j = 0; j < 3; j++) expect(p.data[i * 3 + j]).toBe(img.data[(1 + i) * 5 + 2 + j]);
		}
	});

	it('pads with zeros at a corner', () => {
		const img = makeGrid(4, 4, Array(16).fill(5));
		const p = patchAt(img, IDENTITY, 0, 0, { pad: 1 });
		expect(Array.from(p.data)).toEqual([0, 0, 0, 0, 5, 5, 0, 5, 5]);
	});

	it('sum(patch * kernel) equals conv2d with stride 2 and pad 1', () => {
		const img = ramp(7, 6);
		const k = makeGrid(3, 3, [1, -2, 0, 3, 1, -1, 2, 0, 1]);
		const opts = { stride: 2, pad: 1 };
		const out = conv2d(img, k, opts);
		expect(out.w).toBe(outSize(7, 3, 2, 1));
		for (const [r, c] of [
			[0, 0],
			[1, 2],
			[2, 3],
			[out.h - 1, out.w - 1]
		]) {
			const p = patchAt(img, k, r, c, opts);
			let s = 0;
			for (let i = 0; i < 9; i++) s += p.data[i] * k.data[i];
			expect(s).toBeCloseTo(out.data[r * out.w + c], 5);
		}
	});
});

describe('relu and maxPool', () => {
	it('relu clips negatives', () => {
		expect(Array.from(relu(makeGrid(2, 2, [-1, 2, 0, -0.5])).data)).toEqual([0, 2, 0, 0]);
	});

	it('maxPool on a 5 x 5 grid gives 2 x 2', () => {
		const g = makeGrid(
			5,
			5,
			Array.from({ length: 25 }, (_, i) => i)
		);
		const p = maxPool(g);
		expect(p.w).toBe(2);
		expect(p.h).toBe(2);
		expect(Array.from(p.data)).toEqual([6, 8, 16, 18]);
	});
});
