import { describe, expect, it } from 'vitest';
import { mulberry32 } from './rng.ts';
import { backward, C1, C2, CLASSES, forward, initCnn, loss, receptiveField, type CnnWeights } from './cnn.ts';

function randomImage(seed: number): Float32Array {
	const r = mulberry32(seed);
	return Float32Array.from({ length: 256 }, () => r());
}

/** The same weights as doubles, so finite differences are not limited by Float32 rounding. */
function toF64(w: CnnWeights): CnnWeights {
	const f = (a: Float32Array) => Float64Array.from(a) as unknown as Float32Array;
	return { w1: f(w.w1), b1: f(w.b1), w2: f(w.w2), b2: f(w.b2), w3: f(w.w3), b3: f(w.b3) };
}

describe('forward', () => {
	it('has the documented shapes and normalized probabilities', () => {
		const t = forward(initCnn(1), randomImage(2));
		expect(t.input.length).toBe(256);
		expect(t.z1.length).toBe(C1 * 14 * 14);
		expect(t.a1.length).toBe(C1 * 14 * 14);
		expect(t.p1.length).toBe(C1 * 7 * 7);
		expect(t.p1Arg.length).toBe(C1 * 7 * 7);
		expect(t.z2.length).toBe(C2 * 5 * 5);
		expect(t.a2.length).toBe(C2 * 5 * 5);
		expect(t.g.length).toBe(C2);
		expect(t.gArg.length).toBe(C2);
		expect(t.logits.length).toBe(CLASSES);
		expect(t.probs.length).toBe(CLASSES);
		expect(Array.from(t.probs).reduce((a, b) => a + b, 0)).toBeCloseTo(1, 5);
		for (let i = 0; i < t.a1.length; i++) expect(t.a1[i]).toBeGreaterThanOrEqual(0);
		for (let k = 0; k < C2; k++) expect(t.a2[t.gArg[k]]).toBe(t.g[k]);
	});
	it('is deterministic', () => {
		const w = initCnn(3);
		const x = randomImage(4);
		const a = forward(w, x);
		const b = forward(initCnn(3), x);
		expect(Array.from(a.probs)).toEqual(Array.from(b.probs));
		expect(Array.from(a.a2)).toEqual(Array.from(b.a2));
	});
	it('records where each pooled maximum came from', () => {
		const t = forward(initCnn(5), randomImage(6));
		for (let i = 0; i < t.p1.length; i++) expect(t.a1[t.p1Arg[i]]).toBe(t.p1[i]);
	});
});

describe('backward', () => {
	it('matches central finite differences on all six weight arrays', () => {
		const w = toF64(initCnn(7));
		// Positive biases keep the ReLUs of this random input mostly active.
		w.b1.fill(0.2);
		w.b2.fill(0.3);
		const x = randomImage(8);
		const label = 2;
		const grads = backward(w, forward(w, x, Float64Array), label);
		const f = () => loss(forward(w, x, Float64Array), label);
		const eps = 1e-6;
		const names = ['w1', 'b1', 'w2', 'b2', 'w3', 'b3'] as const;
		let checked = 0;
		let nonzero = 0;
		for (const name of names) {
			const arr = w[name];
			const stride = Math.max(1, Math.floor(arr.length / 12));
			for (let k = 0; k < arr.length; k += stride) {
				const orig = arr[k];
				arr[k] = orig + eps;
				const lp = f();
				arr[k] = orig - eps;
				const lm = f();
				arr[k] = orig;
				const num = (lp - lm) / (2 * eps);
				const ana = grads[name][k];
				const err = Math.abs(num - ana) / Math.max(1e-6, Math.abs(num) + Math.abs(ana));
				expect(err, `${name}[${k}] numeric ${num} analytic ${ana}`).toBeLessThan(1e-3);
				checked++;
				if (Math.abs(num) > 1e-4) nonzero++;
			}
		}
		expect(checked).toBeGreaterThanOrEqual(40);
		expect(nonzero).toBeGreaterThanOrEqual(30);
	});
	it('has the same shapes as the weights', () => {
		const w = initCnn(9);
		const g = backward(w, forward(w, randomImage(1)), 0);
		for (const k of ['w1', 'b1', 'w2', 'b2', 'w3', 'b3'] as const) expect(g[k].length).toBe(w[k].length);
	});
});

describe('receptiveField', () => {
	it('covers 3, 4 and 8 pixels at corners and the center', () => {
		expect(receptiveField('a1', 0, 0)).toEqual({ r0: 0, c0: 0, r1: 2, c1: 2 });
		expect(receptiveField('a1', 13, 13)).toEqual({ r0: 13, c0: 13, r1: 15, c1: 15 });
		expect(receptiveField('a1', 6, 7)).toEqual({ r0: 6, c0: 7, r1: 8, c1: 9 });
		expect(receptiveField('p1', 0, 0)).toEqual({ r0: 0, c0: 0, r1: 3, c1: 3 });
		expect(receptiveField('p1', 6, 6)).toEqual({ r0: 12, c0: 12, r1: 15, c1: 15 });
		expect(receptiveField('p1', 3, 2)).toEqual({ r0: 6, c0: 4, r1: 9, c1: 7 });
		expect(receptiveField('a2', 0, 0)).toEqual({ r0: 0, c0: 0, r1: 7, c1: 7 });
		expect(receptiveField('a2', 4, 4)).toEqual({ r0: 8, c0: 8, r1: 15, c1: 15 });
		expect(receptiveField('a2', 2, 2)).toEqual({ r0: 4, c0: 4, r1: 11, c1: 11 });
	});
});
