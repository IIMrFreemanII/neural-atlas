import { describe, expect, it } from 'vitest';
import {
	applyGrads,
	backward,
	cloneMLP,
	createMLP,
	forward,
	lossValue,
	trainStep,
	type Act,
	type LossKind
} from './mlp.ts';

const pairs: Array<[Act, LossKind]> = [
	['sigmoid', 'bce'],
	['linear', 'mse']
];

describe('backward', () => {
	for (const hidden of ['tanh', 'sigmoid', 'relu'] as const) {
		for (const [output, kind] of pairs) {
			it(`matches finite differences (${hidden}, ${output}, ${kind})`, () => {
				const net = createMLP([2, 3, 2, 1], hidden, output, 11);
				// Positive biases keep relu units away from zero.
				for (const b of net.b) b.fill(0.3);
				const x = [0.7, -0.4];
				const t = [kind === 'bce' ? 1 : 0.4];
				const g = backward(net, forward(net, x), t, kind);
				const loss = () => lossValue(kind, forward(net, x).a[net.W.length], t);
				const eps = 1e-5;
				const check = (arr: Float64Array, grad: Float64Array) => {
					for (let k = 0; k < arr.length; k++) {
						const orig = arr[k];
						arr[k] = orig + eps;
						const lp = loss();
						arr[k] = orig - eps;
						const lm = loss();
						arr[k] = orig;
						const num = (lp - lm) / (2 * eps);
						const err = Math.abs(num - grad[k]) / Math.max(1e-8, Math.abs(num) + Math.abs(grad[k]));
						expect(err).toBeLessThan(1e-4);
					}
				};
				net.W.forEach((w, l) => check(w, g.W[l]));
				net.b.forEach((b, l) => check(b, g.b[l]));
				// delta of the last layer equals dL/db of the last layer.
				const L = net.W.length;
				check(net.b[L - 1], g.delta[L - 1]);
			});
		}
	}
});

describe('createMLP / cloneMLP / applyGrads', () => {
	it('is deterministic per seed and differs across seeds', () => {
		const a = createMLP([2, 4, 1], 'tanh', 'sigmoid', 1);
		const b = createMLP([2, 4, 1], 'tanh', 'sigmoid', 1);
		const c = createMLP([2, 4, 1], 'tanh', 'sigmoid', 2);
		expect(Array.from(a.W[0])).toEqual(Array.from(b.W[0]));
		expect(Array.from(a.W[0])).not.toEqual(Array.from(c.W[0]));
		expect(Array.from(a.b[0]).every((v) => v === 0)).toBe(true);
	});

	it('clones deeply', () => {
		const a = createMLP([2, 3, 1], 'relu', 'linear', 3);
		const c = cloneMLP(a);
		c.W[0][0] += 1;
		c.sizes[0] = 9;
		expect(c.W[0][0]).not.toBe(a.W[0][0]);
		expect(a.sizes[0]).toBe(2);
	});

	it('applyGrads mutates in place', () => {
		const net = createMLP([1, 1], 'tanh', 'linear', 4);
		const w = net.W[0];
		const before = w[0];
		applyGrads(net, { W: [Float64Array.of(2)], b: [Float64Array.of(1)] }, 0.5);
		expect(net.W[0]).toBe(w);
		expect(w[0]).toBeCloseTo(before - 1, 12);
		expect(net.b[0][0]).toBeCloseTo(-0.5, 12);
	});
});

describe('trainStep', () => {
	it('learns XOR', () => {
		const net = createMLP([2, 4, 1], 'tanh', 'sigmoid', 5);
		const xs = Float64Array.of(0, 0, 0, 1, 1, 0, 1, 1);
		const ys = Float64Array.of(0, 1, 1, 0);
		const all = [0, 1, 2, 3];
		let loss = Infinity;
		for (let i = 0; i < 2000; i++) loss = trainStep(net, xs, ys, all, 0.5, 'bce');
		expect(loss).toBeLessThan(0.1);
	});
});
