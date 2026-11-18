import { describe, expect, it } from 'vitest';
import { ACTIVATION_NAMES, activate, derivative, neuron, type ActivationId } from './activations.ts';

describe('activate', () => {
	it('linear is the identity', () => {
		expect(activate(0, -3.5)).toBe(-3.5);
		expect(activate(0, 2)).toBe(2);
	});
	it('sigmoid', () => {
		expect(activate(1, 0)).toBeCloseTo(0.5, 12);
		expect(activate(1, 10)).toBeGreaterThan(0.999);
		expect(activate(1, -10)).toBeLessThan(0.001);
		expect(activate(1, 1) + activate(1, -1)).toBeCloseTo(1, 12);
	});
	it('relu', () => {
		expect(activate(2, -2)).toBe(0);
		expect(activate(2, 0)).toBe(0);
		expect(activate(2, 1.5)).toBe(1.5);
	});
	it('has names', () => {
		expect(ACTIVATION_NAMES[1]).toBe('Sigmoid');
	});
});

describe('new activations', () => {
	it('tanh', () => {
		expect(activate(3, 0)).toBe(0);
		expect(activate(3, 1)).toBeCloseTo(Math.tanh(1), 12);
		expect(activate(3, -2)).toBeCloseTo(-activate(3, 2), 12);
	});
	it('step', () => {
		expect(activate(4, -0.01)).toBe(0);
		expect(activate(4, 0)).toBe(1);
		expect(activate(4, 3)).toBe(1);
	});
	it('leaky relu', () => {
		expect(activate(5, -2)).toBeCloseTo(-0.2, 12);
		expect(activate(5, 0)).toBe(0);
		expect(activate(5, 2)).toBe(2);
	});
	it('has names', () => {
		expect(ACTIVATION_NAMES[3]).toBe('Tanh');
		expect(ACTIVATION_NAMES[4]).toBe('Step');
		expect(ACTIVATION_NAMES[5]).toBe('Leaky ReLU');
	});
});

describe('derivative', () => {
	it('matches a finite difference for smooth and piecewise-linear ids', () => {
		const h = 1e-6;
		for (const id of [0, 1, 3] as ActivationId[]) {
			for (const z of [-2.3, -0.4, 0.7, 3]) {
				const fd = (activate(id, z + h) - activate(id, z - h)) / (2 * h);
				expect(derivative(id, z)).toBeCloseTo(fd, 6);
			}
		}
		for (const id of [2, 5] as ActivationId[]) {
			for (const z of [-2, -0.5, 0.5, 2]) {
				const fd = (activate(id, z + h) - activate(id, z - h)) / (2 * h);
				expect(derivative(id, z)).toBeCloseTo(fd, 6);
			}
		}
	});
	it('step is zero everywhere', () => {
		expect(derivative(4, -1)).toBe(0);
		expect(derivative(4, 0)).toBe(0);
		expect(derivative(4, 2)).toBe(0);
	});
	it('relu and leaky relu at the kink', () => {
		expect(derivative(2, -1)).toBe(0);
		expect(derivative(2, 0)).toBe(1);
		expect(derivative(5, -1)).toBe(0.1);
		expect(derivative(5, 0)).toBe(1);
	});
});

describe('neuron', () => {
	it('sums weighted inputs plus bias', () => {
		expect(neuron([0.8, -0.5], [1, 0.6], 0.5)).toBeCloseTo(0.8 - 0.3 + 0.5, 12);
	});
	it('with no inputs returns the bias', () => {
		expect(neuron([], [], 0.25)).toBe(0.25);
	});
});
