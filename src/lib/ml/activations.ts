/** 0 linear (identity), 1 sigmoid, 2 ReLU, 3 tanh, 4 step, 5 leaky ReLU */
export type ActivationId = 0 | 1 | 2 | 3 | 4 | 5;

export const ACTIVATION_NAMES: Record<ActivationId, string> = {
	0: 'Linear',
	1: 'Sigmoid',
	2: 'ReLU',
	3: 'Tanh',
	4: 'Step',
	5: 'Leaky ReLU'
};

export function activate(id: ActivationId, z: number): number {
	switch (id) {
		case 1:
			return 1 / (1 + Math.exp(-z));
		case 2:
			return Math.max(0, z);
		case 3:
			return Math.tanh(z);
		case 4:
			return z >= 0 ? 1 : 0;
		case 5:
			return z >= 0 ? z : 0.1 * z;
		default:
			return z;
	}
}

/** d/dz of activate(id, z). Step is 0 everywhere; ReLU is 0 for z < 0 and 1 for z >= 0; leaky ReLU is 0.1 for z < 0 and 1 for z >= 0. */
export function derivative(id: ActivationId, z: number): number {
	switch (id) {
		case 1: {
			const s = 1 / (1 + Math.exp(-z));
			return s * (1 - s);
		}
		case 2:
			return z >= 0 ? 1 : 0;
		case 3: {
			const t = Math.tanh(z);
			return 1 - t * t;
		}
		case 4:
			return 0;
		case 5:
			return z >= 0 ? 1 : 0.1;
		default:
			return 1;
	}
}

/** Weighted sum of the inputs plus the bias: sum(w[i] * x[i]) + b. */
export function neuron(x: readonly number[], w: readonly number[], b: number): number {
	let z = b;
	for (let i = 0; i < x.length; i++) z += (w[i] ?? 0) * x[i];
	return z;
}
