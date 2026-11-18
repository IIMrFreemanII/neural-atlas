import type { Dataset } from './datasets.ts';

export interface Logistic {
	w: [number, number];
	b: number;
}

const sigmoid = (z: number) => 1 / (1 + Math.exp(-z));

export function logisticProb(m: Logistic, x0: number, x1: number): number {
	return sigmoid(m.w[0] * x0 + m.w[1] * x1 + m.b);
}

/** Mean binary cross-entropy, probabilities clamped to [1e-7, 1 - 1e-7]. */
export function logisticLoss(m: Logistic, data: Dataset): number {
	let sum = 0;
	for (let k = 0; k < data.n; k++) {
		const p = Math.min(1 - 1e-7, Math.max(1e-7, logisticProb(m, data.x[2 * k], data.x[2 * k + 1])));
		sum -= data.y[k] > 0.5 ? Math.log(p) : Math.log(1 - p);
	}
	return sum / Math.max(1, data.n);
}

/** One full-batch gradient-descent step on mean binary cross-entropy. Mutates m. Returns the loss before the update. */
export function logisticStep(m: Logistic, data: Dataset, lr: number): number {
	const loss = logisticLoss(m, data);
	let g0 = 0;
	let g1 = 0;
	let gb = 0;
	for (let k = 0; k < data.n; k++) {
		const x0 = data.x[2 * k];
		const x1 = data.x[2 * k + 1];
		const e = logisticProb(m, x0, x1) - data.y[k];
		g0 += e * x0;
		g1 += e * x1;
		gb += e;
	}
	const n = Math.max(1, data.n);
	m.w[0] -= (lr * g0) / n;
	m.w[1] -= (lr * g1) / n;
	m.b -= (lr * gb) / n;
	return loss;
}
