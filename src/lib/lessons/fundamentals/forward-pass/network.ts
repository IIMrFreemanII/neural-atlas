import { createMLP, forward, type MLP } from '../../../ml/mlp.ts';
import { mulberry32 } from '../../../ml/rng.ts';

export const SIZES = [2, 3, 3, 1];
export const SEED = 36;

/** The lesson's fixed network. Biases are given small non-zero values so that they are visible in the matrix view. */
export function buildNet(seed = SEED): MLP {
	const net = createMLP(SIZES, 'tanh', 'sigmoid', seed);
	const rand = mulberry32(seed + 1000);
	for (const b of net.b) for (let k = 0; k < b.length; k++) b[k] = (rand() * 2 - 1) * 0.4;
	return net;
}

export function outputRange(net: MLP): { min: number; max: number } {
	let min = Infinity;
	let max = -Infinity;
	for (let i = 0; i <= 20; i++) {
		for (let j = 0; j <= 20; j++) {
			const t = forward(net, [-1 + i * 0.1, -1 + j * 0.1]);
			const y = t.a[t.a.length - 1][0];
			min = Math.min(min, y);
			max = Math.max(max, y);
		}
	}
	return { min, max };
}
