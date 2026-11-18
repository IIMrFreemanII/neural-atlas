import type { Vec } from './vec.ts';

/** (x - mean) / sqrt(variance + eps), no learned gain or bias. */
export function layerNorm(x: Vec, eps = 1e-5): Vec {
	const n = x.length;
	if (n === 0) return [];
	let mean = 0;
	for (const v of x) mean += v;
	mean /= n;
	let variance = 0;
	for (const v of x) variance += (v - mean) * (v - mean);
	variance /= n;
	const inv = 1 / Math.sqrt(variance + eps);
	return x.map((v) => (v - mean) * inv);
}

/** GELU, tanh approximation. */
export function gelu(x: number): number {
	return 0.5 * x * (1 + Math.tanh(Math.sqrt(2 / Math.PI) * (x + 0.044715 * x * x * x)));
}
