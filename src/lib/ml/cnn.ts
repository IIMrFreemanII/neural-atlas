/**
 * A small convolutional network for 16 x 16 images, with a hand-written backward pass.
 *   conv1 (4 kernels 3x3) + ReLU -> max pool 2x2 -> conv2 (8 kernels 3x3x4) + ReLU -> global max pool -> dense -> softmax
 * Pure TypeScript, no DOM. Sums are accumulated in doubles; traces are stored in `Float32Array`s.
 */
import { gaussian, mulberry32 } from './rng.ts';

export const IN = 16;
export const C1 = 4;
export const C2 = 8;
export const CLASSES = 4;

const K = 3;
const M1 = IN - K + 1; // 14, side of the conv1 maps
const P1 = M1 / 2; // 7, side of the pooled maps
const M2 = P1 - K + 1; // 5, side of the conv2 maps

export interface CnnWeights {
	/** (kernel k, row i, col j) -> k * 9 + i * 3 + j */
	w1: Float32Array;
	b1: Float32Array;
	/** (kernel k, input map c, row i, col j) -> ((k * C1 + c) * 3 + i) * 3 + j */
	w2: Float32Array;
	b2: Float32Array;
	/** (class, input) -> class * C2 + input */
	w3: Float32Array;
	b3: Float32Array;
}

export interface CnnTrace {
	input: Float32Array;
	/** C1 * 14 * 14, before ReLU */
	z1: Float32Array;
	a1: Float32Array;
	/** C1 * 7 * 7 */
	p1: Float32Array;
	/** Index into a1 of each pooled maximum. */
	p1Arg: Int32Array;
	/** C2 * 5 * 5, before ReLU */
	z2: Float32Array;
	a2: Float32Array;
	/** C2 global maxima */
	g: Float32Array;
	/** Index into a2 of each global maximum. */
	gArg: Int32Array;
	logits: Float32Array;
	probs: Float32Array;
}

/** He-style Gaussian init (std = sqrt(2 / fan_in)), zero biases. */
export function initCnn(seed: number): CnnWeights {
	const rand = mulberry32(seed);
	const fill = (n: number, fanIn: number) => {
		const a = new Float32Array(n);
		const s = Math.sqrt(2 / fanIn);
		for (let i = 0; i < n; i++) a[i] = gaussian(rand) * s;
		return a;
	};
	return {
		w1: fill(C1 * 9, 9),
		b1: new Float32Array(C1),
		w2: fill(C2 * C1 * 9, C1 * 9),
		b2: new Float32Array(C2),
		w3: fill(CLASSES * C2, C2),
		b3: new Float32Array(CLASSES)
	};
}

type FloatArray = typeof Float32Array | typeof Float64Array;

/**
 * Forward pass. `Arr` picks the storage precision of the trace; it exists so the tests can run the whole pass in
 * doubles (pass Float64Array weights and Float64Array here). Everything else should use the default.
 */
export function forward(w: CnnWeights, input: Float32Array, Arr: FloatArray = Float32Array): CnnTrace {
	const z1 = new Arr(C1 * M1 * M1) as Float32Array;
	const a1 = new Arr(C1 * M1 * M1) as Float32Array;
	for (let k = 0; k < C1; k++) {
		for (let y = 0; y < M1; y++) {
			for (let x = 0; x < M1; x++) {
				let s = w.b1[k];
				for (let i = 0; i < K; i++) for (let j = 0; j < K; j++) s += w.w1[k * 9 + i * 3 + j] * input[(y + i) * IN + x + j];
				const o = (k * M1 + y) * M1 + x;
				z1[o] = s;
				a1[o] = s > 0 ? s : 0;
			}
		}
	}
	const p1 = new Arr(C1 * P1 * P1) as Float32Array;
	const p1Arg = new Int32Array(C1 * P1 * P1);
	for (let c = 0; c < C1; c++) {
		for (let y = 0; y < P1; y++) {
			for (let x = 0; x < P1; x++) {
				let best = -1;
				let bv = -Infinity;
				for (let i = 0; i < 2; i++) {
					for (let j = 0; j < 2; j++) {
						const idx = (c * M1 + 2 * y + i) * M1 + 2 * x + j;
						if (a1[idx] > bv) {
							bv = a1[idx];
							best = idx;
						}
					}
				}
				const o = (c * P1 + y) * P1 + x;
				p1[o] = bv;
				p1Arg[o] = best;
			}
		}
	}
	const z2 = new Arr(C2 * M2 * M2) as Float32Array;
	const a2 = new Arr(C2 * M2 * M2) as Float32Array;
	for (let k = 0; k < C2; k++) {
		for (let y = 0; y < M2; y++) {
			for (let x = 0; x < M2; x++) {
				let s = w.b2[k];
				for (let c = 0; c < C1; c++) {
					for (let i = 0; i < K; i++) {
						for (let j = 0; j < K; j++) {
							s += w.w2[((k * C1 + c) * 3 + i) * 3 + j] * p1[(c * P1 + y + i) * P1 + x + j];
						}
					}
				}
				const o = (k * M2 + y) * M2 + x;
				z2[o] = s;
				a2[o] = s > 0 ? s : 0;
			}
		}
	}
	const g = new Arr(C2) as Float32Array;
	const gArg = new Int32Array(C2);
	for (let k = 0; k < C2; k++) {
		let best = k * M2 * M2;
		for (let i = best + 1; i < (k + 1) * M2 * M2; i++) if (a2[i] > a2[best]) best = i;
		g[k] = a2[best];
		gArg[k] = best;
	}
	const logits = new Arr(CLASSES) as Float32Array;
	for (let c = 0; c < CLASSES; c++) {
		let s = w.b3[c];
		for (let k = 0; k < C2; k++) s += w.w3[c * C2 + k] * g[k];
		logits[c] = s;
	}
	let m = -Infinity;
	for (let c = 0; c < CLASSES; c++) if (logits[c] > m) m = logits[c];
	const probs = new Arr(CLASSES) as Float32Array;
	let sum = 0;
	for (let c = 0; c < CLASSES; c++) {
		const e = Math.exp(logits[c] - m);
		probs[c] = e;
		sum += e;
	}
	for (let c = 0; c < CLASSES; c++) probs[c] /= sum;
	return { input, z1, a1, p1, p1Arg, z2, a2, g, gArg, logits, probs };
}

/** Cross-entropy loss -log(probs[label]). */
export function loss(t: CnnTrace, label: number): number {
	return -Math.log(Math.max(t.probs[label], 1e-12));
}

/** Gradients of `loss` with respect to every weight, same shapes as CnnWeights (always Float32Array). */
export function backward(w: CnnWeights, t: CnnTrace, label: number): CnnWeights {
	const gr: CnnWeights = {
		w1: new Float32Array(w.w1.length),
		b1: new Float32Array(C1),
		w2: new Float32Array(w.w2.length),
		b2: new Float32Array(C2),
		w3: new Float32Array(w.w3.length),
		b3: new Float32Array(CLASSES)
	};
	// Accumulate in doubles, copy out at the end.
	const dw1 = new Float64Array(w.w1.length);
	const db1 = new Float64Array(C1);
	const dw2 = new Float64Array(w.w2.length);
	const db2 = new Float64Array(C2);

	const dl = new Float64Array(CLASSES);
	for (let c = 0; c < CLASSES; c++) dl[c] = t.probs[c] - (c === label ? 1 : 0);
	const dg = new Float64Array(C2);
	for (let c = 0; c < CLASSES; c++) {
		gr.b3[c] = dl[c];
		for (let k = 0; k < C2; k++) {
			gr.w3[c * C2 + k] = dl[c] * t.g[k];
			dg[k] += w.w3[c * C2 + k] * dl[c];
		}
	}
	// Through the global max, then ReLU, into conv2.
	const dz2 = new Float64Array(C2 * M2 * M2);
	for (let k = 0; k < C2; k++) {
		const idx = t.gArg[k];
		dz2[idx] = t.z2[idx] > 0 ? dg[k] : 0;
	}
	const dp1 = new Float64Array(C1 * P1 * P1);
	for (let k = 0; k < C2; k++) {
		for (let y = 0; y < M2; y++) {
			for (let x = 0; x < M2; x++) {
				const d = dz2[(k * M2 + y) * M2 + x];
				if (d === 0) continue;
				db2[k] += d;
				for (let c = 0; c < C1; c++) {
					for (let i = 0; i < K; i++) {
						for (let j = 0; j < K; j++) {
							const wi = ((k * C1 + c) * 3 + i) * 3 + j;
							const pi = (c * P1 + y + i) * P1 + x + j;
							dw2[wi] += d * t.p1[pi];
							dp1[pi] += d * w.w2[wi];
						}
					}
				}
			}
		}
	}
	// Through the max pool (to the argmax cell), then ReLU, into conv1.
	const dz1 = new Float64Array(C1 * M1 * M1);
	for (let i = 0; i < dp1.length; i++) {
		const idx = t.p1Arg[i];
		if (t.z1[idx] > 0) dz1[idx] += dp1[i];
	}
	for (let k = 0; k < C1; k++) {
		for (let y = 0; y < M1; y++) {
			for (let x = 0; x < M1; x++) {
				const d = dz1[(k * M1 + y) * M1 + x];
				if (d === 0) continue;
				db1[k] += d;
				for (let i = 0; i < K; i++) for (let j = 0; j < K; j++) dw1[k * 9 + i * 3 + j] += d * t.input[(y + i) * IN + x + j];
			}
		}
	}
	gr.w1.set(dw1);
	gr.b1.set(db1);
	gr.w2.set(dw2);
	gr.b2.set(db2);
	return gr;
}

/**
 * The square region of the input image that one cell can see, as inclusive pixel bounds.
 * 'a1' (14 x 14 maps): 3 x 3. 'p1' (7 x 7): 4 x 4. 'a2' (5 x 5): 8 x 8.
 */
export function receptiveField(
	layer: 'a1' | 'p1' | 'a2',
	row: number,
	col: number
): { r0: number; c0: number; r1: number; c1: number } {
	const [step, size] = layer === 'a1' ? [1, 3] : layer === 'p1' ? [2, 4] : [2, 8];
	return { r0: row * step, c0: col * step, r1: row * step + size - 1, c1: col * step + size - 1 };
}
