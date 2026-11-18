import { attention, type AttentionResult } from './attention.ts';
import { gaussian, mulberry32 } from './rng.ts';
import { addScaled, matVec, type Mat, type Vec } from './vec.ts';

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

/** Sinusoidal position vectors, n x d: even index 2i = sin(p / 10000^(2i/d)), odd = cos of the same angle. */
export function positional(n: number, d: number): Mat {
	const out: Mat = [];
	for (let p = 0; p < n; p++) {
		const row: Vec = new Array<number>(d);
		for (let k = 0; k < d; k++) {
			const i = Math.floor(k / 2);
			const angle = p / Math.pow(10000, (2 * i) / d);
			row[k] = k % 2 === 0 ? Math.sin(angle) : Math.cos(angle);
		}
		out.push(row);
	}
	return out;
}

export interface BlockWeights {
	/** d x d */
	wq: Mat;
	wk: Mat;
	wv: Mat;
	wo: Mat;
	/** 4d x d, 4d */
	w1: Mat;
	b1: Vec;
	/** d x 4d, d */
	w2: Mat;
	b2: Vec;
}

function randomMat(rows: number, cols: number, sd: number, rand: () => number): Mat {
	const m: Mat = [];
	for (let r = 0; r < rows; r++) {
		const row: Vec = new Array<number>(cols);
		for (let c = 0; c < cols; c++) row[c] = sd * gaussian(rand);
		m.push(row);
	}
	return m;
}

/** Seeded Gaussian weights: attention matrices sd 1 / sqrt(d), w1 sd 1 / sqrt(d), w2 sd 0.5 / sqrt(4d), biases 0. */
export function randomBlock(d: number, seed: number): BlockWeights {
	const rand = mulberry32(seed);
	const sd = 1 / Math.sqrt(d);
	const wq = randomMat(d, d, sd, rand);
	const wk = randomMat(d, d, sd, rand);
	const wv = randomMat(d, d, sd, rand);
	const wo = randomMat(d, d, sd, rand);
	const w1 = randomMat(4 * d, d, sd, rand);
	const w2 = randomMat(d, 4 * d, 0.5 / Math.sqrt(4 * d), rand);
	return { wq, wk, wv, wo, w1, b1: new Array<number>(4 * d).fill(0), w2, b2: new Array<number>(d).fill(0) };
}

export interface BlockTrace {
	/** input, n x d */
	x: Mat;
	/** layerNorm of each row of x */
	norm1: Mat;
	/** attention(wq norm1, wk norm1, wv norm1, { causal }) */
	attn: AttentionResult;
	/** wo applied to each row of attn.out: the update attention adds */
	attnOut: Mat;
	/** x + attnOut */
	mid: Mat;
	/** layerNorm of each row of mid */
	norm2: Mat;
	/** gelu(w1 norm2 + b1), n x 4d */
	hidden: Mat;
	/** w2 hidden + b2: the update the feed-forward layer adds */
	mlpOut: Mat;
	/** mid + mlpOut */
	out: Mat;
}

/** One pre-norm transformer block with a single attention head. causal defaults to true. */
export function runBlock(x: Mat, w: BlockWeights, causal = true): BlockTrace {
	const norm1 = x.map((r) => layerNorm(r));
	const q = norm1.map((r) => matVec(w.wq, r));
	const k = norm1.map((r) => matVec(w.wk, r));
	const v = norm1.map((r) => matVec(w.wv, r));
	const attn = attention(q, k, v, { causal });
	const attnOut = attn.out.map((r) => matVec(w.wo, r));
	const mid = x.map((r, i) => addScaled(r, attnOut[i]));
	const norm2 = mid.map((r) => layerNorm(r));
	const hidden = norm2.map((r) => addScaled(matVec(w.w1, r), w.b1).map(gelu));
	const mlpOut = hidden.map((r) => addScaled(matVec(w.w2, r), w.b2));
	const out = mid.map((r, i) => addScaled(r, mlpOut[i]));
	return { x, norm1, attn, attnOut, mid, norm2, hidden, mlpOut, out };
}
