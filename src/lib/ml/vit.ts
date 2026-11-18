import { gelu } from './block.ts';
import { gaussian, mulberry32 } from './rng.ts';

/** A tiny vision transformer for 16 x 16 drawings: 4 x 4 patches, a class token, pre-norm blocks. Forward pass only. */
export const IMAGE = 16;
export const PATCH = 4;
/** Patches per side. */
export const GRID = 4;
export const PATCHES = 16;
/** Index 0 is the class token, 1 + p is patch p. */
export const TOKENS = 17;
export const PATCH_PIXELS = 16;
export const DIM = 32;
export const HEADS = 2;
export const HEAD_DIM = 16;
export const HIDDEN = 64;
export const LAYERS = 2;
export const CLASSES = 4;

const LN_EPS = 1e-5;

/** Every matrix is row-major [out][in], like a torch Linear weight. */
export interface VitLayer {
	g1: Float32Array;
	b1: Float32Array;
	Wq: Float32Array;
	bq: Float32Array;
	Wk: Float32Array;
	bk: Float32Array;
	Wv: Float32Array;
	bv: Float32Array;
	Wo: Float32Array;
	bo: Float32Array;
	g2: Float32Array;
	b2: Float32Array;
	W1: Float32Array;
	b1m: Float32Array;
	W2: Float32Array;
	b2m: Float32Array;
}

export interface VitWeights {
	E: Float32Array;
	bE: Float32Array;
	cls: Float32Array;
	pos: Float32Array;
	layers: VitLayer[];
	gf: Float32Array;
	bf: Float32Array;
	Wh: Float32Array;
	bh: Float32Array;
}

/** Sizes of one block's tensors, in packing order. */
const LAYER_FIELDS: [keyof VitLayer, number][] = [
	['g1', DIM],
	['b1', DIM],
	['Wq', DIM * DIM],
	['bq', DIM],
	['Wk', DIM * DIM],
	['bk', DIM],
	['Wv', DIM * DIM],
	['bv', DIM],
	['Wo', DIM * DIM],
	['bo', DIM],
	['g2', DIM],
	['b2', DIM],
	['W1', HIDDEN * DIM],
	['b1m', HIDDEN],
	['W2', DIM * HIDDEN],
	['b2m', DIM]
];

export const PARAM_COUNT =
	DIM * PATCH_PIXELS +
	DIM +
	DIM +
	TOKENS * DIM +
	LAYERS * LAYER_FIELDS.reduce((s, [, n]) => s + n, 0) +
	DIM +
	DIM +
	CLASSES * DIM +
	CLASSES;

function zeroLayer(): VitLayer {
	const l = {} as Record<string, Float32Array>;
	for (const [name, n] of LAYER_FIELDS) l[name] = new Float32Array(n);
	return l as unknown as VitLayer;
}

function zeroWeights(): VitWeights {
	return {
		E: new Float32Array(DIM * PATCH_PIXELS),
		bE: new Float32Array(DIM),
		cls: new Float32Array(DIM),
		pos: new Float32Array(TOKENS * DIM),
		layers: Array.from({ length: LAYERS }, zeroLayer),
		gf: new Float32Array(DIM),
		bf: new Float32Array(DIM),
		Wh: new Float32Array(CLASSES * DIM),
		bh: new Float32Array(CLASSES)
	};
}

/** All tensors in the packing order: E, bE, cls, pos, each layer's, then gf, bf, Wh, bh. */
function tensors(w: VitWeights): Float32Array[] {
	const out = [w.E, w.bE, w.cls, w.pos];
	for (const l of w.layers) for (const [name] of LAYER_FIELDS) out.push(l[name]);
	out.push(w.gf, w.bf, w.Wh, w.bh);
	return out;
}

/** The inverse of packVit. Throws if the length is wrong. */
export function unpackVit(b64: string): VitWeights {
	const bin = atob(b64);
	if (bin.length !== PARAM_COUNT * 4) throw new Error(`ViT weights need ${PARAM_COUNT * 4} bytes, got ${bin.length}`);
	const bytes = new Uint8Array(bin.length);
	for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
	// Little-endian float32 regardless of the machine.
	const view = new DataView(bytes.buffer);
	const w = zeroWeights();
	let o = 0;
	for (const t of tensors(w)) {
		for (let i = 0; i < t.length; i++, o += 4) t[i] = view.getFloat32(o, true);
	}
	return w;
}

/** Little-endian float32 base64, in the order the training script writes. */
export function packVit(w: VitWeights): string {
	const bytes = new Uint8Array(PARAM_COUNT * 4);
	const view = new DataView(bytes.buffer);
	let o = 0;
	for (const t of tensors(w)) {
		for (let i = 0; i < t.length; i++, o += 4) view.setFloat32(o, t[i], true);
	}
	let bin = '';
	for (let i = 0; i < bytes.length; i++) bin += String.fromCharCode(bytes[i]);
	return btoa(bin);
}

/** Seeded random weights (for tests): small normal matrices, gains 1, biases 0, cls and pos std 0.02. */
export function initVit(seed: number): VitWeights {
	const rand = mulberry32(seed);
	const fill = (a: Float32Array, std: number) => {
		for (let i = 0; i < a.length; i++) a[i] = gaussian(rand) * std;
	};
	const w = zeroWeights();
	fill(w.E, 1 / Math.sqrt(PATCH_PIXELS));
	fill(w.cls, 0.02);
	fill(w.pos, 0.02);
	for (const l of w.layers) {
		l.g1.fill(1);
		l.g2.fill(1);
		for (const k of ['Wq', 'Wk', 'Wv', 'Wo'] as const) fill(l[k], 1 / Math.sqrt(DIM));
		fill(l.W1, 1 / Math.sqrt(DIM));
		fill(l.W2, 1 / Math.sqrt(HIDDEN));
	}
	w.gf.fill(1);
	fill(w.Wh, 1 / Math.sqrt(DIM));
	return w;
}

/** The 16 pixels of patch p (row-major within the patch). */
export function patchPixels(image: ArrayLike<number>, p: number): Float32Array {
	const pr = Math.floor(p / GRID);
	const pc = p % GRID;
	const out = new Float32Array(PATCH_PIXELS);
	for (let r = 0; r < PATCH; r++) {
		for (let c = 0; c < PATCH; c++) out[r * PATCH + c] = image[(pr * PATCH + r) * IMAGE + pc * PATCH + c];
	}
	return out;
}

export interface VitLayerTrace {
	/** Tokens entering the block, TOKENS x DIM. */
	input: Float32Array[];
	/** attn[head][i][t]: how much token i attends to token t; each row sums to 1. */
	attn: number[][][];
	/** Tokens after the attention update (x + attention), TOKENS x DIM. */
	mid: Float32Array[];
	/** Tokens leaving the block, TOKENS x DIM. */
	output: Float32Array[];
}

export interface VitTrace {
	/** e_p for each patch, before the position is added, PATCHES x DIM. */
	patchEmbed: Float32Array[];
	/** Tokens entering block 1 (class token first), TOKENS x DIM. */
	tokens: Float32Array[];
	layers: VitLayerTrace[];
	/** LNf of the final class token. */
	cls: Float32Array;
	logits: Float32Array;
	probs: Float32Array;
}

export interface VitOptions {
	/** Leave out the position vectors (for showing what attention cannot see). Default false. */
	noPositions?: boolean;
}

/** out = W x + b for a row-major [rows][x.length] matrix. */
function affine(W: Float32Array, b: Float32Array, x: ArrayLike<number>, rows: number): Float32Array {
	const n = x.length;
	const out = new Float32Array(rows);
	for (let j = 0; j < rows; j++) {
		let s = b[j];
		const base = j * n;
		for (let i = 0; i < n; i++) s += W[base + i] * x[i];
		out[j] = s;
	}
	return out;
}

/** LayerNorm with learned gain and bias (biased variance, like torch nn.LayerNorm). */
function norm(x: ArrayLike<number>, g: Float32Array, b: Float32Array): Float32Array {
	const n = x.length;
	let mean = 0;
	for (let i = 0; i < n; i++) mean += x[i];
	mean /= n;
	let variance = 0;
	for (let i = 0; i < n; i++) variance += (x[i] - mean) * (x[i] - mean);
	variance /= n;
	const inv = 1 / Math.sqrt(variance + LN_EPS);
	const out = new Float32Array(n);
	for (let i = 0; i < n; i++) out[i] = (x[i] - mean) * inv * g[i] + b[i];
	return out;
}

function softmax(z: ArrayLike<number>): number[] {
	let max = -Infinity;
	for (let i = 0; i < z.length; i++) if (z[i] > max) max = z[i];
	const e = Array.from(z, (v) => Math.exp(v - max));
	let sum = 0;
	for (const v of e) sum += v;
	return e.map((v) => v / sum);
}

function block(l: VitLayer, x: Float32Array[]): VitLayerTrace {
	const h = x.map((t) => norm(t, l.g1, l.b1));
	const q = h.map((t) => affine(l.Wq, l.bq, t, DIM));
	const k = h.map((t) => affine(l.Wk, l.bk, t, DIM));
	const v = h.map((t) => affine(l.Wv, l.bv, t, DIM));
	const scale = 1 / Math.sqrt(HEAD_DIM);

	const attn: number[][][] = Array.from({ length: HEADS }, () => []);
	const a = x.map(() => new Float32Array(DIM));
	for (let j = 0; j < HEADS; j++) {
		const lo = j * HEAD_DIM;
		for (let i = 0; i < TOKENS; i++) {
			const scores = new Array<number>(TOKENS);
			for (let t = 0; t < TOKENS; t++) {
				let s = 0;
				for (let d = 0; d < HEAD_DIM; d++) s += q[i][lo + d] * k[t][lo + d];
				scores[t] = s * scale;
			}
			const weights = softmax(scores);
			attn[j].push(weights);
			for (let t = 0; t < TOKENS; t++) {
				for (let d = 0; d < HEAD_DIM; d++) a[i][lo + d] += weights[t] * v[t][lo + d];
			}
		}
	}
	const mid = x.map((t, i) => {
		const o = affine(l.Wo, l.bo, a[i], DIM);
		for (let d = 0; d < DIM; d++) o[d] += t[d];
		return o;
	});

	const output = mid.map((t) => {
		const hidden = affine(l.W1, l.b1m, norm(t, l.g2, l.b2), HIDDEN);
		for (let d = 0; d < HIDDEN; d++) hidden[d] = gelu(hidden[d]);
		const o = affine(l.W2, l.b2m, hidden, DIM);
		for (let d = 0; d < DIM; d++) o[d] += t[d];
		return o;
	});
	return { input: x, attn, mid, output };
}

export function vitForward(w: VitWeights, image: ArrayLike<number>, opts: VitOptions = {}): VitTrace {
	const usePos = !opts.noPositions;
	const withPos = (e: ArrayLike<number>, index: number) => {
		const out = Float32Array.from(e);
		if (usePos) for (let d = 0; d < DIM; d++) out[d] += w.pos[index * DIM + d];
		return out;
	};

	const patchEmbed: Float32Array[] = [];
	for (let p = 0; p < PATCHES; p++) patchEmbed.push(affine(w.E, w.bE, patchPixels(image, p), DIM));
	const tokens = [withPos(w.cls, 0), ...patchEmbed.map((e, p) => withPos(e, 1 + p))];

	const layers: VitLayerTrace[] = [];
	let x: Float32Array[] = tokens;
	for (const l of w.layers) {
		const t = block(l, x);
		layers.push(t);
		x = t.output;
	}
	const cls = norm(x[0], w.gf, w.bf);
	const logits = affine(w.Wh, w.bh, cls, CLASSES);
	return { patchEmbed, tokens, layers, cls, logits, probs: Float32Array.from(softmax(logits)) };
}

/** Cosine similarity between the position vectors of patches a and b (0..15). */
export function positionSimilarity(w: VitWeights, a: number, b: number): number {
	let ab = 0;
	let aa = 0;
	let bb = 0;
	for (let d = 0; d < DIM; d++) {
		const x = w.pos[(1 + a) * DIM + d];
		const y = w.pos[(1 + b) * DIM + d];
		ab += x * y;
		aa += x * x;
		bb += y * y;
	}
	return aa === 0 || bb === 0 ? 0 : ab / Math.sqrt(aa * bb);
}
